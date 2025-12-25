"""Headless runtime for Reachy Mini Conversation App.

Main entrypoint for running the conversation system without UI.
Orchestrates wake word detection, state machine, and conversation handling.
"""

import os
import sys
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

from reachy_mini_conversation_app.headless.state_machine import StateMachine, ConversationState
from reachy_mini_conversation_app.headless.wake_word import WakeWordEngine
from reachy_mini_conversation_app.headless.audio_router import AudioRouter, OPENAI_SAMPLE_RATE, NATIVE_SAMPLE_RATE
from reachy_mini_conversation_app.headless.transcription import TranscriptionService, ConversationRecorder
from reachy_mini_conversation_app.memory.manager import MemoryManager


logger = logging.getLogger(__name__)


class HeadlessRuntime:
    """Main runtime for headless operation.

    Coordinates all components for fully autonomous conversation:
    - Wake word detection (offline)
    - State machine (SLEEPING/AWAKE)
    - Audio routing
    - OpenAI conversation
    - Transcription and storage
    """

    def __init__(self) -> None:
        """Initialize the headless runtime."""
        # Load environment variables
        load_dotenv()

        # Configuration from environment
        self._silence_timeout = float(os.getenv("SILENCE_TIMEOUT_SECONDS", "120"))
        self._max_awake_seconds = float(os.getenv("MAX_AWAKE_SECONDS", "1800"))
        self._min_sleep_seconds = float(os.getenv("MIN_SLEEP_SECONDS", "5"))
        self._wake_word_threshold = float(os.getenv("WAKE_WORD_CONFIDENCE_THRESHOLD", "0.5"))
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        # VAD (Voice Activity Detection) threshold for speech detection
        # Typical values: 500-2000 for int16 audio, lower = more sensitive
        self._speech_energy_threshold = float(os.getenv("SPEECH_ENERGY_THRESHOLD", "1000"))
        # Pre-roll buffer seconds
        self._preroll_seconds = float(os.getenv("PREROLL_SECONDS", "2.0"))

        # Audio device configuration
        self._audio_device = os.getenv("AUDIO_INPUT_DEVICE", None)  # None = default device
        self._audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", str(NATIVE_SAMPLE_RATE)))
        self._audio_channels = 1  # Mono
        self._audio_block_size = 1024  # Samples per block

        # Components (initialized lazily)
        self._state_machine: Optional[StateMachine] = None
        self._wake_word_engine: Optional[WakeWordEngine] = None
        self._audio_router: Optional[AudioRouter] = None
        self._transcription_service: Optional[TranscriptionService] = None
        self._recorder: Optional[ConversationRecorder] = None
        self._memory_manager: Optional[MemoryManager] = None

        # Runtime state
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info("HeadlessRuntime created")

    def _resample_audio(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio samples (int16).
            from_rate: Input sample rate.
            to_rate: Output sample rate.

        Returns:
            Resampled audio samples (int16).
        """
        if from_rate == to_rate or len(audio) == 0:
            return audio

        if SCIPY_AVAILABLE:
            # Use scipy for quality resampling
            num_samples = int(len(audio) * to_rate / from_rate)
            resampled = scipy_signal.resample(audio.astype(np.float32), num_samples)
            return np.clip(resampled, -32768, 32767).astype(np.int16)
        else:
            # Linear interpolation fallback (better than nearest-neighbor)
            # Uses np.interp for proper anti-aliasing
            new_length = int(len(audio) * to_rate / from_rate)
            return np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio.astype(np.float32),
            ).astype(np.int16)

    def _on_state_change(self, old_state: ConversationState, new_state: ConversationState) -> None:
        """Handle state changes.

        Args:
            old_state: Previous state.
            new_state: New state.
        """
        if new_state == ConversationState.AWAKE:
            # Get pre-roll audio buffer (audio captured before wake word)
            # This prevents clipping the start of user's speech
            preroll_audio = None
            if self._audio_router:
                preroll_audio = self._audio_router.get_preroll_buffer()
                self._audio_router.clear_preroll_buffer()
                if len(preroll_audio) > 0:
                    duration_ms = len(preroll_audio) / 24  # 24kHz = 24 samples/ms
                    logger.info(f"Pre-roll buffer captured: {duration_ms:.0f}ms of audio")

            # Enable conversation audio routing
            if self._audio_router:
                self._audio_router.enable_conversation()

            # Start recording
            if self._recorder:
                self._recorder.start_session()
                # Include pre-roll audio at the start of recording to prevent clipping
                if preroll_audio is not None and len(preroll_audio) > 0:
                    # Pre-roll is at 24kHz (OPENAI_SAMPLE_RATE), recorder is at native rate (48kHz)
                    # Resample to match recorder's sample rate
                    preroll_resampled = self._resample_audio(
                        preroll_audio,
                        from_rate=OPENAI_SAMPLE_RATE,
                        to_rate=NATIVE_SAMPLE_RATE,
                    )
                    # Add pre-roll audio at the start of the recording
                    self._recorder.add_audio(preroll_resampled)
                    logger.info(
                        f"Pre-roll audio added to recording: "
                        f"{len(preroll_resampled)} samples ({len(preroll_resampled) / NATIVE_SAMPLE_RATE * 1000:.0f}ms)"
                    )

            logger.info("System AWAKE - ready for conversation")

        elif new_state == ConversationState.SLEEPING:
            # Disable conversation audio routing
            if self._audio_router:
                self._audio_router.disable_conversation()

            # End recording and queue for transcription
            if self._recorder:
                audio_path = self._recorder.end_session()
                if audio_path and self._transcription_service:
                    self._transcription_service.queue_transcription(audio_path)

            logger.info("System SLEEPING - listening for wake word")

    async def initialize(self) -> bool:
        """Initialize all components.

        Returns:
            True if initialization succeeded.
        """
        logger.info("Initializing HeadlessRuntime...")

        # Initialize memory manager first (needed for wake word logging and transcription)
        self._memory_manager = MemoryManager(openai_api_key=self._openai_api_key)

        # Initialize state machine with full configuration
        self._state_machine = StateMachine(
            silence_timeout_seconds=self._silence_timeout,
            max_awake_seconds=self._max_awake_seconds,
            min_sleep_seconds=self._min_sleep_seconds,
            on_state_change=self._on_state_change,
        )

        # Initialize wake word engine with database for event logging
        self._wake_word_engine = WakeWordEngine(
            confidence_threshold=self._wake_word_threshold,
            on_wake_word=self._on_wake_word_detected,
            database=self._memory_manager.db,
        )
        if not self._wake_word_engine.initialize():
            logger.warning("Wake word engine failed to initialize - will run without wake word")

        # Initialize audio router with configured sample rate and pre-roll duration
        self._audio_router = AudioRouter(
            input_sample_rate=self._audio_sample_rate,
            preroll_seconds=self._preroll_seconds,
        )

        # Initialize transcription service with database for storage
        self._transcription_service = TranscriptionService(
            openai_api_key=self._openai_api_key,
            memory_database=self._memory_manager.db,
        )
        await self._transcription_service.initialize()

        # Initialize recorder
        self._recorder = ConversationRecorder()

        # Initialize the memory manager's async components
        await self._memory_manager.initialize()

        logger.info("HeadlessRuntime initialized successfully")
        return True

    def _on_wake_word_detected(self, confidence: float) -> None:
        """Callback when wake word is detected.

        Args:
            confidence: Detection confidence score.
        """
        logger.info(f"Wake word detected with confidence {confidence:.2f}")
        # Schedule state transition (we're in a sync callback)
        if self._state_machine:
            asyncio.create_task(self._state_machine.on_wake_word_detected(confidence))

    async def _wake_word_loop(self) -> None:
        """Background loop for wake word detection."""
        if not self._wake_word_engine or not self._wake_word_engine.is_initialized:
            logger.warning("Wake word engine not available, skipping wake word loop")
            return

        if not self._audio_router:
            return

        logger.info("Starting wake word detection loop")

        while self._running:
            try:
                # Get audio from router
                audio = await self._audio_router.get_wake_word_audio(timeout=0.1)
                if audio is None:
                    continue

                # Process for wake word (only in SLEEPING state)
                if self._state_machine and self._state_machine.is_sleeping:
                    await self._wake_word_engine.process_audio_async(audio)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in wake word loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Wake word detection loop ended")

    async def _conversation_loop(self) -> None:
        """Background loop for conversation handling."""
        if not self._audio_router:
            return

        logger.info("Starting conversation loop")

        while self._running:
            try:
                # Only process when AWAKE
                if self._state_machine and not self._state_machine.is_awake:
                    await asyncio.sleep(0.1)
                    continue

                # Get audio from router
                audio = await self._audio_router.get_conversation_audio(timeout=0.1)
                if audio is None:
                    continue

                # PHASE 4 INTEGRATION POINT: OpenAI Realtime API
                # This is where the OpenAI Realtime WebSocket client will be integrated.
                # The audio (24kHz, int16) will be sent to OpenAI for real-time conversation.

                # Voice Activity Detection (VAD) using RMS energy
                # Only signal speech activity if audio energy exceeds threshold
                # This prevents the silence timer from resetting on every frame
                audio_float = audio.astype(np.float32)
                rms_energy = np.sqrt(np.mean(audio_float ** 2))

                if rms_energy > self._speech_energy_threshold:
                    if self._state_machine:
                        await self._state_machine.on_user_speech()

                # Check for max awake timeout (safety mechanism)
                # Forces return to SLEEPING if AWAKE too long (prevents indefinite active state)
                if self._state_machine:
                    await self._state_machine.check_max_awake_timeout()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Conversation loop ended")

    async def _recorder_loop(self) -> None:
        """Background loop for recording audio."""
        if not self._audio_router or not self._recorder:
            return

        logger.info("Starting recorder loop")

        while self._running:
            try:
                # Get audio from router
                audio = await self._audio_router.get_recorder_audio(timeout=0.1)
                if audio is None:
                    continue

                # Add to recorder if recording
                if self._recorder.is_recording:
                    self._recorder.add_audio(audio)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recorder loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Recorder loop ended")

    async def _audio_capture_loop(self) -> None:
        """Background loop for capturing audio from microphone.

        Uses sounddevice to capture audio and routes it to all consumers
        via the AudioRouter. This is the main audio source for the system.

        Note: sounddevice callbacks run in a C-thread, so we use
        loop.call_soon_threadsafe() to safely insert audio into the asyncio queue.
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available - cannot capture audio")
            logger.error("Install with: pip install sounddevice")
            return

        if not self._audio_router:
            logger.error("AudioRouter not initialized")
            return

        logger.info(
            f"Starting audio capture: device={self._audio_device or 'default'}, "
            f"rate={self._audio_sample_rate}Hz, channels={self._audio_channels}, "
            f"block_size={self._audio_block_size}"
        )

        # Create asyncio queue for audio data
        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)

        # Get event loop reference for thread-safe queue insertion
        # sounddevice callback runs in a C-thread, so we need call_soon_threadsafe
        loop = asyncio.get_running_loop()

        def audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            """Callback from sounddevice for each audio block.

            IMPORTANT: This runs in a C-thread, not the asyncio event loop.
            We use loop.call_soon_threadsafe() to safely insert into the queue.
            """
            if status:
                logger.warning(f"Audio input status: {status}")

            # Convert float32 to int16
            # sounddevice returns float32 in range [-1.0, 1.0]
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

            # Thread-safe insertion into asyncio queue
            # Using call_soon_threadsafe to avoid race conditions
            def try_put() -> None:
                try:
                    audio_queue.put_nowait(audio_int16)
                except asyncio.QueueFull:
                    pass  # Drop frame if queue is full

            loop.call_soon_threadsafe(try_put)

        # Open audio stream
        try:
            # Convert device name to index if specified
            device_index = None
            if self._audio_device:
                try:
                    device_index = int(self._audio_device)
                except ValueError:
                    # Device specified by name, let sounddevice resolve it
                    device_index = self._audio_device

            stream = sd.InputStream(
                samplerate=self._audio_sample_rate,
                channels=self._audio_channels,
                dtype="float32",
                blocksize=self._audio_block_size,
                device=device_index,
                callback=audio_callback,
            )

            with stream:
                logger.info("Audio capture started")
                while self._running:
                    try:
                        # Get audio from callback queue
                        audio_data = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.5,
                        )
                        # Route to all consumers
                        await self._audio_router.route_audio(audio_data)

                    except asyncio.TimeoutError:
                        # No audio data, check if still running
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error routing audio: {e}")
                        await asyncio.sleep(0.1)

        except sd.PortAudioError as e:
            logger.error(f"PortAudio error: {e}")
            logger.error("Check audio device configuration")
            # List available devices for debugging
            try:
                devices = sd.query_devices()
                logger.info(f"Available audio devices:\n{devices}")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")

        logger.info("Audio capture loop ended")

    def _list_audio_devices(self) -> None:
        """Log available audio devices for debugging."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available")
            return

        try:
            devices = sd.query_devices()
            logger.info(f"Available audio devices:\n{devices}")
        except Exception as e:
            logger.warning(f"Failed to query audio devices: {e}")

    async def run(self) -> None:
        """Run the headless runtime main loop."""
        logger.info("Starting HeadlessRuntime...")

        self._running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Log available audio devices for debugging
        self._list_audio_devices()

        # Start background tasks
        # Audio capture MUST start first - it feeds all other loops
        tasks = [
            asyncio.create_task(self._audio_capture_loop(), name="audio-capture"),
            asyncio.create_task(self._wake_word_loop(), name="wake-word"),
            asyncio.create_task(self._conversation_loop(), name="conversation"),
            asyncio.create_task(self._recorder_loop(), name="recorder"),
        ]

        logger.info("HeadlessRuntime running - listening for wake word...")
        print("\n🎤 Lily is listening... Say 'Hey Lily' to wake up!\n")

        # Wait for shutdown signal
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("HeadlessRuntime stopped")

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Shutdown all components."""
        logger.info("Shutting down HeadlessRuntime...")

        self._running = False
        self._shutdown_event.set()

        # Shutdown components
        if self._state_machine:
            await self._state_machine.shutdown()

        if self._wake_word_engine:
            self._wake_word_engine.shutdown()

        if self._audio_router:
            self._audio_router.shutdown()

        if self._transcription_service:
            await self._transcription_service.shutdown()

        if self._memory_manager:
            self._memory_manager.close()

        logger.info("HeadlessRuntime shutdown complete")


def setup_logging() -> None:
    """Setup logging for headless mode."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


async def async_main() -> None:
    """Async main function."""
    runtime = HeadlessRuntime()

    try:
        if await runtime.initialize():
            await runtime.run()
    finally:
        await runtime.shutdown()


def main() -> None:
    """Main entrypoint for headless mode."""
    setup_logging()

    print("\n" + "=" * 60)
    print("  Reachy Mini Conversation App - Headless Mode")
    print("  Lily is ready to listen!")
    print("=" * 60 + "\n")

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
