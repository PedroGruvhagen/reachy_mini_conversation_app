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

from dotenv import load_dotenv

from reachy_mini_conversation_app.headless.state_machine import StateMachine, ConversationState
from reachy_mini_conversation_app.headless.wake_word import WakeWordEngine
from reachy_mini_conversation_app.headless.audio_router import AudioRouter
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
        self._wake_word_threshold = float(os.getenv("WAKE_WORD_CONFIDENCE_THRESHOLD", "0.5"))
        self._openai_api_key = os.getenv("OPENAI_API_KEY")

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

    def _on_state_change(self, old_state: ConversationState, new_state: ConversationState) -> None:
        """Handle state changes.

        Args:
            old_state: Previous state.
            new_state: New state.
        """
        if new_state == ConversationState.AWAKE:
            # Enable conversation audio routing
            if self._audio_router:
                self._audio_router.enable_conversation()

            # Start recording
            if self._recorder:
                self._recorder.start_session()

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

        # Initialize state machine
        self._state_machine = StateMachine(
            silence_timeout_seconds=self._silence_timeout,
            on_state_change=self._on_state_change,
        )

        # Initialize wake word engine
        self._wake_word_engine = WakeWordEngine(
            confidence_threshold=self._wake_word_threshold,
            on_wake_word=self._on_wake_word_detected,
        )
        if not self._wake_word_engine.initialize():
            logger.warning("Wake word engine failed to initialize - will run without wake word")

        # Initialize audio router
        self._audio_router = AudioRouter()

        # Initialize memory manager first (needed for transcription storage)
        self._memory_manager = MemoryManager(openai_api_key=self._openai_api_key)

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
                # For now, just track speech activity for state machine timing.
                if self._state_machine:
                    await self._state_machine.on_user_speech()

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

    async def run(self) -> None:
        """Run the headless runtime main loop."""
        logger.info("Starting HeadlessRuntime...")

        self._running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._wake_word_loop()),
            asyncio.create_task(self._conversation_loop()),
            asyncio.create_task(self._recorder_loop()),
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
