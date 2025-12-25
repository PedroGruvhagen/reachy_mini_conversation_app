"""Audio router for fan-out distribution to multiple consumers.

Routes audio from a single capture source to wake word engine,
conversation handler, and background recorder based on current state.
"""

import asyncio
import logging
from typing import Optional
from enum import Enum

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logger = logging.getLogger(__name__)


# Standard sample rates
NATIVE_SAMPLE_RATE = 48000  # Typical device rate
WAKE_WORD_SAMPLE_RATE = 16000  # OpenWakeWord requirement
OPENAI_SAMPLE_RATE = 24000  # OpenAI Realtime requirement


class AudioConsumer(Enum):
    """Audio consumer types."""

    WAKE_WORD = "wake_word"  # 16kHz, always active in SLEEPING
    CONVERSATION = "conversation"  # 24kHz, active in AWAKE
    RECORDER = "recorder"  # Native rate, always active


class AudioRouter:
    """Routes audio to multiple consumers with appropriate resampling.

    Implements fan-out architecture where a single audio capture feeds
    multiple consumers at their required sample rates.
    """

    def __init__(
        self,
        input_sample_rate: int = NATIVE_SAMPLE_RATE,
        max_queue_size: int = 100,
    ) -> None:
        """Initialize the audio router.

        Args:
            input_sample_rate: Sample rate of input audio.
            max_queue_size: Maximum frames in each consumer queue.
        """
        self._input_sample_rate = input_sample_rate
        self._max_queue_size = max_queue_size

        # Consumer queues
        self._wake_word_queue: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._conversation_queue: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._recorder_queue: asyncio.Queue[NDArray[np.int16]] = asyncio.Queue(
            maxsize=max_queue_size
        )

        # Consumer state
        self._wake_word_enabled = True
        self._conversation_enabled = False
        self._recorder_enabled = True

        # Pre-roll buffer for capturing audio before wake word
        self._preroll_buffer: list[NDArray[np.int16]] = []
        self._preroll_seconds = 2.0
        self._preroll_max_frames = int(
            self._preroll_seconds * OPENAI_SAMPLE_RATE / 1024  # Assuming ~1024 samples per frame
        )

        # Stats
        self._frames_routed = 0
        self._frames_dropped = 0

        logger.info(
            f"AudioRouter initialized: input_rate={input_sample_rate}Hz, "
            f"queue_size={max_queue_size}"
        )

    @property
    def frames_routed(self) -> int:
        """Get total frames routed."""
        return self._frames_routed

    @property
    def frames_dropped(self) -> int:
        """Get frames dropped due to full queues."""
        return self._frames_dropped

    def enable_conversation(self) -> None:
        """Enable routing to conversation consumer (AWAKE state)."""
        self._conversation_enabled = True
        logger.debug("Conversation consumer enabled")

    def disable_conversation(self) -> None:
        """Disable routing to conversation consumer (SLEEPING state)."""
        self._conversation_enabled = False
        # Clear the queue
        while not self._conversation_queue.empty():
            try:
                self._conversation_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("Conversation consumer disabled")

    def enable_wake_word(self) -> None:
        """Enable routing to wake word consumer."""
        self._wake_word_enabled = True
        logger.debug("Wake word consumer enabled")

    def disable_wake_word(self) -> None:
        """Disable routing to wake word consumer."""
        self._wake_word_enabled = False
        logger.debug("Wake word consumer disabled")

    def _resample(
        self,
        audio: NDArray[np.int16],
        from_rate: int,
        to_rate: int,
    ) -> NDArray[np.int16]:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio samples.
            from_rate: Input sample rate.
            to_rate: Output sample rate.

        Returns:
            Resampled audio samples.
        """
        if from_rate == to_rate:
            return audio

        if not SCIPY_AVAILABLE:
            # Simple decimation/interpolation fallback
            ratio = to_rate / from_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            return audio[indices]

        # Use scipy for quality resampling
        num_samples = int(len(audio) * to_rate / from_rate)
        resampled = signal.resample(audio.astype(np.float32), num_samples)
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    async def route_audio(self, audio_data: NDArray[np.int16]) -> None:
        """Route audio frame to all enabled consumers.

        Args:
            audio_data: Audio samples at input sample rate.
        """
        self._frames_routed += 1

        # Route to wake word consumer (16kHz)
        if self._wake_word_enabled:
            wake_audio = self._resample(
                audio_data, self._input_sample_rate, WAKE_WORD_SAMPLE_RATE
            )
            try:
                self._wake_word_queue.put_nowait(wake_audio)
            except asyncio.QueueFull:
                self._frames_dropped += 1
                # Drop oldest frame
                try:
                    self._wake_word_queue.get_nowait()
                    self._wake_word_queue.put_nowait(wake_audio)
                except asyncio.QueueEmpty:
                    pass

        # Route to conversation consumer (24kHz)
        if self._conversation_enabled:
            conv_audio = self._resample(
                audio_data, self._input_sample_rate, OPENAI_SAMPLE_RATE
            )
            try:
                self._conversation_queue.put_nowait(conv_audio)
            except asyncio.QueueFull:
                self._frames_dropped += 1
                try:
                    self._conversation_queue.get_nowait()
                    self._conversation_queue.put_nowait(conv_audio)
                except asyncio.QueueEmpty:
                    pass

        # Always update pre-roll buffer with 24kHz audio
        conv_audio_preroll = self._resample(
            audio_data, self._input_sample_rate, OPENAI_SAMPLE_RATE
        )
        self._preroll_buffer.append(conv_audio_preroll)
        if len(self._preroll_buffer) > self._preroll_max_frames:
            self._preroll_buffer.pop(0)

        # Route to recorder consumer (native rate)
        if self._recorder_enabled:
            try:
                self._recorder_queue.put_nowait(audio_data.copy())
            except asyncio.QueueFull:
                self._frames_dropped += 1
                try:
                    self._recorder_queue.get_nowait()
                    self._recorder_queue.put_nowait(audio_data.copy())
                except asyncio.QueueEmpty:
                    pass

    async def get_wake_word_audio(self, timeout: float = 0.1) -> Optional[NDArray[np.int16]]:
        """Get audio frame for wake word processing.

        Args:
            timeout: Seconds to wait for audio.

        Returns:
            Audio frame at 16kHz, or None if timeout.
        """
        try:
            return await asyncio.wait_for(
                self._wake_word_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    async def get_conversation_audio(self, timeout: float = 0.1) -> Optional[NDArray[np.int16]]:
        """Get audio frame for conversation processing.

        Args:
            timeout: Seconds to wait for audio.

        Returns:
            Audio frame at 24kHz, or None if timeout.
        """
        try:
            return await asyncio.wait_for(
                self._conversation_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    async def get_recorder_audio(self, timeout: float = 0.1) -> Optional[NDArray[np.int16]]:
        """Get audio frame for recording.

        Args:
            timeout: Seconds to wait for audio.

        Returns:
            Audio frame at native rate, or None if timeout.
        """
        try:
            return await asyncio.wait_for(
                self._recorder_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    def get_preroll_buffer(self) -> NDArray[np.int16]:
        """Get the pre-roll audio buffer (audio before wake word).

        Returns:
            Concatenated pre-roll audio at 24kHz.
        """
        if not self._preroll_buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._preroll_buffer)

    def clear_preroll_buffer(self) -> None:
        """Clear the pre-roll buffer."""
        self._preroll_buffer.clear()

    def get_stats(self) -> dict[str, int]:
        """Get router statistics.

        Returns:
            Dictionary with routing stats.
        """
        return {
            "frames_routed": self._frames_routed,
            "frames_dropped": self._frames_dropped,
            "wake_word_queue_size": self._wake_word_queue.qsize(),
            "conversation_queue_size": self._conversation_queue.qsize(),
            "recorder_queue_size": self._recorder_queue.qsize(),
            "preroll_frames": len(self._preroll_buffer),
        }

    def shutdown(self) -> None:
        """Shutdown the audio router."""
        logger.info(
            f"AudioRouter shutting down: {self._frames_routed} routed, "
            f"{self._frames_dropped} dropped"
        )
        self._wake_word_enabled = False
        self._conversation_enabled = False
        self._recorder_enabled = False
