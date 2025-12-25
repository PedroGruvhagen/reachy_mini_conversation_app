"""Wake word detection using OpenWakeWord.

Provides offline detection of "Hey Lily" wake phrase with configurable
confidence thresholds and cooldown logic.
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    OWWModel = None


logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_COOLDOWN_SECONDS = 10.0
WAKE_WORD_SAMPLE_RATE = 16000  # OpenWakeWord expects 16kHz


class WakeWordEngine:
    """Wake word detection engine using OpenWakeWord.

    Listens for "Hey Lily" (or similar) phrases offline without any API calls.
    Designed for continuous operation with low CPU usage.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        on_wake_word: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Initialize the wake word engine.

        Args:
            model_path: Path to custom wake word model. If None, uses default.
            confidence_threshold: Minimum confidence score to trigger (0.0-1.0).
            cooldown_seconds: Seconds to wait between detections (prevent spam).
            on_wake_word: Callback when wake word detected (receives confidence).
        """
        if not OPENWAKEWORD_AVAILABLE:
            logger.warning("openwakeword not available - wake word detection disabled")
            self._initialized = False
            return

        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._cooldown_seconds = cooldown_seconds
        self._on_wake_word = on_wake_word

        self._model: Optional[OWWModel] = None
        self._initialized = False
        self._last_detection_time: Optional[datetime] = None
        self._detection_count = 0

    def initialize(self) -> bool:
        """Initialize the wake word model.

        Returns:
            True if initialization succeeded.
        """
        if not OPENWAKEWORD_AVAILABLE:
            return False

        try:
            # Download pre-trained models if needed
            openwakeword.utils.download_models()

            # Load model - use default models for now
            # TODO: Train custom "Hey Lily" model
            self._model = OWWModel(
                wakeword_models=["hey_jarvis"],  # Use as proxy until we train "hey_lily"
                inference_framework="onnx",
            )

            self._initialized = True
            logger.info(
                f"WakeWordEngine initialized: threshold={self._confidence_threshold}, "
                f"cooldown={self._cooldown_seconds}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize wake word engine: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def sample_rate(self) -> int:
        """Get the required sample rate for audio input."""
        return WAKE_WORD_SAMPLE_RATE

    @property
    def detection_count(self) -> int:
        """Get total number of wake word detections."""
        return self._detection_count

    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self._last_detection_time is None:
            return False
        elapsed = (datetime.now() - self._last_detection_time).total_seconds()
        return elapsed < self._cooldown_seconds

    def process_audio(self, audio_data: NDArray[np.int16]) -> Optional[float]:
        """Process audio data and check for wake word.

        Args:
            audio_data: Audio samples at 16kHz, int16 format.

        Returns:
            Confidence score if wake word detected, None otherwise.
        """
        if not self._initialized or self._model is None:
            return None

        if self._in_cooldown():
            return None

        try:
            # Convert to float32 for OpenWakeWord
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Run prediction
            predictions = self._model.predict(audio_float)

            # Check all model predictions
            for model_name, confidence in predictions.items():
                if confidence >= self._confidence_threshold:
                    self._last_detection_time = datetime.now()
                    self._detection_count += 1

                    logger.info(
                        f"Wake word detected: {model_name} with confidence {confidence:.3f}"
                    )

                    if self._on_wake_word:
                        try:
                            self._on_wake_word(confidence)
                        except Exception as e:
                            logger.error(f"Error in wake word callback: {e}")

                    return float(confidence)

            return None

        except Exception as e:
            logger.error(f"Error processing audio for wake word: {e}")
            return None

    async def process_audio_async(self, audio_data: NDArray[np.int16]) -> Optional[float]:
        """Async wrapper for process_audio.

        Args:
            audio_data: Audio samples at 16kHz, int16 format.

        Returns:
            Confidence score if wake word detected, None otherwise.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_audio, audio_data)

    def reset_cooldown(self) -> None:
        """Reset the cooldown timer (allow immediate detection)."""
        self._last_detection_time = None

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold.

        Args:
            threshold: New threshold (0.0-1.0).
        """
        self._confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Wake word confidence threshold set to {self._confidence_threshold}")

    def shutdown(self) -> None:
        """Shutdown the wake word engine."""
        logger.info(f"WakeWordEngine shutting down ({self._detection_count} total detections)")
        self._initialized = False
        self._model = None
