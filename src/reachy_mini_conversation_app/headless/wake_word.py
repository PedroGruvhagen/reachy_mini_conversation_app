"""Wake word detection using OpenWakeWord.

Provides offline detection of "Hey Lily" wake phrase with configurable
confidence thresholds and cooldown logic.

Available pre-trained models (as of 2025):
- hey_jarvis (recommended as proxy for "Hey Lily")
- alexa
- hey_mycroft
- hey_rhasspy
- ok_nabu

Custom model training can be done using OpenWakeWord tools.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional
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

# Available pre-trained models
AVAILABLE_MODELS = ["hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "ok_nabu"]
DEFAULT_MODEL = "hey_jarvis"  # Best proxy for "Hey Lily" (similar phonetics)


class WakeWordEngine:
    """Wake word detection engine using OpenWakeWord.

    Listens for "Hey Lily" (or similar) phrases offline without any API calls.
    Designed for continuous operation with low CPU usage.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        on_wake_word: Optional[Callable[[float], None]] = None,
        database: Optional[Any] = None,
    ) -> None:
        """Initialize the wake word engine.

        Args:
            model_name: Name of pre-trained model (hey_jarvis, alexa, etc).
            model_path: Path to custom wake word model. Overrides model_name.
            confidence_threshold: Minimum confidence score to trigger (0.0-1.0).
            cooldown_seconds: Seconds to wait between detections (prevent spam).
            on_wake_word: Callback when wake word detected (receives confidence).
            database: Optional MemoryDatabase for logging wake events.
        """
        if not OPENWAKEWORD_AVAILABLE:
            logger.warning("openwakeword not available - wake word detection disabled")
            self._initialized = False
            return

        # Model configuration (env var > parameter > default)
        self._model_name = (
            os.getenv("WAKE_WORD_MODEL", model_name)
            or DEFAULT_MODEL
        )
        self._model_path = model_path or os.getenv("WAKE_WORD_MODEL_PATH")
        self._confidence_threshold = float(
            os.getenv("WAKE_WORD_CONFIDENCE_THRESHOLD", str(confidence_threshold))
        )
        self._cooldown_seconds = float(
            os.getenv("WAKE_WORD_COOLDOWN_SECONDS", str(cooldown_seconds))
        )
        self._on_wake_word = on_wake_word
        self._database = database

        self._model: Optional[OWWModel] = None
        self._active_model_name: Optional[str] = None
        self._initialized = False
        self._last_detection_time: Optional[datetime] = None
        self._detection_count = 0
        self._false_positive_count = 0  # For tracking rejected detections

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

            # Determine which model(s) to load
            if self._model_path and Path(self._model_path).exists():
                # Use custom model file
                logger.info(f"Loading custom wake word model: {self._model_path}")
                self._model = OWWModel(
                    wakeword_models=[self._model_path],
                    inference_framework="onnx",
                )
                self._active_model_name = Path(self._model_path).stem
            else:
                # Use pre-trained model
                model_to_use = self._model_name
                if model_to_use not in AVAILABLE_MODELS:
                    logger.warning(
                        f"Model '{model_to_use}' not in available models, using '{DEFAULT_MODEL}'"
                    )
                    model_to_use = DEFAULT_MODEL

                logger.info(f"Loading pre-trained wake word model: {model_to_use}")
                self._model = OWWModel(
                    wakeword_models=[model_to_use],
                    inference_framework="onnx",
                )
                self._active_model_name = model_to_use

            self._initialized = True
            logger.info(
                f"WakeWordEngine initialized: model={self._active_model_name}, "
                f"threshold={self._confidence_threshold}, cooldown={self._cooldown_seconds}s"
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

                    # Log to database if available
                    if self._database:
                        try:
                            self._database.add_wake_event(
                                confidence=float(confidence),
                                model_name=model_name,
                                metadata={
                                    "threshold": self._confidence_threshold,
                                    "detection_number": self._detection_count,
                                },
                            )
                        except Exception as db_err:
                            logger.debug(f"Failed to log wake event to database: {db_err}")

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

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine stats.
        """
        return {
            "initialized": self._initialized,
            "model_name": self._active_model_name,
            "confidence_threshold": self._confidence_threshold,
            "cooldown_seconds": self._cooldown_seconds,
            "detection_count": self._detection_count,
            "in_cooldown": self._in_cooldown(),
            "last_detection_time": (
                self._last_detection_time.isoformat()
                if self._last_detection_time
                else None
            ),
        }

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available pre-trained models.

        Returns:
            List of model names.
        """
        return AVAILABLE_MODELS.copy()

    def shutdown(self) -> None:
        """Shutdown the wake word engine."""
        logger.info(
            f"WakeWordEngine shutting down: model={self._active_model_name}, "
            f"detections={self._detection_count}"
        )
        self._initialized = False
        self._model = None
