"""Tests for the Wake Word Engine module.

Tests for wake word detection, confidence thresholds, and cooldown logic.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.headless.wake_word import (
    WakeWordEngine,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COOLDOWN_SECONDS,
    WAKE_WORD_SAMPLE_RATE,
)


@pytest.fixture
def mock_openwakeword_available():
    """Fixture to mock openwakeword as available."""
    with patch("reachy_mini_conversation_app.headless.wake_word.OPENWAKEWORD_AVAILABLE", True):
        yield


class TestWakeWordEngineInitialization:
    """Tests for wake word engine initialization."""

    def test_default_configuration(self, mock_openwakeword_available) -> None:
        """Test default configuration values."""
        engine = WakeWordEngine()

        # Check internal defaults (before initialize is called)
        assert engine._confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert engine._cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
        assert engine.is_initialized is False

    def test_custom_configuration(self, mock_openwakeword_available) -> None:
        """Test custom configuration values."""
        engine = WakeWordEngine(
            confidence_threshold=0.7,
            cooldown_seconds=5.0,
        )

        assert engine._confidence_threshold == 0.7
        assert engine._cooldown_seconds == 5.0

    def test_callback_registration(self, mock_openwakeword_available) -> None:
        """Test that callback is registered."""
        callback = Mock()
        engine = WakeWordEngine(on_wake_word=callback)

        assert engine._on_wake_word is callback

    def test_sample_rate_constant(self, mock_openwakeword_available) -> None:
        """Test that sample rate is 16kHz as expected by OpenWakeWord."""
        engine = WakeWordEngine()
        assert engine.sample_rate == 16000


class TestWakeWordDetection:
    """Tests for wake word detection logic."""

    def test_process_audio_requires_initialization(self, mock_openwakeword_available) -> None:
        """Test that process_audio returns None if not initialized."""
        engine = WakeWordEngine()

        audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence

        # Should return None when not initialized
        result = engine.process_audio(audio)
        assert result is None

    def test_wake_word_detected_high_confidence_triggers_callback(self) -> None:
        """Test wake word detection with high confidence triggers callback."""
        callback_called = False
        detected_confidence = None

        def on_wake(confidence: float) -> None:
            nonlocal callback_called, detected_confidence
            callback_called = True
            detected_confidence = confidence

        with patch("reachy_mini_conversation_app.headless.wake_word.OPENWAKEWORD_AVAILABLE", True):
            # Mock the model to return high confidence detection
            mock_model = MagicMock()
            mock_model.predict.return_value = {'hey_jarvis': 0.9}

            engine = WakeWordEngine(
                on_wake_word=on_wake,
                confidence_threshold=0.5,
            )
            # Manually initialize without calling initialize()
            engine._model = mock_model
            engine._initialized = True
            engine._active_model_name = 'hey_jarvis'

            audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
            result = engine.process_audio(audio)

            assert result == 0.9
            assert callback_called is True
            assert detected_confidence == 0.9
            assert engine.detection_count == 1

    def test_wake_word_low_confidence_does_not_trigger_callback(self) -> None:
        """Test wake word detection with low confidence does not trigger callback."""
        callback_called = False

        def on_wake(confidence: float) -> None:
            nonlocal callback_called
            callback_called = True

        with patch("reachy_mini_conversation_app.headless.wake_word.OPENWAKEWORD_AVAILABLE", True):
            # Mock the model to return low confidence detection
            mock_model = MagicMock()
            mock_model.predict.return_value = {'hey_jarvis': 0.3}  # Below threshold

            engine = WakeWordEngine(
                on_wake_word=on_wake,
                confidence_threshold=0.5,  # Threshold is 0.5
            )
            # Manually initialize
            engine._model = mock_model
            engine._initialized = True
            engine._active_model_name = 'hey_jarvis'

            audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
            result = engine.process_audio(audio)

            # Low confidence should return None and not trigger callback
            assert result is None
            assert callback_called is False
            assert engine.detection_count == 0

    def test_wake_word_detection_respects_cooldown(self) -> None:
        """Test that wake word detection respects cooldown period."""
        detections = []

        def on_wake(confidence: float) -> None:
            detections.append(confidence)

        with patch("reachy_mini_conversation_app.headless.wake_word.OPENWAKEWORD_AVAILABLE", True):
            mock_model = MagicMock()
            mock_model.predict.return_value = {'hey_jarvis': 0.9}

            engine = WakeWordEngine(
                on_wake_word=on_wake,
                confidence_threshold=0.5,
                cooldown_seconds=10.0,  # Long cooldown
            )
            engine._model = mock_model
            engine._initialized = True
            engine._active_model_name = 'hey_jarvis'

            audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)

            # First detection should trigger
            result1 = engine.process_audio(audio)
            assert result1 == 0.9
            assert len(detections) == 1

            # Second detection immediately after should be blocked by cooldown
            result2 = engine.process_audio(audio)
            assert result2 is None  # Blocked by cooldown
            assert len(detections) == 1  # Still only 1 detection


class TestWakeWordCooldown:
    """Tests for wake word cooldown logic."""

    def test_cooldown_initially_not_active(self, mock_openwakeword_available) -> None:
        """Test that cooldown is not active initially."""
        engine = WakeWordEngine()
        assert engine._in_cooldown() is False

    def test_cooldown_activates_after_detection(self, mock_openwakeword_available) -> None:
        """Test that cooldown activates after a detection."""
        engine = WakeWordEngine(cooldown_seconds=10.0)

        # Simulate a detection
        engine._last_detection_time = datetime.now()

        # Should be in cooldown
        assert engine._in_cooldown() is True

    def test_cooldown_expires(self, mock_openwakeword_available) -> None:
        """Test that cooldown expires after timeout."""
        import time

        engine = WakeWordEngine(cooldown_seconds=0.1)

        # Set a past detection time
        engine._last_detection_time = datetime.now()

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Cooldown should have expired
        assert engine._in_cooldown() is False

    def test_reset_cooldown(self, mock_openwakeword_available) -> None:
        """Test that reset_cooldown clears the cooldown."""
        engine = WakeWordEngine(cooldown_seconds=10.0)

        # Simulate a detection (sets cooldown)
        engine._last_detection_time = datetime.now()
        assert engine._in_cooldown() is True

        # Reset cooldown
        engine.reset_cooldown()
        assert engine._in_cooldown() is False


class TestWakeWordConfidence:
    """Tests for confidence threshold logic."""

    def test_set_confidence_threshold(self, mock_openwakeword_available) -> None:
        """Test setting confidence threshold."""
        engine = WakeWordEngine(confidence_threshold=0.5)

        # Update threshold
        engine.set_confidence_threshold(0.8)
        assert engine._confidence_threshold == 0.8

    def test_confidence_threshold_clamped_to_range(self, mock_openwakeword_available) -> None:
        """Test that confidence threshold is clamped to 0.0-1.0."""
        engine = WakeWordEngine()

        # Test clamping above 1.0
        engine.set_confidence_threshold(1.5)
        assert engine._confidence_threshold == 1.0

        # Test clamping below 0.0
        engine.set_confidence_threshold(-0.5)
        assert engine._confidence_threshold == 0.0


class TestWakeWordAsyncProcessing:
    """Tests for async audio processing."""

    @pytest.mark.asyncio
    async def test_async_process_audio_returns_none_when_not_initialized(
        self, mock_openwakeword_available
    ) -> None:
        """Test async audio processing returns None when not initialized."""
        engine = WakeWordEngine()

        audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence

        # Should return None when not initialized
        result = await engine.process_audio_async(audio)
        assert result is None


class TestWakeWordShutdown:
    """Tests for engine shutdown."""

    def test_shutdown_resets_state(self, mock_openwakeword_available) -> None:
        """Test that shutdown resets internal state."""
        engine = WakeWordEngine()

        # Manually set as initialized
        engine._initialized = True

        # Shutdown
        engine.shutdown()

        # Should not be initialized anymore
        assert engine.is_initialized is False
        assert engine._model is None


class TestWakeWordAudioFormats:
    """Tests for different audio format handling."""

    def test_int16_audio_accepted(self, mock_openwakeword_available) -> None:
        """Test processing int16 audio doesn't raise."""
        engine = WakeWordEngine()

        audio = np.random.randint(-32768, 32767, 1600, dtype=np.int16)

        # Should not raise (will return None since not initialized)
        result = engine.process_audio(audio)
        assert result is None

    def test_empty_audio_handled(self, mock_openwakeword_available) -> None:
        """Test that empty audio is handled gracefully."""
        engine = WakeWordEngine()

        audio = np.array([], dtype=np.int16)

        # Should not raise
        result = engine.process_audio(audio)
        assert result is None


class TestWakeWordStats:
    """Tests for engine statistics."""

    def test_get_stats_returns_dict(self, mock_openwakeword_available) -> None:
        """Test that get_stats returns a proper dict."""
        engine = WakeWordEngine(confidence_threshold=0.6, cooldown_seconds=8.0)

        stats = engine.get_stats()

        assert isinstance(stats, dict)
        assert "initialized" in stats
        assert "confidence_threshold" in stats
        assert "cooldown_seconds" in stats
        assert "detection_count" in stats
        assert "in_cooldown" in stats

        assert stats["initialized"] is False
        assert stats["confidence_threshold"] == 0.6
        assert stats["cooldown_seconds"] == 8.0
        assert stats["detection_count"] == 0

    def test_detection_count_initially_zero(self, mock_openwakeword_available) -> None:
        """Test that detection count starts at zero."""
        engine = WakeWordEngine()
        assert engine.detection_count == 0


class TestWakeWordAvailableModels:
    """Tests for model listing."""

    def test_get_available_models(self) -> None:
        """Test getting available models."""
        models = WakeWordEngine.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert DEFAULT_MODEL in models

    def test_available_models_is_copy(self) -> None:
        """Test that get_available_models returns a copy, not the original."""
        models1 = WakeWordEngine.get_available_models()
        models2 = WakeWordEngine.get_available_models()

        # Modify one, shouldn't affect the other
        models1.append("test_model")
        assert "test_model" not in models2


class TestWakeWordWhenUnavailable:
    """Tests for behavior when openwakeword is not available."""

    def test_engine_initializes_but_disabled(self) -> None:
        """Test that engine initializes but is disabled when openwakeword unavailable."""
        with patch("reachy_mini_conversation_app.headless.wake_word.OPENWAKEWORD_AVAILABLE", False):
            engine = WakeWordEngine()
            assert engine.is_initialized is False

    def test_sample_rate_still_available(self, mock_openwakeword_available) -> None:
        """Test that sample rate property works even without initialization."""
        engine = WakeWordEngine()
        assert engine.sample_rate == WAKE_WORD_SAMPLE_RATE
