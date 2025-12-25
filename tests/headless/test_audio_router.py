"""Unit tests for AudioRouter.

Tests fan-out audio distribution, resampling, queue management,
and performance monitoring.
"""

import pytest
import numpy as np
import asyncio

from reachy_mini_conversation_app.headless.audio_router import (
    AudioRouter,
    NATIVE_SAMPLE_RATE,
    WAKE_WORD_SAMPLE_RATE,
    OPENAI_SAMPLE_RATE,
)


class TestAudioRouterResampling:
    """Test sample rate conversion."""

    def test_resample_48k_to_16k(self) -> None:
        """Verify resampling from 48kHz to 16kHz produces correct length."""
        router = AudioRouter(input_sample_rate=48000)

        # Generate 1 second of 48kHz audio (sine wave)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio_48k = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

        # Resample to 16kHz
        resampled = router._resample(audio_48k, 48000, 16000)

        # Should be exactly 16000 samples (1 second at 16kHz)
        assert len(resampled) == 16000, f"Expected 16000 samples, got {len(resampled)}"
        assert resampled.dtype == np.int16, f"Expected int16, got {resampled.dtype}"

    def test_resample_48k_to_24k(self) -> None:
        """Verify resampling from 48kHz to 24kHz produces correct length."""
        router = AudioRouter(input_sample_rate=48000)

        # Generate 0.5 seconds of 48kHz audio
        audio_48k = np.zeros(24000, dtype=np.int16)

        # Resample to 24kHz
        resampled = router._resample(audio_48k, 48000, 24000)

        # Should be 12000 samples (0.5 seconds at 24kHz)
        assert len(resampled) == 12000, f"Expected 12000 samples, got {len(resampled)}"

    def test_resample_same_rate_no_change(self) -> None:
        """Verify no resampling when rates are equal."""
        router = AudioRouter(input_sample_rate=48000)

        audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        resampled = router._resample(audio, 48000, 48000)

        # Should return same array
        assert np.array_equal(audio, resampled)

    def test_resample_preserves_type(self) -> None:
        """Verify output is always int16."""
        router = AudioRouter(input_sample_rate=48000)

        audio = np.array([32767, -32768, 0, 16384, -16384], dtype=np.int16)
        resampled = router._resample(audio, 48000, 16000)

        assert resampled.dtype == np.int16


class TestAudioRouterRouting:
    """Test audio routing to consumers."""

    @pytest.mark.asyncio
    async def test_route_to_all_consumers(self) -> None:
        """Verify audio reaches all enabled consumers."""
        router = AudioRouter(input_sample_rate=48000, max_queue_size=10)
        router.enable_conversation()

        # Route a frame
        audio = np.ones(1024, dtype=np.int16) * 1000
        await router.route_audio(audio)

        # Check all queues have data
        wake_audio = await router.get_wake_word_audio(timeout=0.1)
        conv_audio = await router.get_conversation_audio(timeout=0.1)
        rec_audio = await router.get_recorder_audio(timeout=0.1)

        assert wake_audio is not None, "Wake word queue should have data"
        assert conv_audio is not None, "Conversation queue should have data"
        assert rec_audio is not None, "Recorder queue should have data"

    @pytest.mark.asyncio
    async def test_conversation_disabled_by_default(self) -> None:
        """Verify conversation consumer is disabled by default (SLEEPING state)."""
        router = AudioRouter(input_sample_rate=48000)

        # Route a frame
        audio = np.ones(1024, dtype=np.int16)
        await router.route_audio(audio)

        # Conversation queue should be empty
        conv_audio = await router.get_conversation_audio(timeout=0.1)
        assert conv_audio is None, "Conversation should be disabled by default"

        # Wake word should still work
        wake_audio = await router.get_wake_word_audio(timeout=0.1)
        assert wake_audio is not None, "Wake word should be enabled"

    @pytest.mark.asyncio
    async def test_enable_disable_conversation(self) -> None:
        """Verify conversation can be enabled/disabled."""
        router = AudioRouter(input_sample_rate=48000)

        # Enable and route
        router.enable_conversation()
        await router.route_audio(np.ones(1024, dtype=np.int16))

        conv_audio = await router.get_conversation_audio(timeout=0.1)
        assert conv_audio is not None, "Conversation should work when enabled"

        # Disable and verify queue cleared
        router.disable_conversation()
        await router.route_audio(np.ones(1024, dtype=np.int16) * 2)

        conv_audio = await router.get_conversation_audio(timeout=0.1)
        assert conv_audio is None, "Conversation should not receive when disabled"


class TestAudioRouterBackpressure:
    """Test queue overflow handling."""

    @pytest.mark.asyncio
    async def test_backpressure_drops_oldest(self) -> None:
        """Verify oldest frames are dropped when queue is full."""
        router = AudioRouter(input_sample_rate=48000, max_queue_size=2)

        # Fill queue beyond capacity
        await router.route_audio(np.ones(1024, dtype=np.int16) * 1)
        await router.route_audio(np.ones(1024, dtype=np.int16) * 2)
        await router.route_audio(np.ones(1024, dtype=np.int16) * 3)

        # Should have dropped frames
        assert router.frames_dropped > 0, "Frames should be dropped when queue full"

        # Stats should reflect drops
        stats = router.get_stats()
        assert stats["frames_dropped"] > 0

    @pytest.mark.asyncio
    async def test_frames_routed_counter(self) -> None:
        """Verify frame counter increments correctly."""
        router = AudioRouter(input_sample_rate=48000)

        assert router.frames_routed == 0

        await router.route_audio(np.ones(1024, dtype=np.int16))
        assert router.frames_routed == 1

        await router.route_audio(np.ones(1024, dtype=np.int16))
        assert router.frames_routed == 2


class TestAudioRouterPrerollBuffer:
    """Test pre-roll buffer for wake word transitions."""

    @pytest.mark.asyncio
    async def test_preroll_buffer_accumulates(self) -> None:
        """Verify pre-roll buffer stores audio."""
        router = AudioRouter(input_sample_rate=48000, preroll_seconds=2.0)

        # Route several frames
        for _ in range(5):
            await router.route_audio(np.ones(1024, dtype=np.int16))

        # Pre-roll buffer should have content
        preroll = router.get_preroll_buffer()
        assert len(preroll) > 0, "Pre-roll buffer should have content"

    @pytest.mark.asyncio
    async def test_preroll_buffer_clears(self) -> None:
        """Verify pre-roll buffer can be cleared."""
        router = AudioRouter(input_sample_rate=48000)

        await router.route_audio(np.ones(1024, dtype=np.int16))
        assert len(router.get_preroll_buffer()) > 0

        router.clear_preroll_buffer()
        assert len(router.get_preroll_buffer()) == 0

    @pytest.mark.asyncio
    async def test_preroll_buffer_at_24khz(self) -> None:
        """Verify pre-roll buffer is stored at OpenAI sample rate (24kHz)."""
        router = AudioRouter(input_sample_rate=48000, preroll_seconds=1.0)

        # Route 1 second of 48kHz audio (48000 samples)
        await router.route_audio(np.ones(48000, dtype=np.int16))

        # Pre-roll should be at 24kHz (24000 samples)
        preroll = router.get_preroll_buffer()
        assert len(preroll) == 24000, f"Expected 24000 samples at 24kHz, got {len(preroll)}"

    def test_preroll_configuration(self) -> None:
        """Verify preroll_seconds parameter is respected."""
        router1 = AudioRouter(preroll_seconds=1.0)
        router2 = AudioRouter(preroll_seconds=5.0)

        assert router1._preroll_seconds == 1.0
        assert router2._preroll_seconds == 5.0


class TestAudioRouterStats:
    """Test statistics and performance monitoring."""

    @pytest.mark.asyncio
    async def test_stats_include_timing(self) -> None:
        """Verify stats include performance metrics."""
        router = AudioRouter(input_sample_rate=48000)

        await router.route_audio(np.ones(1024, dtype=np.int16))

        stats = router.get_stats()

        assert "frames_routed" in stats
        assert "frames_dropped" in stats
        assert "slow_routes_count" in stats
        assert "avg_route_time_ms" in stats
        assert "total_route_time_ms" in stats

        # Timing should be non-negative
        assert stats["avg_route_time_ms"] >= 0
        assert stats["total_route_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_stats_queue_sizes(self) -> None:
        """Verify stats include queue sizes."""
        router = AudioRouter(input_sample_rate=48000)

        stats = router.get_stats()

        assert "wake_word_queue_size" in stats
        assert "conversation_queue_size" in stats
        assert "recorder_queue_size" in stats
        assert "preroll_frames" in stats


class TestAudioRouterShutdown:
    """Test shutdown behavior."""

    def test_shutdown_disables_consumers(self) -> None:
        """Verify shutdown disables all consumers."""
        router = AudioRouter()
        router.enable_conversation()

        assert router._wake_word_enabled is True
        assert router._conversation_enabled is True
        assert router._recorder_enabled is True

        router.shutdown()

        assert router._wake_word_enabled is False
        assert router._conversation_enabled is False
        assert router._recorder_enabled is False


class TestAudioRouterSampleRateConstants:
    """Test sample rate constant values."""

    def test_sample_rate_values(self) -> None:
        """Verify sample rate constants have correct values."""
        assert NATIVE_SAMPLE_RATE == 48000
        assert WAKE_WORD_SAMPLE_RATE == 16000
        assert OPENAI_SAMPLE_RATE == 24000
