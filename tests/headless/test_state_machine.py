"""Tests for the State Machine module.

Tests for SLEEPING/AWAKE transitions, timeouts, and state persistence.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from reachy_mini_conversation_app.headless.state_machine import StateMachine, ConversationState


class TestStateMachineStates:
    """Tests for state machine state management."""

    def test_initial_state_is_sleeping(self) -> None:
        """Test that state machine starts in SLEEPING state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(state_file=str(state_file), restore_state=False)
            assert sm.state == ConversationState.SLEEPING
            assert sm.is_sleeping is True
            assert sm.is_awake is False

    def test_state_properties(self) -> None:
        """Test state boolean properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(state_file=str(state_file), restore_state=False)

            # Initially sleeping
            assert sm.is_sleeping is True
            assert sm.is_awake is False

            # Manually set state to AWAKE
            sm._state = ConversationState.AWAKE
            assert sm.is_sleeping is False
            assert sm.is_awake is True


class TestStateMachineTransitions:
    """Tests for state transitions."""

    @pytest.mark.asyncio
    async def test_wake_word_triggers_awake(self) -> None:
        """Test that wake word detection triggers AWAKE state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            callback_called = False
            old_state_captured = None
            new_state_captured = None

            def on_change(old: ConversationState, new: ConversationState) -> None:
                nonlocal callback_called, old_state_captured, new_state_captured
                callback_called = True
                old_state_captured = old
                new_state_captured = new

            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                on_state_change=on_change,
                min_sleep_seconds=0  # Allow immediate wake
            )
            result = await sm.on_wake_word_detected(confidence=0.8)

            assert result is True
            assert sm.state == ConversationState.AWAKE
            assert callback_called is True
            # Callback fires for TRANSITIONING and AWAKE states
            assert new_state_captured == ConversationState.AWAKE

    @pytest.mark.asyncio
    async def test_force_sleep_transitions_to_sleeping(self) -> None:
        """Test force_sleep transitions from AWAKE to SLEEPING."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # First wake up
            await sm.on_wake_word_detected(confidence=0.8)
            assert sm.is_awake is True

            # Then go to sleep
            await sm.force_sleep()
            assert sm.is_sleeping is True

    @pytest.mark.asyncio
    async def test_user_speech_updates_last_speech_time(self) -> None:
        """Test that user speech updates the last speech time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                silence_timeout_seconds=2,
                min_sleep_seconds=0
            )

            # Wake up
            await sm.on_wake_word_detected(confidence=0.8)

            # Simulate user speech
            initial_time = sm._last_speech_time
            await asyncio.sleep(0.05)
            await sm.on_user_speech()

            # Last speech time should be updated
            assert sm._last_speech_time > initial_time


class TestStateMachineTimeouts:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_silence_timeout_triggers_sleep(self) -> None:
        """Test that silence timeout triggers transition to SLEEPING."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                silence_timeout_seconds=0.1,
                min_sleep_seconds=0
            )

            # Wake up
            await sm.on_wake_word_detected(confidence=0.8)
            assert sm.is_awake is True

            # Wait for the background silence timer to trigger automatically
            await asyncio.sleep(0.2)

            # The background timer should have triggered sleep automatically
            assert sm.is_sleeping is True

    @pytest.mark.asyncio
    async def test_max_awake_timeout_triggers_sleep(self) -> None:
        """Test that max awake timeout forces transition to SLEEPING."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                max_awake_seconds=0.1,
                min_sleep_seconds=0,
                silence_timeout_seconds=60  # Long timeout to not interfere
            )

            # Wake up
            await sm.on_wake_word_detected(confidence=0.8)
            assert sm.is_awake is True

            # Wait for max awake timeout
            await asyncio.sleep(0.15)
            result = await sm.check_max_awake_timeout()

            assert result is True
            assert sm.is_sleeping is True

    @pytest.mark.asyncio
    async def test_min_sleep_time_prevents_immediate_wake(self) -> None:
        """Test that min sleep time prevents immediate re-wake."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0.1,  # Short min sleep for testing
                silence_timeout_seconds=60  # Long timeout to not interfere
            )

            # Wait for min_sleep_time to pass before first wake
            await asyncio.sleep(0.15)

            # Wake up
            result1 = await sm.on_wake_word_detected(confidence=0.8)
            assert result1 is True  # Should wake successfully

            # Go back to sleep
            await sm.force_sleep()
            assert sm.is_sleeping is True

            # Try to wake up immediately (should be blocked by min_sleep_time)
            result2 = await sm.on_wake_word_detected(confidence=0.8)
            assert result2 is False  # Should be blocked
            assert sm.is_sleeping is True  # Still sleeping

            # Wait for min_sleep_time to pass
            await asyncio.sleep(0.15)

            # Now wake should succeed
            result3 = await sm.on_wake_word_detected(confidence=0.8)
            assert result3 is True
            assert sm.is_awake is True


class TestStateMachinePersistence:
    """Tests for state persistence."""

    @pytest.mark.asyncio
    async def test_state_persisted_to_file(self) -> None:
        """Test that state is persisted to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # Wake up (should persist)
            await sm.on_wake_word_detected(confidence=0.8)

            # Check file exists
            assert state_file.exists()

            # Load and verify
            data = json.loads(state_file.read_text())
            assert data["state"] == "awake"

    @pytest.mark.asyncio
    async def test_state_file_cleared(self) -> None:
        """Test that state file can be cleared."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # Wake up to create state file
            await sm.on_wake_word_detected(confidence=0.8)
            assert state_file.exists()

            # Clear state file
            sm.clear_state_file()
            assert not state_file.exists()


class TestStateMachineMetrics:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_transition_count_tracked(self) -> None:
        """Test that state transitions are counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            initial_count = sm.transition_count

            # Wake up (SLEEPING -> TRANSITIONING -> AWAKE = 2 transitions)
            await sm.on_wake_word_detected(confidence=0.8)

            # Force sleep (AWAKE -> SLEEPING = 1 transition)
            await sm.force_sleep()

            assert sm.transition_count >= initial_count + 3

    @pytest.mark.asyncio
    async def test_get_metrics_returns_dict(self) -> None:
        """Test that get_metrics returns a proper dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # Wake up
            await sm.on_wake_word_detected(confidence=0.8)

            metrics = sm.get_metrics()

            assert isinstance(metrics, dict)
            assert "current_state" in metrics
            assert "transition_count" in metrics
            assert "total_awake_time" in metrics
            assert metrics["current_state"] == "awake"


class TestStateMachineShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self) -> None:
        """Test that shutdown cancels background tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # Start some background activity
            await sm.on_wake_word_detected(confidence=0.8)
            assert sm.is_awake is True

            # Shutdown
            await sm.shutdown()

            # State should be SLEEPING after shutdown
            assert sm.is_sleeping is True

    @pytest.mark.asyncio
    async def test_shutdown_saves_state(self) -> None:
        """Test that shutdown saves final state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False,
                min_sleep_seconds=0
            )

            # Wake up
            await sm.on_wake_word_detected(confidence=0.8)

            # Shutdown
            await sm.shutdown()

            # State should be saved
            assert state_file.exists()
            data = json.loads(state_file.read_text())
            assert data["state"] == "sleeping"


class TestStateMachineTimeTracking:
    """Tests for time tracking."""

    def test_time_in_current_state(self) -> None:
        """Test time_in_current_state property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False
            )

            # Time should be positive
            assert sm.time_in_current_state >= 0

    def test_seconds_since_last_speech_initially_none(self) -> None:
        """Test seconds_since_last_speech is None initially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateMachine(
                state_file=str(state_file),
                restore_state=False
            )

            assert sm.seconds_since_last_speech is None
