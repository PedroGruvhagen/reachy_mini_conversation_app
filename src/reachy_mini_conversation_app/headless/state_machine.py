"""State machine for headless conversation mode.

Manages transitions between SLEEPING and AWAKE states based on
wake word detection and silence timeout.
"""

import os
import json
import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from datetime import datetime


logger = logging.getLogger(__name__)

# State persistence configuration
DEFAULT_STATE_FILE = "~/.reachy_mini/state.json"
MAX_STATE_AGE_SECONDS = 3600  # 1 hour - stale states older than this are ignored


class ConversationState(Enum):
    """Conversation state for headless mode."""

    SLEEPING = "sleeping"  # Wake word listening only, no API connection
    AWAKE = "awake"  # Full conversation active with OpenAI
    TRANSITIONING = "transitioning"  # Startup/shutdown transition


class StateMachine:
    """State machine for managing SLEEPING/AWAKE transitions.

    Handles wake word triggers, silence timeout, and OpenAI connection lifecycle.
    Supports state persistence to survive restarts.
    """

    def __init__(
        self,
        silence_timeout_seconds: float = 120.0,
        max_awake_seconds: float = 1800.0,
        min_sleep_seconds: float = 5.0,
        on_state_change: Optional[Callable[[ConversationState, ConversationState], None]] = None,
        state_file: Optional[str] = None,
        restore_state: bool = True,
    ) -> None:
        """Initialize the state machine.

        Args:
            silence_timeout_seconds: Seconds of silence before returning to SLEEPING.
            max_awake_seconds: Maximum time in AWAKE state (force sleep).
            min_sleep_seconds: Minimum time in SLEEPING state (prevent thrashing).
            on_state_change: Callback when state changes (old_state, new_state).
            state_file: Path to state persistence file. Defaults to ~/.reachy_mini/state.json.
            restore_state: If True, attempt to restore state from file on init.
        """
        self._state = ConversationState.SLEEPING
        self._silence_timeout = silence_timeout_seconds
        self._max_awake_time = max_awake_seconds
        self._min_sleep_time = min_sleep_seconds
        self._on_state_change = on_state_change

        # State persistence
        self._state_file = Path(
            os.getenv("STATE_FILE_PATH", state_file or DEFAULT_STATE_FILE)
        ).expanduser()
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        self._last_speech_time: Optional[datetime] = None
        self._state_entry_time: datetime = datetime.now()
        self._silence_timer_task: Optional[asyncio.Task[None]] = None

        # State transition metrics
        self._transition_count = 0
        self._total_awake_time = 0.0
        self._last_awake_duration: Optional[float] = None

        # Attempt to restore state from file
        if restore_state:
            self._try_restore_state()

        logger.info(
            f"StateMachine initialized: silence_timeout={silence_timeout_seconds}s, "
            f"max_awake={max_awake_seconds}s, state_file={self._state_file}"
        )

    @property
    def state(self) -> ConversationState:
        """Get the current state."""
        return self._state

    @property
    def is_sleeping(self) -> bool:
        """Check if currently in SLEEPING state."""
        return self._state == ConversationState.SLEEPING

    @property
    def is_awake(self) -> bool:
        """Check if currently in AWAKE state."""
        return self._state == ConversationState.AWAKE

    @property
    def time_in_current_state(self) -> float:
        """Get seconds spent in current state."""
        return (datetime.now() - self._state_entry_time).total_seconds()

    @property
    def seconds_since_last_speech(self) -> Optional[float]:
        """Get seconds since last speech activity, or None if never."""
        if self._last_speech_time is None:
            return None
        return (datetime.now() - self._last_speech_time).total_seconds()

    @property
    def transition_count(self) -> int:
        """Get total number of state transitions."""
        return self._transition_count

    def _try_restore_state(self) -> bool:
        """Try to restore state from persistence file.

        Returns:
            True if state was restored, False otherwise.
        """
        try:
            if not self._state_file.exists():
                logger.debug(f"No state file found at {self._state_file}")
                return False

            with open(self._state_file, "r") as f:
                data = json.load(f)

            # Validate required fields
            if "state" not in data or "timestamp" not in data:
                logger.warning("Invalid state file format, ignoring")
                return False

            # Check if state is stale
            saved_time = datetime.fromisoformat(data["timestamp"])
            age_seconds = (datetime.now() - saved_time).total_seconds()

            if age_seconds > MAX_STATE_AGE_SECONDS:
                logger.info(
                    f"State file is stale ({age_seconds:.0f}s old > {MAX_STATE_AGE_SECONDS}s), "
                    "starting in SLEEPING state"
                )
                return False

            # Restore state (only SLEEPING/AWAKE, not TRANSITIONING)
            saved_state = data["state"]
            if saved_state == ConversationState.AWAKE.value:
                # Restore to AWAKE only if recently active
                self._state = ConversationState.AWAKE
                self._state_entry_time = saved_time
                logger.info(
                    f"Restored state: AWAKE (saved {age_seconds:.0f}s ago)"
                )
                return True
            else:
                # Default to SLEEPING for any other state
                self._state = ConversationState.SLEEPING
                logger.debug("State file indicated SLEEPING, using default")
                return True

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse state file: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to restore state: {e}")
            return False

    def _save_state(self) -> None:
        """Save current state to persistence file."""
        try:
            data = {
                "state": self._state.value,
                "timestamp": datetime.now().isoformat(),
                "state_entry_time": self._state_entry_time.isoformat(),
                "transition_count": self._transition_count,
                "total_awake_time": self._total_awake_time,
            }

            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"State saved: {self._state.value}")

        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _set_state(self, new_state: ConversationState) -> None:
        """Set the state, trigger callback, and persist."""
        old_state = self._state
        if old_state == new_state:
            return

        # Track metrics for AWAKE state duration
        if old_state == ConversationState.AWAKE:
            awake_duration = self.time_in_current_state
            self._total_awake_time += awake_duration
            self._last_awake_duration = awake_duration
            logger.debug(f"AWAKE session lasted {awake_duration:.1f}s")

        self._state = new_state
        self._state_entry_time = datetime.now()
        self._transition_count += 1

        logger.info(
            f"State transition #{self._transition_count}: "
            f"{old_state.value} → {new_state.value}"
        )

        # Persist state to file
        self._save_state()

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    async def on_wake_word_detected(self, confidence: float = 1.0) -> bool:
        """Handle wake word detection.

        Args:
            confidence: Detection confidence score.

        Returns:
            True if transitioned to AWAKE, False if ignored.
        """
        if self._state == ConversationState.AWAKE:
            logger.debug("Wake word detected but already AWAKE, ignoring")
            return False

        if self._state == ConversationState.TRANSITIONING:
            logger.debug("Wake word detected during transition, ignoring")
            return False

        # Check minimum sleep time
        if self.time_in_current_state < self._min_sleep_time:
            logger.debug(
                f"Wake word detected but min sleep time not met "
                f"({self.time_in_current_state:.1f}s < {self._min_sleep_time}s)"
            )
            return False

        logger.info(f"Wake word detected with confidence {confidence:.2f}, transitioning to AWAKE")

        self._set_state(ConversationState.TRANSITIONING)
        self._last_speech_time = datetime.now()

        # Start silence timer
        await self._start_silence_timer()

        self._set_state(ConversationState.AWAKE)
        return True

    async def on_user_speech(self) -> None:
        """Handle user speech activity - resets silence timer."""
        self._last_speech_time = datetime.now()

        if self._state == ConversationState.AWAKE:
            # Restart the silence timer
            await self._start_silence_timer()

    async def on_silence_timeout(self) -> bool:
        """Handle silence timeout - transition to SLEEPING.

        Returns:
            True if transitioned to SLEEPING.
        """
        if self._state != ConversationState.AWAKE:
            return False

        logger.info(f"Silence timeout after {self._silence_timeout}s, transitioning to SLEEPING")

        self._set_state(ConversationState.TRANSITIONING)

        # Cancel any existing timer
        if self._silence_timer_task:
            self._silence_timer_task.cancel()
            self._silence_timer_task = None

        self._set_state(ConversationState.SLEEPING)
        return True

    async def check_max_awake_timeout(self) -> bool:
        """Check if max AWAKE time exceeded.

        Returns:
            True if forced to SLEEPING due to max time.
        """
        if self._state != ConversationState.AWAKE:
            return False

        if self.time_in_current_state >= self._max_awake_time:
            logger.warning(
                f"Max AWAKE time exceeded ({self._max_awake_time}s), forcing SLEEPING"
            )
            await self.on_silence_timeout()
            return True

        return False

    async def _start_silence_timer(self) -> None:
        """Start or restart the silence timer."""
        # Cancel existing timer
        if self._silence_timer_task:
            self._silence_timer_task.cancel()
            try:
                await self._silence_timer_task
            except asyncio.CancelledError:
                pass

        # Start new timer
        self._silence_timer_task = asyncio.create_task(self._silence_timer_loop())

    async def _silence_timer_loop(self) -> None:
        """Background task that triggers timeout after silence period."""
        try:
            await asyncio.sleep(self._silence_timeout)
            await self.on_silence_timeout()
        except asyncio.CancelledError:
            pass

    async def force_sleep(self) -> None:
        """Force transition to SLEEPING state."""
        if self._silence_timer_task:
            self._silence_timer_task.cancel()
            self._silence_timer_task = None

        self._set_state(ConversationState.SLEEPING)

    def get_metrics(self) -> Dict[str, Any]:
        """Get state machine metrics.

        Returns:
            Dictionary with metrics for monitoring/debugging.
        """
        return {
            "current_state": self._state.value,
            "transition_count": self._transition_count,
            "time_in_current_state": self.time_in_current_state,
            "seconds_since_last_speech": self.seconds_since_last_speech,
            "total_awake_time": self._total_awake_time,
            "last_awake_duration": self._last_awake_duration,
            "silence_timeout": self._silence_timeout,
            "max_awake_time": self._max_awake_time,
            "state_file": str(self._state_file),
        }

    def clear_state_file(self) -> None:
        """Clear the state persistence file."""
        try:
            if self._state_file.exists():
                self._state_file.unlink()
                logger.debug(f"State file cleared: {self._state_file}")
        except Exception as e:
            logger.warning(f"Failed to clear state file: {e}")

    async def shutdown(self) -> None:
        """Clean shutdown of state machine."""
        logger.info("StateMachine shutting down")

        if self._silence_timer_task:
            self._silence_timer_task.cancel()
            try:
                await self._silence_timer_task
            except asyncio.CancelledError:
                pass

        # Set to SLEEPING and save final state
        self._set_state(ConversationState.SLEEPING)

        logger.info(
            f"StateMachine shutdown complete: {self._transition_count} transitions, "
            f"{self._total_awake_time:.1f}s total awake time"
        )
