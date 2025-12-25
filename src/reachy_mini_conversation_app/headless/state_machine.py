"""State machine for headless conversation mode.

Manages transitions between SLEEPING and AWAKE states based on
wake word detection and silence timeout.
"""

import asyncio
import logging
from enum import Enum
from typing import Callable, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation state for headless mode."""

    SLEEPING = "sleeping"  # Wake word listening only, no API connection
    AWAKE = "awake"  # Full conversation active with OpenAI
    TRANSITIONING = "transitioning"  # Startup/shutdown transition


class StateMachine:
    """State machine for managing SLEEPING/AWAKE transitions.

    Handles wake word triggers, silence timeout, and OpenAI connection lifecycle.
    """

    def __init__(
        self,
        silence_timeout_seconds: float = 120.0,
        max_awake_seconds: float = 1800.0,
        min_sleep_seconds: float = 5.0,
        on_state_change: Optional[Callable[[ConversationState, ConversationState], None]] = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            silence_timeout_seconds: Seconds of silence before returning to SLEEPING.
            max_awake_seconds: Maximum time in AWAKE state (force sleep).
            min_sleep_seconds: Minimum time in SLEEPING state (prevent thrashing).
            on_state_change: Callback when state changes (old_state, new_state).
        """
        self._state = ConversationState.SLEEPING
        self._silence_timeout = silence_timeout_seconds
        self._max_awake_time = max_awake_seconds
        self._min_sleep_time = min_sleep_seconds
        self._on_state_change = on_state_change

        self._last_speech_time: Optional[datetime] = None
        self._state_entry_time: datetime = datetime.now()
        self._silence_timer_task: Optional[asyncio.Task[None]] = None

        logger.info(
            f"StateMachine initialized: silence_timeout={silence_timeout_seconds}s, "
            f"max_awake={max_awake_seconds}s"
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

    def _set_state(self, new_state: ConversationState) -> None:
        """Set the state and trigger callback."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_entry_time = datetime.now()

        logger.info(f"State transition: {old_state.value} → {new_state.value}")

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

    async def shutdown(self) -> None:
        """Clean shutdown of state machine."""
        logger.info("StateMachine shutting down")

        if self._silence_timer_task:
            self._silence_timer_task.cancel()
            try:
                await self._silence_timer_task
            except asyncio.CancelledError:
                pass

        self._set_state(ConversationState.SLEEPING)
