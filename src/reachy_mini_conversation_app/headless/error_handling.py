"""Error handling utilities for headless operation.

Provides custom exceptions, retry logic, circuit breakers, and recovery
mechanisms for robust operation without user intervention.
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


# Type variables for generic decorators
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Custom Exception Types
# =============================================================================


class HeadlessError(Exception):
    """Base exception for headless system errors."""

    def __init__(self, message: str, recoverable: bool = True) -> None:
        """Initialize error.

        Args:
            message: Error description.
            recoverable: Whether the error can be recovered from automatically.
        """
        super().__init__(message)
        self.recoverable = recoverable
        self.timestamp = datetime.now()


class AudioDeviceError(HeadlessError):
    """Error related to audio device operations."""

    def __init__(self, message: str, device_name: Optional[str] = None) -> None:
        """Initialize audio device error.

        Args:
            message: Error description.
            device_name: Name of the affected audio device.
        """
        super().__init__(message, recoverable=True)
        self.device_name = device_name


class WakeWordError(HeadlessError):
    """Error in wake word detection system."""

    pass


class StateTransitionError(HeadlessError):
    """Error during state machine transition."""

    pass


class TranscriptionError(HeadlessError):
    """Error in transcription service."""

    def __init__(self, message: str, is_rate_limit: bool = False) -> None:
        """Initialize transcription error.

        Args:
            message: Error description.
            is_rate_limit: Whether this is a rate limit error.
        """
        super().__init__(message, recoverable=True)
        self.is_rate_limit = is_rate_limit


class APIError(HeadlessError):
    """Error from external API calls."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        is_transient: bool = True,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error description.
            status_code: HTTP status code if applicable.
            is_transient: Whether the error might succeed on retry.
        """
        super().__init__(message, recoverable=is_transient)
        self.status_code = status_code
        self.is_transient = is_transient


class ConfigurationError(HeadlessError):
    """Error in configuration."""

    def __init__(self, message: str) -> None:
        """Initialize configuration error.

        Args:
            message: Error description.
        """
        super().__init__(message, recoverable=False)


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================


class CircuitState(Enum):
    """States for the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls.

    Prevents cascading failures by failing fast when a service is unhealthy.

    Example:
        >>> breaker = CircuitBreaker(name="openai", failure_threshold=5)
        >>> async with breaker:
        ...     await call_openai_api()
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_requests: int = 1  # requests to allow in half-open state

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _half_open_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _total_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN (recovery timeout)")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_count = 0
        return self._state

    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_count < self.half_open_requests
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        self._total_calls += 1
        self._success_count += 1

        # Reset failure count on success (prevents accumulated failures over time)
        self._failure_count = 0

        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (success)")
            self._state = CircuitState.CLOSED

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        self._total_calls += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure in test)")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit {self.name}: CLOSED -> OPEN "
                f"(failures: {self._failure_count}/{self.failure_threshold})"
            )
            self._state = CircuitState.OPEN

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        if not self.is_available:
            raise APIError(
                f"Circuit {self.name} is OPEN - service unavailable",
                is_transient=True,
            )
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Async context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val if isinstance(exc_val, Exception) else None)
        return False  # Don't suppress exceptions

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_count = 0
        logger.info(f"Circuit {self.name}: reset to CLOSED")


# =============================================================================
# Retry Decorator
# =============================================================================


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """Decorator for async function retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try).
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback(attempt, exception, delay) on each retry.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> @retry_async(max_attempts=3, base_delay=1.0)
        ... async def call_api():
        ...     return await some_api_call()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper  # type: ignore

    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for sync function retry with exponential backoff.

    Same as retry_async but for synchronous functions.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Error Aggregator for Monitoring
# =============================================================================


@dataclass
class ErrorStats:
    """Statistics for a specific error type."""

    error_type: str
    count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    sample_messages: list[str] = field(default_factory=list)
    max_samples: int = field(default=5, repr=False)

    def record(self, message: str) -> None:
        """Record an error occurrence."""
        now = datetime.now()
        self.count += 1
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now
        if len(self.sample_messages) < self.max_samples:
            self.sample_messages.append(message)


class ErrorAggregator:
    """Aggregates errors for monitoring and reporting.

    Tracks error rates, patterns, and provides summaries for monitoring.
    """

    def __init__(
        self,
        window_size: timedelta = timedelta(minutes=5),
        alert_threshold: int = 10,
    ) -> None:
        """Initialize error aggregator.

        Args:
            window_size: Time window for rate calculations.
            alert_threshold: Number of errors in window to trigger alert.
        """
        self._window_size = window_size
        self._alert_threshold = alert_threshold
        self._errors: dict[str, ErrorStats] = {}
        self._recent_errors: list[tuple[datetime, str, str]] = []
        self._alert_callback: Optional[Callable[[str, int], None]] = None

    def set_alert_callback(
        self, callback: Callable[[str, int], None]
    ) -> None:
        """Set callback for threshold alerts.

        Args:
            callback: Function(error_type, count) called when threshold exceeded.
        """
        self._alert_callback = callback

    def record_error(
        self,
        error: Exception,
        context: Optional[str] = None,
    ) -> None:
        """Record an error occurrence.

        Args:
            error: The exception that occurred.
            context: Optional context string.
        """
        error_type = type(error).__name__
        message = str(error)
        if context:
            message = f"[{context}] {message}"

        # Update stats
        if error_type not in self._errors:
            self._errors[error_type] = ErrorStats(error_type=error_type)
        self._errors[error_type].record(message)

        # Track recent errors
        now = datetime.now()
        self._recent_errors.append((now, error_type, message))

        # Prune old errors
        cutoff = now - self._window_size
        self._recent_errors = [
            (t, et, m) for t, et, m in self._recent_errors if t > cutoff
        ]

        # Check for alert threshold
        window_count = sum(
            1 for t, et, _ in self._recent_errors if et == error_type
        )
        if window_count >= self._alert_threshold:
            logger.error(
                f"Error threshold exceeded: {error_type} occurred "
                f"{window_count} times in {self._window_size}"
            )
            if self._alert_callback:
                self._alert_callback(error_type, window_count)

    def get_error_rate(self, error_type: Optional[str] = None) -> float:
        """Get error rate per minute.

        Args:
            error_type: Specific error type, or None for all errors.

        Returns:
            Errors per minute in the current window.
        """
        now = datetime.now()
        cutoff = now - self._window_size
        window_minutes = self._window_size.total_seconds() / 60

        if error_type:
            count = sum(
                1 for t, et, _ in self._recent_errors
                if t > cutoff and et == error_type
            )
        else:
            count = sum(1 for t, _, _ in self._recent_errors if t > cutoff)

        return count / window_minutes if window_minutes > 0 else 0

    def get_summary(self) -> dict[str, Any]:
        """Get error summary for monitoring."""
        return {
            "total_error_types": len(self._errors),
            "errors_in_window": len(self._recent_errors),
            "error_rate_per_minute": round(self.get_error_rate(), 2),
            "window_size_minutes": self._window_size.total_seconds() / 60,
            "by_type": {
                et: {
                    "count": stats.count,
                    "first_seen": stats.first_seen.isoformat() if stats.first_seen else None,
                    "last_seen": stats.last_seen.isoformat() if stats.last_seen else None,
                    "rate_per_minute": round(self.get_error_rate(et), 2),
                }
                for et, stats in self._errors.items()
            },
        }

    def reset(self) -> None:
        """Reset all error statistics."""
        self._errors.clear()
        self._recent_errors.clear()


# =============================================================================
# Recovery Utilities
# =============================================================================


class ComponentHealth(Enum):
    """Health status for a component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result for a component."""

    component: str
    status: ComponentHealth
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == ComponentHealth.HEALTHY


class RecoveryManager:
    """Manages component health checks and recovery actions.

    Monitors component health and triggers recovery actions when needed.
    """

    def __init__(self) -> None:
        """Initialize recovery manager."""
        self._health_checks: dict[str, HealthCheck] = {}
        self._recovery_actions: dict[str, Callable[[], bool]] = {}
        self._max_recovery_attempts: int = 3

    def register_component(
        self,
        name: str,
        recovery_action: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Register a component for health monitoring.

        Args:
            name: Component name.
            recovery_action: Optional function to call for recovery.
                Should return True if recovery succeeded.
        """
        self._health_checks[name] = HealthCheck(
            component=name,
            status=ComponentHealth.UNKNOWN,
        )
        if recovery_action:
            self._recovery_actions[name] = recovery_action

    def update_health(
        self,
        name: str,
        status: ComponentHealth,
        message: str = "",
    ) -> None:
        """Update component health status.

        Args:
            name: Component name.
            status: New health status.
            message: Optional status message.
        """
        if name not in self._health_checks:
            self.register_component(name)

        check = self._health_checks[name]
        old_status = check.status

        check.status = status
        check.message = message
        check.last_check = datetime.now()

        if status != ComponentHealth.HEALTHY:
            check.consecutive_failures += 1
        else:
            check.consecutive_failures = 0

        # Log status changes
        if old_status != status:
            if status == ComponentHealth.HEALTHY:
                logger.info(f"Component {name}: {old_status.value} -> {status.value}")
            else:
                logger.warning(
                    f"Component {name}: {old_status.value} -> {status.value}: {message}"
                )

    def attempt_recovery(self, name: str) -> bool:
        """Attempt to recover a component.

        Args:
            name: Component name.

        Returns:
            True if recovery succeeded.
        """
        if name not in self._recovery_actions:
            logger.warning(f"No recovery action registered for {name}")
            return False

        check = self._health_checks.get(name)
        if check and check.consecutive_failures > self._max_recovery_attempts:
            logger.error(
                f"Component {name}: max recovery attempts "
                f"({self._max_recovery_attempts}) exceeded"
            )
            return False

        logger.info(f"Attempting recovery for {name}...")
        try:
            if self._recovery_actions[name]():
                self.update_health(name, ComponentHealth.HEALTHY, "Recovered")
                return True
            else:
                self.update_health(
                    name, ComponentHealth.UNHEALTHY, "Recovery failed"
                )
                return False
        except Exception as e:
            logger.error(f"Recovery action for {name} raised exception: {e}")
            self.update_health(name, ComponentHealth.UNHEALTHY, str(e))
            return False

    def get_overall_health(self) -> ComponentHealth:
        """Get overall system health.

        Returns:
            HEALTHY if all components healthy,
            DEGRADED if any degraded,
            UNHEALTHY if any unhealthy.
        """
        if not self._health_checks:
            return ComponentHealth.UNKNOWN

        statuses = [check.status for check in self._health_checks.values()]

        if all(s == ComponentHealth.HEALTHY for s in statuses):
            return ComponentHealth.HEALTHY
        if any(s == ComponentHealth.UNHEALTHY for s in statuses):
            return ComponentHealth.UNHEALTHY
        if any(s == ComponentHealth.DEGRADED for s in statuses):
            return ComponentHealth.DEGRADED
        return ComponentHealth.UNKNOWN

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary for all components."""
        return {
            "overall": self.get_overall_health().value,
            "components": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "last_check": check.last_check.isoformat(),
                    "consecutive_failures": check.consecutive_failures,
                }
                for name, check in self._health_checks.items()
            },
        }


# =============================================================================
# Global Instances
# =============================================================================

# Global error aggregator for the headless system
error_aggregator = ErrorAggregator()

# Global recovery manager
recovery_manager = RecoveryManager()
