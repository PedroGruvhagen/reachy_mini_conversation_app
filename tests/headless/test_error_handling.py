"""Tests for the Error Handling module.

Tests for custom exceptions, retry logic, circuit breakers, and recovery management.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import pytest

from reachy_mini_conversation_app.headless.error_handling import (
    # Exceptions
    HeadlessError,
    AudioDeviceError,
    WakeWordError,
    StateTransitionError,
    TranscriptionError,
    APIError,
    ConfigurationError,
    # Circuit Breaker
    CircuitBreaker,
    CircuitState,
    # Retry Decorators
    retry_async,
    retry_sync,
    # Error Aggregator
    ErrorAggregator,
    ErrorStats,
    # Recovery Manager
    ComponentHealth,
    HealthCheck,
    RecoveryManager,
)


class TestCustomExceptions:
    """Tests for custom exception types."""

    def test_headless_error_defaults_to_recoverable(self) -> None:
        """Test HeadlessError default recoverable state."""
        error = HeadlessError("test error")
        assert error.recoverable is True
        assert error.timestamp is not None

    def test_headless_error_non_recoverable(self) -> None:
        """Test HeadlessError non-recoverable state."""
        error = HeadlessError("critical error", recoverable=False)
        assert error.recoverable is False

    def test_audio_device_error_has_device_name(self) -> None:
        """Test AudioDeviceError stores device name."""
        error = AudioDeviceError("device disconnected", device_name="hw:1,0")
        assert error.device_name == "hw:1,0"
        assert error.recoverable is True

    def test_transcription_error_rate_limit_flag(self) -> None:
        """Test TranscriptionError rate limit detection."""
        error = TranscriptionError("rate limited", is_rate_limit=True)
        assert error.is_rate_limit is True
        assert error.recoverable is True

    def test_api_error_with_status_code(self) -> None:
        """Test APIError stores HTTP status code."""
        error = APIError("server error", status_code=500, is_transient=True)
        assert error.status_code == 500
        assert error.is_transient is True
        assert error.recoverable is True

    def test_configuration_error_not_recoverable(self) -> None:
        """Test ConfigurationError is never recoverable."""
        error = ConfigurationError("missing API key")
        assert error.recoverable is False


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state_is_closed(self) -> None:
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available is True

    def test_opens_after_failure_threshold(self) -> None:
        """Test circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_available is False

    def test_success_resets_failure_count(self) -> None:
        """Test success after failures keeps circuit closed."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        # Should still be closed
        assert breaker.state == CircuitState.CLOSED

        # Count should be reset
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

    def test_recovery_timeout_transitions_to_half_open(self) -> None:
        """Test circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for testing
        )

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        import time
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self) -> None:
        """Test success in half-open state closes circuit."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for half-open
        import time
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN

        # Success should close it
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Test failure in half-open state reopens circuit."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for half-open
        import time
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen it
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_context_manager_records_success(self) -> None:
        """Test async context manager records success."""
        breaker = CircuitBreaker(name="test")

        async with breaker:
            pass  # No exception

        assert breaker._success_count == 1
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_records_failure(self) -> None:
        """Test async context manager records failure."""
        breaker = CircuitBreaker(name="test")

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker._success_count == 0
        assert breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_raises_when_open(self) -> None:
        """Test async context manager raises when circuit is open."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        with pytest.raises(APIError) as exc_info:
            async with breaker:
                pass

        assert "OPEN" in str(exc_info.value)

    def test_get_stats(self) -> None:
        """Test get_stats returns proper structure."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)
        breaker.record_success()
        breaker.record_failure()

        stats = breaker.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1
        assert stats["total_calls"] == 2

    def test_reset_clears_state(self) -> None:
        """Test reset returns circuit to initial state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0


class TestRetryAsync:
    """Tests for async retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        """Test no retry needed on success."""
        call_count = 0

        @retry_async(max_attempts=3)
        async def succeed_immediately() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed_immediately()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Test retry on failure."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = await fail_twice()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        """Test raises exception after max attempts exceeded."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        with pytest.raises(ValueError) as exc_info:
            await always_fail()

        assert "permanent failure" in str(exc_info.value)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self) -> None:
        """Test only retries on specified exception types."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def raise_type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            await raise_type_error()

        # Should fail immediately without retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        """Test on_retry callback is called."""
        retry_calls = []

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            retry_calls.append((attempt, str(exc), delay))

        @retry_async(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        async def fail_twice() -> str:
            if len(retry_calls) < 2:
                raise ValueError("fail")
            return "success"

        await fail_twice()

        assert len(retry_calls) == 2
        assert retry_calls[0][0] == 1  # First retry
        assert retry_calls[1][0] == 2  # Second retry


class TestRetrySync:
    """Tests for sync retry decorator."""

    def test_success_on_first_try(self) -> None:
        """Test no retry needed on success."""
        call_count = 0

        @retry_sync(max_attempts=3)
        def succeed_immediately() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_immediately()

        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self) -> None:
        """Test retry on failure."""
        call_count = 0

        @retry_sync(max_attempts=3, base_delay=0.01)
        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = fail_twice()

        assert result == "success"
        assert call_count == 3


class TestErrorAggregator:
    """Tests for error aggregator."""

    def test_records_error(self) -> None:
        """Test error recording."""
        aggregator = ErrorAggregator()

        error = ValueError("test error")
        aggregator.record_error(error)

        summary = aggregator.get_summary()
        assert summary["total_error_types"] == 1
        assert "ValueError" in summary["by_type"]
        assert summary["by_type"]["ValueError"]["count"] == 1

    def test_tracks_multiple_error_types(self) -> None:
        """Test tracking multiple error types."""
        aggregator = ErrorAggregator()

        aggregator.record_error(ValueError("val error"))
        aggregator.record_error(TypeError("type error"))
        aggregator.record_error(ValueError("val error 2"))

        summary = aggregator.get_summary()
        assert summary["total_error_types"] == 2
        assert summary["by_type"]["ValueError"]["count"] == 2
        assert summary["by_type"]["TypeError"]["count"] == 1

    def test_error_rate_calculation(self) -> None:
        """Test error rate per minute calculation."""
        aggregator = ErrorAggregator(window_size=timedelta(minutes=1))

        # Record 6 errors
        for _ in range(6):
            aggregator.record_error(ValueError("test"))

        rate = aggregator.get_error_rate()
        assert rate == 6.0  # 6 errors in 1 minute window

    def test_alert_callback_triggered(self) -> None:
        """Test alert callback is triggered at threshold."""
        aggregator = ErrorAggregator(alert_threshold=3)
        alerts = []

        def on_alert(error_type: str, count: int) -> None:
            alerts.append((error_type, count))

        aggregator.set_alert_callback(on_alert)

        for _ in range(3):
            aggregator.record_error(ValueError("test"))

        assert len(alerts) == 1
        assert alerts[0] == ("ValueError", 3)

    def test_reset_clears_errors(self) -> None:
        """Test reset clears all error data."""
        aggregator = ErrorAggregator()

        aggregator.record_error(ValueError("test"))
        assert aggregator.get_summary()["total_error_types"] == 1

        aggregator.reset()
        assert aggregator.get_summary()["total_error_types"] == 0


class TestHealthCheck:
    """Tests for health check dataclass."""

    def test_healthy_check(self) -> None:
        """Test healthy health check."""
        check = HealthCheck(
            component="test",
            status=ComponentHealth.HEALTHY,
        )
        assert check.is_healthy is True

    def test_unhealthy_check(self) -> None:
        """Test unhealthy health check."""
        check = HealthCheck(
            component="test",
            status=ComponentHealth.UNHEALTHY,
            message="component failed",
        )
        assert check.is_healthy is False


class TestRecoveryManager:
    """Tests for recovery manager."""

    def test_register_component(self) -> None:
        """Test component registration."""
        manager = RecoveryManager()
        manager.register_component("audio")

        summary = manager.get_health_summary()
        assert "audio" in summary["components"]
        assert summary["components"]["audio"]["status"] == "unknown"

    def test_update_health(self) -> None:
        """Test health status update."""
        manager = RecoveryManager()
        manager.register_component("audio")

        manager.update_health("audio", ComponentHealth.HEALTHY)

        summary = manager.get_health_summary()
        assert summary["components"]["audio"]["status"] == "healthy"

    def test_tracks_consecutive_failures(self) -> None:
        """Test consecutive failure tracking."""
        manager = RecoveryManager()
        manager.register_component("audio")

        manager.update_health("audio", ComponentHealth.UNHEALTHY, "fail 1")
        manager.update_health("audio", ComponentHealth.UNHEALTHY, "fail 2")
        manager.update_health("audio", ComponentHealth.UNHEALTHY, "fail 3")

        summary = manager.get_health_summary()
        assert summary["components"]["audio"]["consecutive_failures"] == 3

    def test_success_resets_failure_count(self) -> None:
        """Test health recovery resets failure count."""
        manager = RecoveryManager()
        manager.register_component("audio")

        manager.update_health("audio", ComponentHealth.UNHEALTHY, "fail")
        manager.update_health("audio", ComponentHealth.UNHEALTHY, "fail")
        manager.update_health("audio", ComponentHealth.HEALTHY)

        summary = manager.get_health_summary()
        assert summary["components"]["audio"]["consecutive_failures"] == 0

    def test_recovery_action_called(self) -> None:
        """Test recovery action is called."""
        manager = RecoveryManager()
        recovery_called = False

        def recover() -> bool:
            nonlocal recovery_called
            recovery_called = True
            return True

        manager.register_component("audio", recovery_action=recover)
        manager.update_health("audio", ComponentHealth.UNHEALTHY)

        result = manager.attempt_recovery("audio")

        assert recovery_called is True
        assert result is True
        assert manager._health_checks["audio"].status == ComponentHealth.HEALTHY

    def test_recovery_action_fails(self) -> None:
        """Test failed recovery action."""
        manager = RecoveryManager()

        def fail_recover() -> bool:
            return False

        manager.register_component("audio", recovery_action=fail_recover)
        manager.update_health("audio", ComponentHealth.UNHEALTHY)

        result = manager.attempt_recovery("audio")

        assert result is False
        assert manager._health_checks["audio"].status == ComponentHealth.UNHEALTHY

    def test_max_recovery_attempts_exceeded(self) -> None:
        """Test recovery stops after max attempts."""
        manager = RecoveryManager()
        manager._max_recovery_attempts = 2

        def never_recover() -> bool:
            return False

        manager.register_component("audio", recovery_action=never_recover)

        # Simulate multiple failures
        for _ in range(3):
            manager.update_health("audio", ComponentHealth.UNHEALTHY)

        # Should refuse to attempt recovery
        result = manager.attempt_recovery("audio")
        assert result is False

    def test_overall_health_all_healthy(self) -> None:
        """Test overall health when all components healthy."""
        manager = RecoveryManager()
        manager.register_component("audio")
        manager.register_component("wake_word")

        manager.update_health("audio", ComponentHealth.HEALTHY)
        manager.update_health("wake_word", ComponentHealth.HEALTHY)

        assert manager.get_overall_health() == ComponentHealth.HEALTHY

    def test_overall_health_one_unhealthy(self) -> None:
        """Test overall health when one component unhealthy."""
        manager = RecoveryManager()
        manager.register_component("audio")
        manager.register_component("wake_word")

        manager.update_health("audio", ComponentHealth.HEALTHY)
        manager.update_health("wake_word", ComponentHealth.UNHEALTHY)

        assert manager.get_overall_health() == ComponentHealth.UNHEALTHY

    def test_overall_health_one_degraded(self) -> None:
        """Test overall health when one component degraded."""
        manager = RecoveryManager()
        manager.register_component("audio")
        manager.register_component("wake_word")

        manager.update_health("audio", ComponentHealth.HEALTHY)
        manager.update_health("wake_word", ComponentHealth.DEGRADED)

        assert manager.get_overall_health() == ComponentHealth.DEGRADED
