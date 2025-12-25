"""Headless mode for Reachy Mini Conversation App.

This module provides a fully headless runtime for running the conversation
system without any UI. It includes:
- Offline wake word detection ("Hey Lily")
- State machine (SLEEPING/AWAKE)
- Audio routing and pipeline
- Full conversation transcription
- Systemd service integration
- Error handling and recovery utilities
"""

from reachy_mini_conversation_app.headless.state_machine import ConversationState, StateMachine
from reachy_mini_conversation_app.headless.wake_word import WakeWordEngine
from reachy_mini_conversation_app.headless.audio_router import AudioRouter
from reachy_mini_conversation_app.headless.transcription import TranscriptionService
from reachy_mini_conversation_app.headless.error_handling import (
    # Exceptions
    HeadlessError,
    AudioDeviceError,
    WakeWordError,
    TranscriptionError,
    APIError,
    ConfigurationError,
    # Circuit Breaker
    CircuitBreaker,
    CircuitState,
    # Retry decorators
    retry_async,
    retry_sync,
    # Monitoring
    ErrorAggregator,
    RecoveryManager,
    ComponentHealth,
    error_aggregator,
    recovery_manager,
)


__all__ = [
    # Core components
    "ConversationState",
    "StateMachine",
    "WakeWordEngine",
    "AudioRouter",
    "TranscriptionService",
    # Exception types
    "HeadlessError",
    "AudioDeviceError",
    "WakeWordError",
    "TranscriptionError",
    "APIError",
    "ConfigurationError",
    # Error handling utilities
    "CircuitBreaker",
    "CircuitState",
    "retry_async",
    "retry_sync",
    "ErrorAggregator",
    "RecoveryManager",
    "ComponentHealth",
    # Global instances
    "error_aggregator",
    "recovery_manager",
]
