"""Headless mode for Reachy Mini Conversation App.

This module provides a fully headless runtime for running the conversation
system without any UI. It includes:
- Offline wake word detection ("Hey Lily")
- State machine (SLEEPING/AWAKE)
- Audio routing and pipeline
- Full conversation transcription
- Systemd service integration
"""

from reachy_mini_conversation_app.headless.state_machine import ConversationState, StateMachine
from reachy_mini_conversation_app.headless.wake_word import WakeWordEngine
from reachy_mini_conversation_app.headless.audio_router import AudioRouter
from reachy_mini_conversation_app.headless.transcription import TranscriptionService


__all__ = [
    "ConversationState",
    "StateMachine",
    "WakeWordEngine",
    "AudioRouter",
    "TranscriptionService",
]
