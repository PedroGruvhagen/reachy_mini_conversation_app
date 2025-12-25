# Ubuntu Headless Deployment Guide

This guide covers deploying the Reachy Mini Conversation App as a systemd service on Ubuntu for headless (no GUI) operation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Service Management](#service-management)
- [Troubleshooting](#troubleshooting)
- [CLI Commands](#cli-commands)
- [Security Considerations](#security-considerations)
- [Backup and Restore](#backup-and-restore)
- [Error Handling and Recovery](#error-handling-and-recovery)
- [Headless Module API](#headless-module-api)
- [Performance Tuning](#performance-tuning)

## Prerequisites

- Ubuntu 22.04 or 24.04 LTS
- Python 3.11+
- Working microphone and speakers
- OpenAI API key

## Hardware Requirements

- **Minimum**: 4GB RAM, 2-core CPU (Raspberry Pi 4 compatible)
- **Recommended**: 8GB RAM, 4-core CPU

## Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/PedroGruvhagen/reachy_mini_conversation_app.git
cd reachy_mini_conversation_app

# 2. Create and activate virtual environment
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# 3. Install the package
pip install -e .

# 4. Run the installer
bash deploy/scripts/install_service.sh

# 5. Configure your API key
nano ~/.reachy_mini/.env
# Add: OPENAI_API_KEY=sk-your-key-here

# 6. Start the service
systemctl --user start lily-conversation
```

## Detailed Installation

### Step 1: System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install audio dependencies
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libsndfile1-dev \
    pulseaudio \
    alsa-utils

# Install OpenCV dependencies (for face recognition)
sudo apt install -y \
    libopencv-dev \
    python3-opencv
```

### Step 2: Audio Configuration

```bash
# Add user to audio group
sudo usermod -aG audio $USER

# Enable user lingering (service runs without login)
sudo loginctl enable-linger $USER

# Verify audio devices
arecord -l  # List recording devices
aplay -l    # List playback devices
```

### Step 3: Python Environment

```bash
# Create virtual environment in home directory
python3 -m venv ~/.venv

# Activate it
source ~/.venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the conversation app
pip install -e /path/to/reachy_mini_conversation_app
```

### Step 4: Configuration

Create the configuration directory and file:

```bash
mkdir -p ~/.reachy_mini/{logs,recordings,models}
```

Edit `~/.reachy_mini/.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Wake Word Detection
WAKE_WORD_CONFIDENCE_THRESHOLD=0.5
WAKE_WORD_COOLDOWN_SECONDS=10

# State Machine
SILENCE_TIMEOUT_SECONDS=120
MAX_AWAKE_SECONDS=1800
MIN_SLEEP_SECONDS=5

# Audio
AUDIO_INPUT_DEVICE=
AUDIO_SAMPLE_RATE=48000

# Transcription
ENABLE_RECORDING=true
AUDIO_RETENTION_DAYS=30
TRANSCRIPT_RETENTION_DAYS=90

# Logging
LOG_LEVEL=INFO
LOG_FILE=~/.reachy_mini/logs/conversation.log
```

### Step 5: Install Systemd Service

```bash
# Run the installer
bash deploy/scripts/install_service.sh
```

This will:
1. Add your user to the audio group
2. Enable user lingering
3. Create required directories
4. Install the systemd service
5. Enable auto-start on boot

## Service Management

### Basic Commands

```bash
# Start the service
systemctl --user start lily-conversation

# Stop the service
systemctl --user stop lily-conversation

# Restart the service
systemctl --user restart lily-conversation

# Check status
systemctl --user status lily-conversation

# Disable auto-start
systemctl --user disable lily-conversation

# Enable auto-start
systemctl --user enable lily-conversation
```

### Viewing Logs

```bash
# Real-time journal logs
journalctl --user -u lily-conversation -f

# Application log file
tail -f ~/.reachy_mini/logs/conversation.log

# Error log
tail -f ~/.reachy_mini/logs/conversation-error.log

# Last 100 lines
journalctl --user -u lily-conversation -n 100
```

## Troubleshooting

### Service Won't Start

1. **Check logs**:
   ```bash
   journalctl --user -u lily-conversation -n 50
   ```

2. **Verify Python environment**:
   ```bash
   ~/.venv/bin/reachy-mini-headless --version
   ```

3. **Test manual run**:
   ```bash
   source ~/.venv/bin/activate
   reachy-mini-headless
   ```

### No Audio

1. **Check audio devices**:
   ```bash
   arecord -l
   aplay -l
   ```

2. **Verify user is in audio group**:
   ```bash
   groups | grep audio
   ```

3. **Test microphone**:
   ```bash
   arecord -d 5 test.wav && aplay test.wav
   ```

4. **Check PulseAudio/PipeWire**:
   ```bash
   systemctl --user status pulseaudio
   # or
   systemctl --user status pipewire
   ```

### Wake Word Not Detecting

1. **Check confidence threshold** in `~/.reachy_mini/.env`:
   ```bash
   WAKE_WORD_CONFIDENCE_THRESHOLD=0.3  # Lower = more sensitive
   ```

2. **Test wake word manually**:
   ```bash
   source ~/.venv/bin/activate
   python scripts/test_wake_word.py
   ```

3. **Check for ambient noise** - the system filters very quiet audio

### OpenAI API Errors

1. **Verify API key**:
   ```bash
   cat ~/.reachy_mini/.env | grep OPENAI
   ```

2. **Test API connection**:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

## CLI Commands

The `lily-cli` tool provides management capabilities over SSH:

```bash
# User management
lily-cli user list
lily-cli user add "Name" --photo photo.jpg
lily-cli user delete "Name" --force

# Memory management
lily-cli memory list
lily-cli memory search "keywords"
lily-cli memory facts --pending

# Transcript management
lily-cli transcript search "topic"
lily-cli transcript show SESSION_ID
lily-cli transcript export SESSION_ID -o output.json

# System status
lily-cli status
lily-cli version
```

## Security Considerations

### API Key Protection

- Never commit `.env` files
- Use environment-specific keys
- Rotate keys periodically

The headless module includes built-in security utilities for API key handling:

```python
from reachy_mini_conversation_app.headless import APIKeyValidator

# Validate OpenAI API key format (returns tuple of is_valid, message)
is_valid, message = APIKeyValidator.validate_openai_key(api_key)
if is_valid:
    # Use the key
    pass
else:
    print(f"Invalid key: {message}")

# Mask keys for safe logging
masked = APIKeyValidator.mask_key(api_key)  # "********xyz12345"

# Get key hash for comparison/logging
key_hash = APIKeyValidator.get_key_hash(api_key)
```

### Input Sanitization

Protect against injection attacks and path traversal:

```python
from reachy_mini_conversation_app.headless import InputSanitizer

# Sanitize user input text
clean_text = InputSanitizer.sanitize_text(user_input, max_length=1000)

# Sanitize filenames (removes dangerous characters)
safe_filename = InputSanitizer.sanitize_filename(user_filename)

# Validate and sanitize file paths (prevents path traversal)
safe_path = InputSanitizer.sanitize_path(
    user_path,
    base_dir=Path("~/.reachy_mini").expanduser()
)

# Check if a command is safe to execute
if InputSanitizer.is_safe_command(user_command):
    # Command has no dangerous shell characters
    pass
```

### Secure File Handling

The module provides secure file operations with proper permissions:

```python
from reachy_mini_conversation_app.headless import SecureFileHandler

# Ensure directory has secure permissions (0o700)
SecureFileHandler.secure_directory(Path("~/.reachy_mini").expanduser())

# Secure file permissions (0o600 for sensitive files)
SecureFileHandler.secure_file(Path("~/.reachy_mini/.env"), file_type="env_file")

# Atomic secure write (temp file + rename)
SecureFileHandler.write_secure(
    Path("~/.reachy_mini/config.json"),
    config_data,
    file_type="config"
)

# Check if file is world-readable (security issue)
if SecureFileHandler.is_world_readable(env_path):
    print("Warning: sensitive file is world-readable!")

# Check .env file for security issues
warnings = SecureFileHandler.check_env_file_security(env_path)
```

### Rate Limiting

Protect against API abuse:

```python
from reachy_mini_conversation_app.headless import RateLimiter, rate_limited

# Create a rate limiter (10 requests per 60-second window)
limiter = RateLimiter(max_requests=10, window_seconds=60)

# Check if request is allowed
if limiter.allow():
    # Make the request
    pass

# Check remaining requests
remaining = limiter.remaining()

# Use as decorator for automatic rate limiting
@rate_limited(max_requests=5, window_seconds=30)
def api_call():
    pass
```

### Secure Logging

Automatically redact sensitive data from logs:

```python
import logging
from reachy_mini_conversation_app.headless import SecureLogger, SecureLoggingFilter

# Method 1: Use SecureLogger static methods directly
message = f"Using API key: {api_key}"
redacted = SecureLogger.redact(message)  # API key is replaced with <REDACTED_API_KEY>

# Method 2: Add filter to logging handlers for automatic redaction
handler = logging.StreamHandler()
handler.addFilter(SecureLoggingFilter())
logger.addHandler(handler)

# Now all logs through this handler are automatically redacted
logger.info(f"Email: {email}")  # Email redacted as <REDACTED_EMAIL>
```

### Security Auditing

Audit your installation for security issues:

```python
from reachy_mini_conversation_app.headless import SecurityAuditor

# Audit environment variables and configuration
issues = SecurityAuditor.audit_environment()
for severity, issue, recommendation in issues:
    print(f"[{severity}] {issue}")
    print(f"  Fix: {recommendation}")

# Audit file permissions in a directory
file_issues = SecurityAuditor.audit_file_permissions(Path("~/.reachy_mini"))
for severity, issue, recommendation in file_issues:
    print(f"[{severity}] {issue}")

# Generate a full security report
report = SecurityAuditor.generate_report()
print(report)
```

### Systemd Hardening (Optional)

The service file includes optional security settings:

```ini
# Uncomment in lily-conversation.service for stricter security
ProtectHome=read-only
ProtectSystem=strict
ReadWritePaths=%h/.reachy_mini
NoNewPrivileges=true
```

## Backup and Restore

The headless module includes a comprehensive backup system with checksums, encryption support, and automatic rotation.

### Quick Backup Commands

```bash
# Using the CLI
lily-cli backup create
lily-cli backup list
lily-cli backup restore BACKUP_NAME
```

### Programmatic Backup

```python
from pathlib import Path
from reachy_mini_conversation_app.headless import (
    BackupManager,
    create_backup,
    restore_latest_backup,
    list_available_backups,
)

# Create a backup manager
manager = BackupManager(
    backup_dir=Path("~/.reachy_mini/backups").expanduser(),
    data_dir=Path("~/.reachy_mini").expanduser(),
    max_backups=10,  # Keep last 10 backups (auto-rotation)
)

# Create a full backup (config + transcripts)
backup_path = manager.create_backup(backup_type="full")
print(f"Backup created: {backup_path}")

# Create a config-only backup
config_backup = manager.create_backup(backup_type="config")

# Create a transcripts-only backup
transcript_backup = manager.create_backup(backup_type="transcripts")

# Include audio recordings (can be large)
full_backup = manager.create_backup(backup_type="full", include_audio=True)

# List available backups
for backup in manager.list_backups():
    print(f"{backup.path.name}: {backup.total_size} bytes, created {backup.created_at}")
```

### Restore from Backup

```python
# Verify backup integrity before restoring
is_valid, errors = manager.verify_backup(backup_path)
if not is_valid:
    print(f"Backup corrupted: {errors}")

# Restore from a specific backup file
result = manager.restore_backup(
    backup_path=Path("~/.reachy_mini/backups/backup_full_20241225_120000.tar.gz"),
    verify_checksums=True,   # Verify file integrity
    overwrite=False,         # Skip existing files
)

if result.success:
    print(f"Restored {len(result.files_restored)} files")
    for f in result.files_restored:
        print(f"  - {f}")
else:
    print(f"Restore failed: {result.errors}")

# Files that were skipped (already exist)
for f in result.files_skipped:
    print(f"Skipped (exists): {f}")
```

### Convenience Functions

```python
# Quick one-liner backup with default settings
backup_path = create_backup(backup_type="full")

# Restore the most recent backup of a specific type
result = restore_latest_backup(backup_type="full", overwrite=False)

# List all available backups
backups = list_available_backups()
for backup in backups:
    print(f"{backup.path.name}: {backup.backup_type}")
```

### Backup Features

- **Checksums**: SHA256 verification for data integrity
- **Manifest**: JSON manifest with file list and metadata
- **Rotation**: Automatic cleanup of old backups (configurable max_backups)
- **Atomic Operations**: Uses temp files and rename for safety
- **Secure Permissions**: Backups created with 0o600 permissions
- **Import/Export**: Copy backups to/from external locations

### Manual Backup (Alternative)

If you prefer manual backups:

```bash
# Backup all Lily data
tar -czvf lily-backup-$(date +%Y%m%d).tar.gz ~/.reachy_mini
```

### Manual Restore

```bash
# Stop service first
systemctl --user stop lily-conversation

# Restore backup
tar -xzvf lily-backup-YYYYMMDD.tar.gz -C ~/

# Restart service
systemctl --user start lily-conversation
```

## Error Handling and Recovery

The headless module includes robust error handling with circuit breakers, automatic retries, and health monitoring.

### Exception Hierarchy

```python
from reachy_mini_conversation_app.headless import (
    HeadlessError,      # Base exception
    AudioDeviceError,   # Microphone/speaker issues
    WakeWordError,      # Wake word detection problems
    TranscriptionError, # Speech-to-text failures
    APIError,           # OpenAI API errors
    ConfigurationError, # Configuration issues
)

try:
    await wake_word_engine.start()
except AudioDeviceError as e:
    print(f"Audio problem: {e}")
except WakeWordError as e:
    print(f"Wake word issue: {e}")
```

### Circuit Breaker Pattern

Prevent cascading failures with circuit breakers:

```python
from reachy_mini_conversation_app.headless import CircuitBreaker, CircuitState

# Create a circuit breaker
breaker = CircuitBreaker(
    name="openai_api",
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30.0,    # Try again after 30 seconds
    half_open_requests=1,     # Allow 1 test request when half-open
)

# Method 1: Use as async context manager
async def call_api():
    async with breaker:
        result = await openai_client.chat(...)
        return result

# Method 2: Manual recording
async def call_api_manual():
    if not breaker.is_available:
        raise APIError("Circuit breaker is open")

    try:
        result = await openai_client.chat(...)
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure(e)
        raise

# Check circuit state
if breaker.state == CircuitState.OPEN:
    print("API is temporarily unavailable")

# Get statistics
stats = breaker.get_stats()
print(f"Total calls: {stats['total_calls']}, Failures: {stats['failure_count']}")

# Reset circuit breaker
breaker.reset()
```

### Automatic Retries

Retry failed operations with exponential backoff:

```python
from reachy_mini_conversation_app.headless import retry_async, retry_sync

# Async retry decorator
@retry_async(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    retryable_exceptions=(APIError, TimeoutError),
)
async def fetch_data():
    return await api.get_data()

# Sync retry decorator
@retry_sync(max_attempts=5, base_delay=0.5)
def connect_audio():
    return audio_device.connect()
```

### Health Monitoring

Monitor component health across the system:

```python
from reachy_mini_conversation_app.headless import (
    ErrorAggregator,
    RecoveryManager,
    ComponentHealth,
    error_aggregator,  # Global instance
    recovery_manager,  # Global instance
)

# Record errors with optional context
error_aggregator.record_error(
    error=AudioDeviceError("Microphone disconnected"),
    context="audio_router",  # Optional context string
)

# Get error rate per minute
rate = error_aggregator.get_error_rate()  # All errors
rate_api = error_aggregator.get_error_rate("APIError")  # Specific error type

# Get error summary for monitoring
summary = error_aggregator.get_summary()
print(f"Errors in window: {summary['errors_in_window']}")
print(f"Error rate: {summary['error_rate_per_minute']}/min")

# Set up alert callback for threshold monitoring
def on_alert(error_type: str, count: int):
    print(f"Alert: {error_type} occurred {count} times!")

error_aggregator.set_alert_callback(on_alert)

# Reset error statistics
error_aggregator.reset()
```

### Recovery Manager

Automatic recovery actions for failed components:

```python
# Register a component with a recovery action
def restart_audio() -> bool:
    # Attempt to restart audio - return True if successful
    try:
        audio_router.restart()
        return True
    except Exception:
        return False

recovery_manager.register_component(
    name="audio_router",
    recovery_action=restart_audio,
)

# Update component health status
recovery_manager.update_health(
    name="audio_router",
    status=ComponentHealth.DEGRADED,
    message="High latency detected",
)

# Attempt recovery
if recovery_manager.attempt_recovery("audio_router"):
    print("Recovery successful!")

# Get overall system health
overall = recovery_manager.get_overall_health()
print(f"System health: {overall.value}")

# Get detailed health summary
health_summary = recovery_manager.get_health_summary()
print(health_summary)
```

### Combined Error Handling Example

```python
from reachy_mini_conversation_app.headless import (
    CircuitBreaker,
    retry_async,
    error_aggregator,
    recovery_manager,
    ComponentHealth,
    APIError,
)

# Setup circuit breaker
api_breaker = CircuitBreaker("api", failure_threshold=3, recovery_timeout=30)

# Register API recovery action
def recover_api() -> bool:
    api_breaker.reset()
    return True

recovery_manager.register_component("api", recovery_action=recover_api)

@retry_async(max_attempts=3, exceptions=(APIError,))
async def resilient_api_call(prompt: str) -> str:
    if not api_breaker.is_available:
        error_aggregator.record_error(APIError("Circuit open"), "api")
        recovery_manager.attempt_recovery("api")
        raise APIError("Service temporarily unavailable")

    try:
        async with api_breaker:
            result = await openai_client.chat(prompt)
            recovery_manager.update_health("api", ComponentHealth.HEALTHY)
            return result
    except Exception as e:
        error_aggregator.record_error(e, "api")
        recovery_manager.update_health("api", ComponentHealth.UNHEALTHY, str(e))
        raise
```

## Headless Module API

The headless module provides a complete programmatic API for building custom conversation applications.

### Core Components

#### Wake Word Detection

Offline wake word detection using OpenWakeWord:

```python
from reachy_mini_conversation_app.headless import WakeWordEngine

# Create engine with custom settings
engine = WakeWordEngine(
    model_path=Path("~/.reachy_mini/models/hey_lily.onnx"),
    confidence_threshold=0.5,
    cooldown_seconds=10,
)

# Set callback for detection
engine.on_wake_word = lambda confidence: print(f"Wake word detected! ({confidence:.2%})")

# Start listening
await engine.start()

# Process audio frames
await engine.process_audio(audio_chunk)

# Stop when done
await engine.stop()
```

#### State Machine

Manage conversation states (SLEEPING/AWAKE):

```python
from reachy_mini_conversation_app.headless import StateMachine, ConversationState

# Create state machine
state_machine = StateMachine(
    silence_timeout=120.0,    # Return to sleep after 2 min silence
    max_awake_duration=1800,  # Max 30 min conversation
    min_sleep_duration=5.0,   # Stay asleep for at least 5 sec
)

# Register state change callback
@state_machine.on_state_change
def handle_state(old_state, new_state):
    if new_state == ConversationState.AWAKE:
        print("Lily is now listening!")
    else:
        print("Lily is now sleeping...")

# Wake up on wake word
await state_machine.wake_up()

# Check current state
if state_machine.state == ConversationState.AWAKE:
    # Process conversation
    pass

# Go back to sleep
await state_machine.sleep()
```

#### Audio Router

Manage audio input/output pipelines:

```python
from reachy_mini_conversation_app.headless import AudioRouter

# Create audio router
router = AudioRouter(
    input_device="default",
    output_device="default",
    sample_rate=48000,
    channels=1,
    block_size=1024,
)

# Start audio capture
await router.start()

# Get audio chunks
async for chunk in router.audio_stream():
    # Process chunk (numpy array)
    process_audio(chunk)

# Play audio response
await router.play_audio(response_audio)

# Stop when done
await router.stop()
```

#### Transcription Service

Record and transcribe conversations:

```python
from reachy_mini_conversation_app.headless import TranscriptionService

# Create transcription service
transcription = TranscriptionService(
    output_dir=Path("~/.reachy_mini/recordings"),
    enable_recording=True,
    retention_days=30,
)

# Start a new session
session_id = await transcription.start_session()

# Add transcript entries
await transcription.add_entry(
    session_id=session_id,
    speaker="user",
    text="Hello, Lily!",
    timestamp=datetime.now(),
)

await transcription.add_entry(
    session_id=session_id,
    speaker="lily",
    text="Hello! How can I help you today?",
    timestamp=datetime.now(),
)

# End session
await transcription.end_session(session_id)

# Search transcripts
results = await transcription.search("weather forecast")
```

### Complete Example

Here's a minimal headless conversation system:

```python
import asyncio
from pathlib import Path
from reachy_mini_conversation_app.headless import (
    WakeWordEngine,
    StateMachine,
    AudioRouter,
    TranscriptionService,
    ConversationState,
    CircuitBreaker,
    error_aggregator,
)

async def main():
    # Initialize components
    wake_word = WakeWordEngine()
    state_machine = StateMachine()
    audio = AudioRouter()
    transcription = TranscriptionService()
    api_breaker = CircuitBreaker("openai", failure_threshold=3)

    # Wire up callbacks
    @wake_word.on_wake_word
    async def on_wake(confidence):
        await state_machine.wake_up()
        session_id = await transcription.start_session()

    @state_machine.on_state_change
    async def on_state(old, new):
        if new == ConversationState.SLEEPING:
            await transcription.end_session()

    # Start all components
    await asyncio.gather(
        wake_word.start(),
        audio.start(),
    )

    # Main loop
    try:
        async for chunk in audio.audio_stream():
            # Always feed wake word engine
            await wake_word.process_audio(chunk)

            # If awake, process conversation
            if state_machine.state == ConversationState.AWAKE:
                # Your conversation logic here
                pass

    except KeyboardInterrupt:
        pass
    finally:
        await wake_word.stop()
        await audio.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Module Exports

All headless utilities are available from the main module:

```python
from reachy_mini_conversation_app.headless import (
    # Core components
    ConversationState,
    StateMachine,
    WakeWordEngine,
    AudioRouter,
    TranscriptionService,

    # Exceptions
    HeadlessError,
    AudioDeviceError,
    WakeWordError,
    TranscriptionError,
    APIError,
    ConfigurationError,

    # Error handling
    CircuitBreaker,
    CircuitState,
    retry_async,
    retry_sync,
    ErrorAggregator,
    RecoveryManager,
    ComponentHealth,
    error_aggregator,
    recovery_manager,

    # Security
    APIKeyValidator,
    SecureFileHandler,
    InputSanitizer,
    SecureLogger,
    SecureLoggingFilter,
    RateLimiter,
    rate_limited,
    SecurityAuditor,

    # Backup
    BackupManifest,
    BackupInfo,
    RestoreResult,
    BackupManager,
    create_backup,
    restore_latest_backup,
    list_available_backups,
)
```

## Uninstallation

```bash
# Run uninstaller
bash deploy/scripts/uninstall_service.sh

# Optional: Remove all data
rm -rf ~/.reachy_mini
```

## Performance Tuning

### CPU Usage

If CPU usage is too high:

```bash
# Lower wake word sensitivity
WAKE_WORD_CONFIDENCE_THRESHOLD=0.7

# Increase audio block size
AUDIO_BLOCK_SIZE=2048
```

### Memory Usage

The service is limited to 2GB RAM by default. Adjust in the service file:

```ini
MemoryMax=1G
MemoryHigh=512M
```

## Support

- **Issues**: [GitHub Issues](https://github.com/PedroGruvhagen/reachy_mini_conversation_app/issues)
- **Logs**: Always include output from `journalctl --user -u lily-conversation`
