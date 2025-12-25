# Ubuntu Headless Deployment Guide

This guide covers deploying the Reachy Mini Conversation App as a systemd service on Ubuntu for headless (no GUI) operation.

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

### Backup

```bash
# Backup all Lily data
tar -czvf lily-backup-$(date +%Y%m%d).tar.gz ~/.reachy_mini
```

### Restore

```bash
# Stop service first
systemctl --user stop lily-conversation

# Restore backup
tar -xzvf lily-backup-YYYYMMDD.tar.gz -C ~/

# Restart service
systemctl --user start lily-conversation
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
