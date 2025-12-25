#!/bin/bash
# Install Lily Conversation Service for Ubuntu Headless Operation
#
# This script installs the systemd user service for auto-start.
# Prerequisites:
#   - Python venv at ~/.venv with reachy-mini-headless installed
#   - User must be in audio group for microphone access
#
# Usage: bash scripts/install_service.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Lily Conversation Service Installer  ${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Linux/Ubuntu.${NC}"
    echo -e "${YELLOW}Current OS: $OSTYPE${NC}"
    echo ""
fi

# Get script and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}Project directory: $PROJECT_DIR${NC}"
echo ""

# Step 1: Check if user is in audio group
echo -e "${BLUE}Step 1: Checking audio group membership...${NC}"
if groups | grep -q audio; then
    echo -e "${GREEN}✓ User is already in audio group${NC}"
else
    echo -e "${YELLOW}Adding user to audio group (requires sudo)...${NC}"
    sudo usermod -aG audio "$USER"
    echo -e "${GREEN}✓ User added to audio group${NC}"
    echo -e "${YELLOW}NOTE: You may need to log out and back in for group changes to take effect${NC}"
fi
echo ""

# Step 2: Enable user lingering (allows service to run without active login)
echo -e "${BLUE}Step 2: Enabling user lingering...${NC}"
if command -v loginctl &> /dev/null; then
    sudo loginctl enable-linger "$USER" 2>/dev/null || true
    echo -e "${GREEN}✓ User lingering enabled${NC}"
else
    echo -e "${YELLOW}Warning: loginctl not found, skipping linger setup${NC}"
fi
echo ""

# Step 3: Create required directories with secure permissions
echo -e "${BLUE}Step 3: Creating directories...${NC}"
mkdir -p ~/.reachy_mini
chmod 700 ~/.reachy_mini  # Restrict to owner only (protects API keys)
mkdir -p ~/.reachy_mini/logs
mkdir -p ~/.reachy_mini/recordings
mkdir -p ~/.reachy_mini/models
mkdir -p ~/.config/systemd/user
echo -e "${GREEN}✓ Created ~/.reachy_mini/{logs,recordings,models}${NC}"
echo -e "${GREEN}✓ Set secure permissions (700) on ~/.reachy_mini${NC}"
echo -e "${GREEN}✓ Created ~/.config/systemd/user${NC}"
echo ""

# Step 4: Copy .env template if not exists
echo -e "${BLUE}Step 4: Checking environment configuration...${NC}"
if [[ -f ~/.reachy_mini/.env ]]; then
    echo -e "${GREEN}✓ Environment file exists at ~/.reachy_mini/.env${NC}"
else
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        cp "$PROJECT_DIR/.env.example" ~/.reachy_mini/.env
        echo -e "${GREEN}✓ Created ~/.reachy_mini/.env from template${NC}"
        echo -e "${YELLOW}IMPORTANT: Edit ~/.reachy_mini/.env to add your OPENAI_API_KEY${NC}"
    else
        # Create minimal .env
        cat > ~/.reachy_mini/.env << 'EOF'
# Lily Conversation Service Configuration
# Edit this file before starting the service

# REQUIRED: OpenAI API Key
OPENAI_API_KEY=sk-YOUR-KEY-HERE

# Wake Word Configuration
WAKE_WORD_CONFIDENCE_THRESHOLD=0.5

# State Machine
SILENCE_TIMEOUT_SECONDS=120
MAX_AWAKE_SECONDS=1800

# Logging
LOG_LEVEL=INFO
LOG_FILE=~/.reachy_mini/logs/conversation.log

# Audio Device (leave empty for default)
AUDIO_INPUT_DEVICE=

# Transcription
ENABLE_RECORDING=true
AUDIO_RETENTION_DAYS=30
TRANSCRIPT_RETENTION_DAYS=90
EOF
        echo -e "${GREEN}✓ Created minimal ~/.reachy_mini/.env${NC}"
        echo -e "${YELLOW}IMPORTANT: Edit ~/.reachy_mini/.env to add your OPENAI_API_KEY${NC}"
    fi
fi
echo ""

# Step 5: Check for virtual environment
echo -e "${BLUE}Step 5: Checking virtual environment...${NC}"
if [[ -f ~/.venv/bin/reachy-mini-headless ]]; then
    echo -e "${GREEN}✓ Found reachy-mini-headless at ~/.venv/bin${NC}"
else
    echo -e "${YELLOW}Warning: ~/.venv/bin/reachy-mini-headless not found${NC}"
    echo -e "${YELLOW}Please install with:${NC}"
    echo -e "${YELLOW}  python -m venv ~/.venv${NC}"
    echo -e "${YELLOW}  source ~/.venv/bin/activate${NC}"
    echo -e "${YELLOW}  pip install -e $PROJECT_DIR${NC}"
    echo ""
fi
echo ""

# Step 6: Copy systemd service file
echo -e "${BLUE}Step 6: Installing systemd service...${NC}"
if [[ -f "$PROJECT_DIR/deploy/systemd/lily-conversation.service" ]]; then
    cp "$PROJECT_DIR/deploy/systemd/lily-conversation.service" ~/.config/systemd/user/
    echo -e "${GREEN}✓ Copied service file to ~/.config/systemd/user/${NC}"
else
    echo -e "${RED}Error: Service file not found at $PROJECT_DIR/deploy/systemd/lily-conversation.service${NC}"
    exit 1
fi
echo ""

# Step 7: Reload systemd and enable service
echo -e "${BLUE}Step 7: Enabling systemd service...${NC}"
systemctl --user daemon-reload
systemctl --user enable lily-conversation.service
echo -e "${GREEN}✓ Service enabled${NC}"
echo ""

# Step 8: Install logrotate configuration (if available)
echo -e "${BLUE}Step 8: Installing logrotate configuration...${NC}"
if [[ -f "$PROJECT_DIR/deploy/logrotate/lily-conversation" ]]; then
    if [[ -d /etc/logrotate.d ]] && command -v sudo &> /dev/null; then
        sudo cp "$PROJECT_DIR/deploy/logrotate/lily-conversation" /etc/logrotate.d/
        echo -e "${GREEN}✓ Logrotate configuration installed${NC}"
    else
        echo -e "${YELLOW}Skipping logrotate (no /etc/logrotate.d or no sudo)${NC}"
    fi
else
    echo -e "${YELLOW}No logrotate configuration found, skipping${NC}"
fi
echo ""

# Final summary
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Installation Complete!  ${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${YELLOW}Edit ~/.reachy_mini/.env${NC} to configure your settings"
echo -e "     (especially OPENAI_API_KEY)"
echo ""
echo -e "  2. Start the service:"
echo -e "     ${BLUE}systemctl --user start lily-conversation${NC}"
echo ""
echo -e "  3. Check status:"
echo -e "     ${BLUE}systemctl --user status lily-conversation${NC}"
echo ""
echo -e "  4. View logs:"
echo -e "     ${BLUE}journalctl --user -u lily-conversation -f${NC}"
echo -e "     or"
echo -e "     ${BLUE}tail -f ~/.reachy_mini/logs/conversation.log${NC}"
echo ""
echo -e "  5. To stop:"
echo -e "     ${BLUE}systemctl --user stop lily-conversation${NC}"
echo ""
echo -e "The service will automatically start on boot."
echo -e "Say 'Hey Lily' to wake her up!"
echo ""
