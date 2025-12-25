#!/bin/bash
# Uninstall Lily Conversation Service
#
# This script removes the systemd user service.
# Does NOT remove ~/.reachy_mini data (logs, recordings, config)
#
# Usage: bash scripts/uninstall_service.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Lily Conversation Service Uninstaller  ${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Step 1: Stop the service if running
echo -e "${BLUE}Step 1: Stopping service...${NC}"
if systemctl --user is-active --quiet lily-conversation.service 2>/dev/null; then
    systemctl --user stop lily-conversation.service
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo -e "${YELLOW}Service was not running${NC}"
fi
echo ""

# Step 2: Disable the service
echo -e "${BLUE}Step 2: Disabling service...${NC}"
if systemctl --user is-enabled --quiet lily-conversation.service 2>/dev/null; then
    systemctl --user disable lily-conversation.service
    echo -e "${GREEN}✓ Service disabled${NC}"
else
    echo -e "${YELLOW}Service was not enabled${NC}"
fi
echo ""

# Step 3: Remove service file
echo -e "${BLUE}Step 3: Removing service file...${NC}"
if [[ -f ~/.config/systemd/user/lily-conversation.service ]]; then
    rm ~/.config/systemd/user/lily-conversation.service
    echo -e "${GREEN}✓ Service file removed${NC}"
else
    echo -e "${YELLOW}Service file not found${NC}"
fi
echo ""

# Step 4: Reload systemd daemon
echo -e "${BLUE}Step 4: Reloading systemd...${NC}"
systemctl --user daemon-reload
echo -e "${GREEN}✓ Systemd daemon reloaded${NC}"
echo ""

# Step 5: Remove logrotate config if installed
echo -e "${BLUE}Step 5: Removing logrotate configuration...${NC}"
if [[ -f /etc/logrotate.d/lily-conversation ]]; then
    if command -v sudo &> /dev/null; then
        sudo rm /etc/logrotate.d/lily-conversation
        echo -e "${GREEN}✓ Logrotate configuration removed${NC}"
    else
        echo -e "${YELLOW}Cannot remove /etc/logrotate.d/lily-conversation (no sudo)${NC}"
    fi
else
    echo -e "${YELLOW}No logrotate configuration found${NC}"
fi
echo ""

# Final summary
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Uninstallation Complete!  ${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "The service has been removed."
echo ""
echo -e "${YELLOW}Note: Your data has NOT been removed:${NC}"
echo -e "  - Configuration: ~/.reachy_mini/.env"
echo -e "  - Logs: ~/.reachy_mini/logs/"
echo -e "  - Recordings: ~/.reachy_mini/recordings/"
echo -e "  - Database: ~/.reachy_mini/lily_memory.db"
echo ""
echo -e "To completely remove all data, run:"
echo -e "  ${RED}rm -rf ~/.reachy_mini${NC}"
echo ""
