#!/bin/sh
set -e

echo "Starting Ryvion Node Agent (CPU mode)..."
echo "Hub URL: ${AK_HUB_URL:-https://ryvion-hub.onrender.com}"
echo "Device Type: ${AK_DEVICE_TYPE:-cpu}"
echo "UI Port: ${AK_UI_PORT:-3000}"

# Install basic Python AI dependencies
echo "Installing Python AI libraries..."
pip3 install --no-cache-dir requests numpy || echo "Warning: Failed to install some dependencies"

# Start the node agent in CPU mode (no Docker containers)
echo "Starting node agent in CPU mode..."
exec /usr/local/bin/node-agent \
    -hub "${AK_HUB_URL:-https://ryvion-hub.onrender.com}" \
    -type "${AK_DEVICE_TYPE:-cpu}" \
    -ui-port "${AK_UI_PORT:-3000}"