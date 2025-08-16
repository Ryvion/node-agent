#!/bin/sh
set -e

echo "Starting Ryvion Node Agent..."
echo "Hub URL: ${AK_HUB_URL:-https://ryvion-hub.onrender.com}"
echo "Device Type: ${AK_DEVICE_TYPE:-gpu}"
echo "UI Port: ${AK_UI_PORT:-3000}"

# Start Docker daemon in background
echo "Starting Docker daemon..."
dockerd-entrypoint.sh &
sleep 10

# Wait for Docker daemon to be ready
echo "Waiting for Docker daemon..."
until docker info >/dev/null 2>&1; do
    echo "Docker daemon not ready, waiting..."
    sleep 2
done
echo "Docker daemon ready"

# Start the node agent with explicit parameters
echo "Starting node agent..."
exec /usr/local/bin/node-agent \
    -hub "${AK_HUB_URL:-https://ryvion-hub.onrender.com}" \
    -type "${AK_DEVICE_TYPE:-gpu}" \
    -ui-port "${AK_UI_PORT:-3000}"