#!/bin/sh
set -e

echo "Starting Ryvion Node Agent (CPU mode)..."
echo "Hub URL: ${AK_HUB_URL:-https://ryvion-hub.onrender.com}"
echo "Device Type: ${AK_DEVICE_TYPE:-cpu}"
echo "UI Port: ${AK_UI_PORT:-3000}"

# Install basic Python AI dependencies inside a throwaway virtualenv to avoid PEP 668 issues
echo "Installing Python AI libraries..."
if command -v python3 >/dev/null 2>&1; then
  VENV_DIR=/tmp/ak_py_venv
  python3 -m venv "$VENV_DIR" >/dev/null 2>&1 || true
  if [ -f "$VENV_DIR/bin/activate" ]; then
    . "$VENV_DIR/bin/activate"
    pip install --no-cache-dir requests numpy >/dev/null 2>&1 || echo "Warning: Failed to install some dependencies"
    deactivate || true
  else
    echo "Warning: Python venv unavailable; skipping optional Python deps"
  fi
else
  echo "Warning: python3 not found; skipping optional Python deps"
fi

# Start the node agent in CPU mode (no Docker containers)
echo "Starting node agent in CPU mode..."
exec /usr/local/bin/node-agent \
    -hub "${AK_HUB_URL:-https://ryvion-hub.onrender.com}" \
    -type "${AK_DEVICE_TYPE:-cpu}" \
    -ui-port "${AK_UI_PORT:-3000}"