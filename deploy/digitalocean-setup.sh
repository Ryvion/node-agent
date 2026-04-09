set -e

echo "Ryvion DePIN Node Agent - DigitalOcean Setup"
echo "=============================================="

echo "Updating system packages..."
apt-get update && apt-get upgrade -y

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh

    systemctl enable docker
    systemctl start docker

    usermod -aG docker $USER
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

echo "Creating application directories..."
mkdir -p /opt/ryvion/{config,data,logs}
cd /opt/ryvion

echo "Downloading latest ryvion-node binary..."
RELEASE_URL="https://api.github.com/repos/Ryvion/node-agent/releases/latest"
DOWNLOAD_URL=$(curl -s $RELEASE_URL | grep "browser_download_url.*ryvion-node.*linux-amd64" | cut -d '"' -f 4)

if [ -n "$DOWNLOAD_URL" ]; then
    curl -L $DOWNLOAD_URL -o ryvion-node
    chmod +x ryvion-node
    echo "ryvion-node downloaded"
else
    echo "Using Docker image instead of binary"
fi

cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  ryvion-node:
    image: ghcr.io/ryvion/node-agent:latest
    container_name: ryvion-node
    restart: unless-stopped
    environment:
      - RYV_HUB_URL=https://api.ryvion.ai
      - RYV_DEVICE_TYPE=gpu
      - RYV_GPUS=auto
      - RYV_LOG_LEVEL=info
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./data:/work
      - ./logs:/var/log/ryvion
    networks:
      - ryvion-net
    healthcheck:
      test: ["CMD-SHELL", "pgrep ryvion-node >/dev/null || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ryvion-net:
    driver: bridge

volumes:
  ryvion-data:
  ryvion-logs:
EOF

cat > /etc/systemd/system/ryvion-node.service << 'EOF'
[Unit]
Description=Ryvion DePIN Node Agent
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ryvion
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

echo "Setting up systemd service..."
systemctl daemon-reload
systemctl enable ryvion-node.service

chown -R 1001:1001 /opt/ryvion
chmod -R 755 /opt/ryvion

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Start the service: sudo systemctl start ryvion-node"
echo "2. Check status: sudo systemctl status ryvion-node"
echo "3. View logs: sudo docker-compose -f /opt/ryvion/docker-compose.yml logs -f"
echo "4. Verify heartbeat: sudo docker-compose -f /opt/ryvion/docker-compose.yml logs --tail=50"
echo ""
echo "To customize configuration:"
echo "- Edit env values in: /opt/ryvion/docker-compose.yml"
echo "- Restart: sudo systemctl restart ryvion-node"
echo ""
echo "Droplet Requirements:"
echo "- Minimum: 2GB RAM, 1 vCPU"
echo "- Recommended: 4GB RAM, 2 vCPU"
echo "- For AI workloads: 8GB+ RAM"
