# Ryvion DePIN Node Agent - DigitalOcean Deployment

## Quick Start

### 1. Create DigitalOcean Droplet

**Recommended Configuration:**
- **Image**: Ubuntu 22.04 LTS
- **Size**: Basic - 4GB RAM, 2 vCPUs ($24/month)
- **Region**: Choose closest to your location
- **Additional Options**: 
  - ✅ Monitoring
  - ✅ IPv6
  - 🔑 Add your SSH key

### 2. Deploy Node Agent

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/Ryvion/node-agent/main/deploy/digitalocean-setup.sh | bash

# Start the service
systemctl start ryvion-node

# Check status
systemctl status ryvion-node
```

### 3. Verify Installation

```bash
# Check if containers are running
docker ps

# View logs
docker-compose -f /opt/ryvion/docker-compose.yml logs -f

# Access UI (replace with your droplet IP)
curl http://YOUR_DROPLET_IP:3000/health
```

## Architecture

```
┌─────────────────────────────────────────┐
│           DigitalOcean Droplet          │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │         Node Agent              │    │
│  │  ┌─────────────────────────┐    │    │
│  │  │     Docker Engine       │    │    │
│  │  │  ┌─────────────────┐    │    │    │
│  │  │  │  AI Runners     │    │    │    │
│  │  │  │  - LLM          │    │    │    │
│  │  │  │  - Image Gen    │    │    │    │
│  │  │  │  - Audio/Video  │    │    │    │
│  │  │  └─────────────────┘    │    │    │
│  │  └─────────────────────────┘    │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
           │
           │ HTTPS
           ▼
┌─────────────────────────────────────────┐
│              Render                     │
│  ┌─────────────────────────────────┐    │
│  │        Hub Orchestrator         │    │
│  │         + Database              │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Pricing Comparison

| Provider | Config | Monthly Cost | Docker-in-Docker | GPU Support |
|----------|--------|-------------|------------------|-------------|
| **DigitalOcean** | 4GB/2CPU | $24 | ✅ Yes | Available |
| Render | Standard | $25 | ❌ No | ❌ No |
| AWS EC2 | t3.medium | ~$30 | ✅ Yes | Extra cost |
| Railway | 4GB/2CPU | ~$35 | ✅ Yes | ❌ No |

## Node Operator Benefits

### For Individual Operators
- **Easy Setup**: One-command deployment
- **Low Cost**: $24/month for basic node
- **Full Control**: Root access, custom configuration
- **Monitoring**: Built-in health checks and logs

### For Enterprise Operators
- **Scalable**: Deploy multiple nodes across regions
- **Automated**: Systemd service management
- **Secure**: Isolated environments per node
- **Profitable**: Earn tokens for AI workload processing

## Troubleshooting

### Common Issues

**Node not connecting to hub:**
```bash
# Check hub URL configuration
cat /opt/ryvion/config/node.json

# Test connectivity
curl -I https://ryvion-hub.onrender.com/health
```

**Docker containers not starting:**
```bash
# Check Docker daemon
systemctl status docker

# Check privileges
docker run --rm --privileged hello-world
```

**Port not accessible:**
```bash
# Check firewall (Ubuntu UFW)
ufw status
ufw allow 3000/tcp

# Check if service is listening
netstat -tlnp | grep :3000
```

## Advanced Configuration

### GPU Support
For AI workloads requiring GPU acceleration:

1. **Create GPU Droplet** (when available)
2. **Install NVIDIA Docker**:
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install nvidia-docker2
   apt-get update && apt-get install -y nvidia-docker2
   systemctl restart docker
   ```

3. **Update docker-compose.yml**:
   ```yaml
   runtime: nvidia
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
   ```

### Custom Runners
Add your own AI model runners by mounting custom containers:

```yaml
volumes:
  - ./custom-runners:/custom-runners
environment:
  - AK_CUSTOM_RUNNERS_PATH=/custom-runners
```

## Support

- **Documentation**: [docs.ryvion.io](https://docs.ryvion.io)
- **Discord**: [discord.gg/ryvion](https://discord.gg/ryvion)
- **GitHub Issues**: [github.com/Ryvion/node-agent/issues](https://github.com/Ryvion/node-agent/issues)