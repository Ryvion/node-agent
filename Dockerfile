FROM golang:1.22 as build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd cmd/node-agent && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -tags containers -o /out/node-agent

# Use Alpine for lightweight runtime
FROM alpine:latest

# Install dependencies for AI workloads
RUN apk add --no-cache \
    curl \
    ca-certificates \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    musl-dev

# Create app user first (before copying binary)
RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -D appuser -G appgroup

# Create necessary directories
RUN mkdir -p /work /var/log/ryvion && \
    chown -R appuser:appgroup /work /var/log/ryvion

# Copy the built binary and startup script
COPY --from=build /out/node-agent /usr/local/bin/node-agent
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/node-agent /usr/local/bin/start.sh

# Environment variables
ENV AK_HUB_URL="https://ryvion-hub.onrender.com"
ENV AK_DEVICE_TYPE="cpu"
ENV AK_UI_PORT="3000"

# Expose UI port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep node-agent || exit 1

# Run as root for Docker access (required for docker:dind)
ENTRYPOINT ["/usr/local/bin/start.sh"]

