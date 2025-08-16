FROM golang:1.22 as build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd cmd/node-agent && go build -tags containers -o /out/node-agent

# Use Docker-in-Docker for container execution capabilities
FROM docker:dind

# Install additional dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    python3 \
    py3-pip

# Copy the built binary
COPY --from=build /out/node-agent /usr/local/bin/node-agent

# Create app user
RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -D appuser -G appgroup

# Create necessary directories
RUN mkdir -p /work /var/log/ryvion && \
    chown -R appuser:appgroup /work /var/log/ryvion

# Environment variables
ENV AK_HUB_URL="https://ryvion-hub.onrender.com"
ENV AK_DEVICE_TYPE="gpu"
ENV AK_UI_PORT="3000"

# Expose UI port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep node-agent || exit 1

# Switch to app user for security
USER appuser

ENTRYPOINT ["/usr/local/bin/node-agent"]

