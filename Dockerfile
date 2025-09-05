FROM golang:1.25.0 as build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd cmd/node-agent && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -tags containers -o /out/node-agent

FROM alpine:latest

RUN apk add --no-cache \
    curl \
    ca-certificates \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    musl-dev

RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -D appuser -G appgroup

RUN mkdir -p /work /var/log/ryvion && \
    chown -R appuser:appgroup /work /var/log/ryvion

COPY --from=build /out/node-agent /usr/local/bin/node-agent
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/node-agent /usr/local/bin/start.sh

ENV AK_HUB_URL="https://ryvion-hub.onrender.com"
ENV AK_DEVICE_TYPE="cpu"
ENV AK_UI_PORT="3000"

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep node-agent || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]

