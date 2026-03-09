FROM golang:1.26.1 AS build
WORKDIR /src
COPY . .
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd cmd/ryvion-node && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags "-s -w -X main.version=${VERSION}" -o /out/ryvion-node

FROM alpine:3.21

RUN apk add --no-cache \
    curl \
    ca-certificates \
    docker-cli

RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -D appuser -G appgroup

RUN mkdir -p /work /var/log/ryvion && \
    chown -R appuser:appgroup /work /var/log/ryvion

COPY --from=build /out/ryvion-node /usr/local/bin/ryvion-node
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/ryvion-node /usr/local/bin/start.sh

ENV RYV_HUB_URL="https://ryvion-hub.fly.dev"
ENV RYV_DEVICE_TYPE="cpu"
ENV RYV_GPUS="auto"

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep ryvion-node || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]
