FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    iputils-ping \
    iperf3 \
    tcpdump \
    tshark \
    procps \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lab
