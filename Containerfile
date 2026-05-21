# ---- Builder ----
FROM docker.io/library/ubuntu:24.04 AS builder

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        pkg-config \
        libzmq3-dev \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup (stable, aarch64-unknown-linux-gnu target).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:$PATH
RUN curl -fSL https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    rustup default stable && rustup target add aarch64-unknown-linux-gnu

WORKDIR /build
COPY . .
RUN cargo build --target aarch64-unknown-linux-gnu --release

# ---- Runtime ----
FROM docker.io/library/ubuntu:24.04

# Minimal runtime deps.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        libzmq5 \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /models

COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/frigate-sidecar /usr/local/bin/
COPY --from=builder /build/entrypoint.sh /entrypoint.sh

# Teflon delegate path (set by host mount at runtime).
ENV TEFLON_LIB=/usr/lib/teflon/libteflon.so
ENV TFLITE_LIB=/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so
ENV MODEL_DIR=/models

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--endpoint", "tcp://0.0.0.0:5555"]
