# ---- Builder ----
FROM docker.io/library/ubuntu:26.04 AS builder

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
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
FROM docker.io/library/ubuntu:26.04

# Minimal runtime deps: TFLite 2.14.1, GOMP, Teflon delegate.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        libgomp1 \
        libtensorflow-lite2.14.1 \
        mesa-teflon-delegate \
        && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /models

COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/frigate-sidecar /usr/local/bin/
COPY --from=builder /build/entrypoint.sh /entrypoint.sh

# Teflon delegate and TFLite 2.14.1 are installed above.
ENV TEFLON_LIB=/usr/lib/teflon/libteflon.so
ENV TFLITE_LIB=/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--endpoint", "tcp://0.0.0.0:5555"]
