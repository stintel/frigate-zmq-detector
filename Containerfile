# SPDX-License-Identifier: Apache-2.0 OR MIT

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
# EdgeTPU (std) .deb from feranick/libedgetpu — Ubuntu 26.04 is not yet
# covered by upstream releases, so we install the 24.04 deb directly with dpkg.
ENV EDGETPU_RELEASE=16.0TF2.19.1-1
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        curl \
        libgomp1 \
        libtensorflow-lite2.14.1 \
        libusb-1.0-0 \
        mesa-teflon-delegate \
        && curl -fsSL "https://github.com/feranick/libedgetpu/releases/download/${EDGETPU_RELEASE}/libedgetpu1-std_16.0tf2.19.1-1.ubuntu24.04_arm64.deb" -o /tmp/libedgetpu1-std.deb && \
    dpkg -i /tmp/libedgetpu1-std.deb && \
    rm /tmp/libedgetpu1-std.deb && \
    apt-get remove --purge -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /models

COPY --from=builder /build/target/aarch64-unknown-linux-gnu/release/frigate-zmq-detector /usr/local/bin/
COPY --from=builder /build/entrypoint.sh /entrypoint.sh

# Delegate and TFLite library paths.
ENV EDGETPU_LIB=/usr/lib/libedgetpu.so.1.0
ENV TEFLON_LIB=/usr/lib/teflon/libteflon.so
ENV TFLITE_LIB=/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--endpoint", "tcp://0.0.0.0:5555"]
