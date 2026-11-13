
# SPDX-License-Identifier: Apache-2.0 OR MIT

# ---- Builder ----
FROM --platform=$BUILDPLATFORM docker.io/library/ubuntu:26.04 AS builder

ARG TARGETARCH

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Rust installation
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:$PATH

RUN curl -fSL https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    rustup default stable

# Map Docker architecture to Rust target
RUN case "${TARGETARCH}" in \
        amd64) RUST_TARGET="x86_64-unknown-linux-gnu" ;; \
        arm64) RUST_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    rustup target add "${RUST_TARGET}"

WORKDIR /build
COPY . .

# Cross-compilation
RUN case "${TARGETARCH}" in \
        amd64) RUST_TARGET="x86_64-unknown-linux-gnu" ;; \
        arm64) RUST_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) exit 1 ;; \
    esac && \
    cargo build --target "${RUST_TARGET}" --release && \
    mkdir -p /out && \
    cp "target/${RUST_TARGET}/release/frigate-zmq-detector" /out/

# ---- Runtime ----
FROM docker.io/library/ubuntu:26.04

ARG TARGETARCH

ENV EDGETPU_RELEASE=16.0TF2.19.1-1

# Architecture-specific EdgeTPU package
RUN case "${TARGETARCH}" in \
        amd64) EDGETPU_ARCH="amd64" ;; \
        arm64) EDGETPU_ARCH="arm64" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        curl \
        libgomp1 \
        libtensorflow-lite2.14.1 \
        libusb-1.0-0 \
        mesa-teflon-delegate \
    && curl -fsSL \
        "https://github.com/feranick/libedgetpu/releases/download/${EDGETPU_RELEASE}/libedgetpu1-std_16.0tf2.19.1-1.ubuntu24.04_${EDGETPU_ARCH}.deb" \
        -o /tmp/libedgetpu1-std.deb \
    && dpkg -i /tmp/libedgetpu1-std.deb \
    && rm /tmp/libedgetpu1-std.deb \
    && apt-get remove --purge -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models

COPY --from=builder /out/frigate-zmq-detector /usr/local/bin/
COPY --from=builder /build/entrypoint.sh /entrypoint.sh

# Delegate and TFLite library paths
ENV EDGETPU_LIB=/usr/lib/libedgetpu.so.1.0 \
    TEFLON_LIB=/usr/lib/teflon/libteflon.so

# Architecture-specific TFLite library path
ENV TFLITE_LIB=/usr/lib/${TARGETARCH/amd64/x86_64}/libtensorflow-lite.so.2.14.1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--endpoint", "tcp://0.0.0.0:5555"]
