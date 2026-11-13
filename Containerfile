
# SPDX-License-Identifier: Apache-2.0 OR MIT

# ---- Builder ----
# Build for the target architecture. Buildx/QEMU handles ARM64 builds.
FROM docker.io/library/ubuntu:26.04 AS builder

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:$PATH

RUN curl -fSL https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    rustup default stable

WORKDIR /build
COPY . .

# Compile for the current build architecture
RUN cargo build --locked --release && \
    mkdir -p /out && \
    cp target/release/frigate-zmq-detector /out/

# ---- Runtime ----
FROM docker.io/library/ubuntu:26.04 AS runtime

ARG TARGETARCH

ENV EDGETPU_RELEASE=16.0TF2.19.1-1

# Install common runtime dependencies
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libtensorflow-lite2.14.1 \
        libusb-1.0-0; \
    \
    # Teflon is only available for ARM64
    if [ "${TARGETARCH}" = "arm64" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            mesa-teflon-delegate; \
    fi; \
    \
    # Select EdgeTPU package architecture
    case "${TARGETARCH}" in \
        amd64) EDGETPU_ARCH="amd64" ;; \
        arm64) EDGETPU_ARCH="arm64" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}"; exit 1 ;; \
    esac; \
    \
    # Download EdgeTPU runtime
    curl -fsSL \
        "https://github.com/feranick/libedgetpu/releases/download/${EDGETPU_RELEASE}/libedgetpu1-std_16.0tf2.19.1-1.ubuntu24.04_${EDGETPU_ARCH}.deb" \
        -o /tmp/libedgetpu1-std.deb; \
    \
    # Install EdgeTPU runtime and resolve dependencies
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        /tmp/libedgetpu1-std.deb; \
    \
    rm -f /tmp/libedgetpu1-std.deb; \
    \
    # Map Debian multiarch directory
    case "${TARGETARCH}" in \
        amd64) TFLITE_ARCH="x86_64-linux-gnu" ;; \
        arm64) TFLITE_ARCH="aarch64-linux-gnu" ;; \
    esac; \
    \
    # Provide a stable TFLite library path
    test -f "/usr/lib/${TFLITE_ARCH}/libtensorflow-lite.so.2.14.1"; \
    ln -s \
        "/usr/lib/${TFLITE_ARCH}/libtensorflow-lite.so.2.14.1" \
        /usr/lib/libtensorflow-lite.so.2.14.1; \
    \
    # Remove build-time download utility
    apt-get purge -y curl; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /models

# Application
COPY --from=builder /out/frigate-zmq-detector /usr/local/bin/
COPY --from=builder /build/entrypoint.sh /entrypoint.sh

# Runtime configuration
ENV BACKEND=edgetpu \
    EDGETPU_LIB=/usr/lib/libedgetpu.so.1.0 \
    TEFLON_LIB=/usr/lib/teflon/libteflon.so \
    TFLITE_LIB=/usr/lib/libtensorflow-lite.so.2.14.1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--endpoint", "tcp://0.0.0.0:5555"]
