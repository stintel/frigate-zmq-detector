#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Entrypoint — passes all CLI arguments to frigate-zmq-detector.
exec /usr/local/bin/frigate-zmq-detector "$@"
