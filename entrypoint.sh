#!/usr/bin/env bash
# Entrypoint — passes all CLI arguments to frigate-sidecar.
exec /usr/local/bin/frigate-sidecar "$@"
