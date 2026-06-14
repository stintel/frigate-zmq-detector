#!/usr/bin/env bash
# Entrypoint — passes all CLI arguments to frigate-zmq-detector.
exec /usr/local/bin/frigate-zmq-detector "$@"
