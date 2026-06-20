#!/usr/bin/env bash
# Arize Phoenix observability UI on :6006 (CPU, never competes for the GPU).
set -euo pipefail
python -m phoenix.server.main serve
