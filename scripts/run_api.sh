#!/usr/bin/env bash
# Backend API on :8000. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
