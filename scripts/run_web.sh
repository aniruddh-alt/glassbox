#!/usr/bin/env bash
# Frontend dev server on :5173 (proxies /api -> :8000).
set -euo pipefail
cd "$(dirname "$0")/../frontend"
npm run dev
