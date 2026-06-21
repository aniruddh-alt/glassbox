#!/usr/bin/env bash
# Train persona-vector probes (Family B) from JSON artifacts under backend/science/artifacts/.
#
# Usage:
#   ./scripts/train_probe_artifacts.sh                  # train templates missing direction
#   ./scripts/train_probe_artifacts.sh --force          # retrain all artifacts
#   ./scripts/train_probe_artifacts.sh --artifact backend/science/artifacts/harmful.json
#
# Requires cached HF weights when offline:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./scripts/train_probe_artifacts.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

ARGS=()
if [[ "${1:-}" == "--force" ]]; then
  ARGS+=(--all --force)
  shift
elif [[ "${1:-}" == "--all" ]]; then
  ARGS+=(--all)
  shift
elif [[ -n "${1:-}" ]]; then
  ARGS+=(--artifact "$1")
  shift
else
  ARGS+=(--all)
fi

echo "Training probes (provider=engine, layers=${LAYERS:-9,17,22,29}) ..."
uv run python -m backend.validation.harmfulness_pipeline \
  "${ARGS[@]}" \
  --provider engine \
  --layers "${LAYERS:-9,17,22,29}" \
  --direction-method "${DIRECTION_METHOD:-raw}" \
  --min-auroc "${MIN_AUROC:-0.7}" \
  "$@"

echo
echo "Loaded trackers:"
uv run python -c "from backend.science import persona; persona.clear_trackers(); print(persona.load_artifacts())"
