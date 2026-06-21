# Persona-vector probe training (Family B)

Offline pipeline that turns a **probe artifact JSON** into a live scorer: contrastive activations → diff-of-means direction → calibrated LogReg → AUROC validation → writeback to `backend/science/artifacts/*.json`.

At runtime, `gpu_service` calls `persona.load_artifacts()` on startup and `score_all_trackers()` scores each turn using **layer-17** residual activations.

---

## Artifact format

Each file under `backend/science/artifacts/` needs:

| Field | Purpose |
|-------|---------|
| `id` | Tracker id (`harmful`, `uncertainty`, …) |
| `instruction[0].pos` / `.neg` | Contrastive system prompts (harmful vs safe, uncertain vs confident, …) |
| `questions` | Medical prompts (≥4 recommended) |
| `alert_direction` | `"high"` (flag when score ≥ threshold) or `"low"` (risk awareness) |
| `eval_prompt` | Judge template for future expansion (not used by current trainer) |

After training, the pipeline adds: `direction`, `threshold`, `layer`, `auroc`, `validation`.

---

## Quick commands

### Train all templates missing `direction`

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./scripts/train_probe_artifacts.sh
```

### Retrain everything at layer 17 (runtime hook)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LAYERS=17 ./scripts/train_probe_artifacts.sh --all --force
```

### Train with normalized diff-of-means (recommended)

Matches the external Colab sweep fix (z-score each dimension before computing the direction):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LAYERS=17 \
  uv run python -m backend.validation.harmfulness_pipeline \
  --all --force --direction-method normed
```

Artifacts will include `direction_method`, `norm_mean`, and `norm_std`. Runtime scoring applies the same z-score before projection.

---

## External CSV sweep alignment

If you have (or export) `harmful.csv` + `benign.csv` with a `prompt` column, use the GlassBox-native port of the nnsight Colab script:

```bash
# Export from harmful.json contrastive pairs
uv run python -m backend.validation.csv_probe_sweep \
  --export-from-artifact backend/science/artifacts/harmful.json

# Layer sweep 12–20 (slow — one forward+generate per prompt per layer)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python -m backend.validation.csv_probe_sweep \
    --layers 12,13,14,15,16,17,18,19,20 \
    --pool response \
    --write-artifact backend/science/artifacts/harmful.json
```

| | External nnsight script | GlassBox `harmfulness_pipeline` | GlassBox `csv_probe_sweep` |
|--|-------------------------|----------------------------------|----------------------------|
| Model | `google/gemma-3-4b-it` | `unsloth/gemma-3-4b-it` (same weights) | same as pipeline |
| Hook lib | nnsight `VisionLanguageModel` | `backend.engine` transformers hook | `backend.engine` |
| Data | separate harmful/benign CSV corpora | contrastive pos/neg system prompts × shared questions | CSV (exportable from artifact) |
| Pooling | mean over **prompt** tokens | mean over **response** tokens after generation | `--pool prompt` or `response` |
| Direction | **normed** diff-of-means + CV | `--direction-method raw` (default) or `normed` | reports raw + normed + LR; can write normed artifact |
| Layer sweep | 12–20 | 9,17,22,29 (config) | configurable |
| Runtime | pickle bundle (`mean`, `std`, `direction`) | JSON artifact loaded by `gpu_service` | same JSON path |

**What aligns:** normed diff-of-means math, LogReg baseline, layer sweep, Gemma-3 4B IT.

**What differs (important):** GlassBox scores **response-token** activations at the live hook layer (17) during real chat. The Colab script scores **prompt-only** mean activations unless you use `--pool response` in `csv_probe_sweep`.

**Recommendation:** use `--direction-method normed` + `LAYERS=17` in the artifact pipeline for production; use `csv_probe_sweep --pool response` when reproducing the external sweep methodology.

---

```bash
# 1. Create artifact (API or copy a template)
curl -s -X POST http://localhost:8000/api/track \
  -H 'Content-Type: application/json' \
  -d '{"concept":"my_risk","description":"Detects X in clinical answers"}'

# 2. Edit backend/science/artifacts/my_risk.json — add/refine questions if needed

# 3. Train
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python -m backend.validation.harmfulness_pipeline \
  --artifact backend/science/artifacts/my_risk.json \
  --provider engine --layers 17
```

### Verify artifacts load

```bash
uv run python -c "
from backend.science import persona
persona.clear_trackers()
print(persona.load_artifacts())
"
```

---

## Deploy trained probes to the GPU pod

After training locally, sync artifacts and restart `gpu_service`:

```bash
# Over Tailscale (replace IP if needed)
rsync -avz backend/science/artifacts/ root@100.98.245.123:/workspace/glassbox/backend/science/artifacts/

ssh root@100.98.245.123 'pkill -f "uvicorn backend.gpu_service" || true; cd /workspace/glassbox && \
  export HF_HOME=/workspace/.cache/huggingface POD_TOKEN=glassbox-dev-secret PYTHONPATH=/workspace/glassbox && \
  nohup .venv/bin/uvicorn backend.gpu_service:app --host 127.0.0.1 --port 8000 > gpu_service.log 2>&1 &'

# Confirm trackers loaded
curl -s http://localhost:8001/health
# Chat turn should now include trackers in cognition_event
```

---

## Built-in probes

| Artifact | Tracker id | Alert direction |
|----------|------------|-----------------|
| `harmful.json` | `harmful` | high |
| `uncertainty.json` | `uncertainty` | high (maps to event uncertainty meter) |
| `hallucination.json` | `hallucination` | high |
| `risk_awareness.json` | `risk_awareness` | low |

`events.build_cognition_event` maps `trackers["uncertainty"]` to the top-level `uncertainty` / `flag` fields.

---

## Caveats

1. **Small contrastive sets** (8–16 examples) can yield perfect AUROC without real generalization. Treat AUROC as a smoke test; expand `questions` before trusting scores.

2. **Layer must match runtime** — scoring uses `config.LAYER` (17). Prefer `--layers 17` or ensure the layer sweep tie-break picks 17 (pipeline prefers `config.LAYER` on equal AUROC).

3. **HF offline** — if DNS is blocked, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and ensure the model is cached under `~/.cache/huggingface/`.

4. **Calibrator not persisted** — artifacts store `direction` + `threshold`; live scoring uses direction projection + sigmoid unless a calibrator is registered in-memory. Full LogReg weights are not yet serialized to JSON.

5. **Custom probes via UI** — `POST /api/track` writes the artifact template; training is still an offline step (`train_probe_artifacts.sh`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cannot train persona direction: positive/negative means are identical` | More/diverse `questions`; check pos/neg prompts differ |
| `best AUROC below minimum` | Lower `--min-auroc` for smoke tests only |
| Chat shows `trackers: {}` | Artifacts lack `direction`; run training + sync to pod + restart gpu_service |
| `uncertainty: null` in event | No `uncertainty` tracker loaded or id mismatch |
