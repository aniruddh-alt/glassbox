<div align="center">

<img src="docs/assets/logo.svg" alt="GlassBox" width="72" height="72" />

# GlassBox

**Interpretability-grade observability for open-weight LLMs.**

*Surface uncertainty, never suppress it.*

[![License: MIT](https://img.shields.io/badge/License-MIT-000.svg)](LICENSE)

</div>

---

GlassBox lets you watch an open-weight model's internal state while it answers. It runs the model with a single forward hook on one layer and turns that activation into two signals per turn:

- **Feature cloud** — the top SAE features firing in the residual stream, i.e. which concepts are active. Exploratory: labels are auto-interp and always shown with a caveat.
- **Probes** — calibrated linear probes (diff-of-means + logistic regression) that score concepts such as over-confidence or harmful intent, and flag when the model is *internally uncertain but verbally confident*.

You can train your own probe from a plain-language description on the **Build** tab. Every turn emits one structured event (`CognitionEvent`, see `backend/schema.py`) that the UI renders and Sentry alerts on when a probe trips.

GlassBox is general-purpose. A medical clinical-decision-support setup ships as one labeled, opt-in example profile in `config.example.yaml`.

## Architecture

Two FastAPI processes with a hard split:

- **Orchestration backend** (`backend/app.py`) runs on CPU and never imports torch. It assembles the per-turn event and fans it out.
- **GPU pod service** (`backend/gpu_service.py`) owns torch. It generates with the model while one hook on the configured layer captures the residual stream; that single activation feeds both the SAE feature cloud and the probes.

```
UI ──POST /api/chat──▶  GPU pod: generate + layer hook
                              │ (one residual activation)
                   ┌──────────┴──────────┐
              SAE feature cloud     persona-vector probes
                   └──────────┬──────────┘
                   backend: build one CognitionEvent
                              │ fanout
                   ┌──────────┼───────────┐
                  UI        Sentry     Claude judge
                          (when flagged)   (async, when flagged)
```

`POST /api/chat` returns `application/x-ndjson`: zero or more `{"type":"token",...}` lines, then exactly one `{"type":"event", ...CognitionEvent}`. The default model is `unsloth/gemma-3-4b-it` with Gemma Scope SAEs at layer 17, which is ungated and loads without a HuggingFace token.

## Quickstart

```bash
# 1. Install
uv sync                 # base install — runs in synthetic fallback mode, no GPU
uv sync --extra ml      # full install — torch + sae_lens, for real activations

# 2. Configure
cp config.example.yaml config.yaml   # edit as needed; config.yaml is gitignored
cp .env.example .env                  # ANTHROPIC_API_KEY enables Build + labels; SENTRY_DSN optional

# 3. Backend (port 8000)
uv run uvicorn backend.app:app --port 8000

# 4. Frontend (Vite on :5173, proxies /api -> :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. Check the backend with `curl -s localhost:8000/api/health`:

```json
{"mode":"fallback","model":"unsloth/gemma-3-4b-it","layer":17,"trackers":[]}
```

Without the `ml` extra (or with no GPU and no weights), the backend reports `"mode":"fallback"` and serves synthetic features, so the whole UI works on any machine. With the `ml` extra, weights present, and a running GPU pod, it reports `"mode":"real"` and the activations, live probes, and Build pipeline are real.

Secrets (`ANTHROPIC_API_KEY`, `SENTRY_DSN`, `POD_TOKEN`, `HF_TOKEN`) live only in `.env`, never in `config.yaml`. Sentry receives anomaly flags and scalar metrics only; raw prompts and responses stay local unless you set `observability.sentry.send_io: true`.

## Documentation

- [`docs/probe-training.md`](docs/probe-training.md) — training and calibrating probe vectors.
- [`docs/tailscale-pod-runbook.md`](docs/tailscale-pod-runbook.md) — running the GPU pod over Tailscale when campus or office WiFi blocks public SSH.

## License

[MIT](LICENSE).
