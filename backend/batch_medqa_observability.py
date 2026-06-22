"""Batch MedQA-style prompts through live inference + SAE feature labels, then fan out
to Sentry + Arize Phoenix.

Prereqs:
  - GPU pod reachable (POD_URL) OR local real mode
  - Optional: SENTRY_DSN in .env
  - Optional: Phoenix at PHOENIX_COLLECTOR_ENDPOINT (bash scripts/run_phoenix.sh)

  uv pip install arize-phoenix openinference-instrumentation python-dotenv  # if missing

Run:
  export POD_URL=http://localhost:8001 POD_TOKEN=glassbox-dev-secret
  python -m backend.batch_medqa_observability --limit 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from . import config, runtime
from .analyze import analyze_turn
from .fanout import fanout, init_sponsors

_cfg = config.load_config()  # Transitional cfg — Task 11 threads cfg through call chain.

_FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "medqa_prompts.json"


def _load_fixture_prompts(limit: int) -> list[dict]:
    rows = json.loads(_FIXTURE.read_text())
    return rows[:limit]


def _load_dataset_prompts(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    out: list[dict] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        q = row.get("question") or row.get("Question") or ""
        if not q.strip():
            continue
        opts = []
        for key in ("opa", "opb", "opc", "opd", "A", "B", "C", "D"):
            if row.get(key):
                opts.append(str(row[key]))
        prompt = q.strip()
        if opts:
            prompt += "\n\nOptions:\n" + "\n".join(f"- {o}" for o in opts[:4])
        out.append({"id": f"medmcqa-{row.get('id', i)}", "question": prompt})
    return out


def _ensure_real_mode() -> None:
    runtime.refresh_pod_health(_cfg)
    if runtime.STATE["mode"] != "real":
        print(
            f"[batch] backend mode={runtime.STATE['mode']} — need real (check POD_URL / pod health)",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Batch MedQA prompts → cognition events → observability")
    p.add_argument("--limit", type=int, default=10, help="number of prompts to run")
    p.add_argument(
        "--source",
        choices=("fixture", "medmcqa"),
        default="fixture",
        help="prompt source (fixture=local JSON, medmcqa=HuggingFace dataset)",
    )
    p.add_argument("--dry-run", action="store_true", help="skip fanout to Sentry/Phoenix")
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="abort on pod failure or synthetic fallback (default: on)",
    )
    args = p.parse_args()

    prompts = (
        _load_dataset_prompts(args.limit)
        if args.source == "medmcqa"
        else _load_fixture_prompts(args.limit)
    )
    if not prompts:
        print("[batch] no prompts loaded", file=sys.stderr)
        sys.exit(1)

    _ensure_real_mode()
    if not args.dry_run:
        init_sponsors(_cfg.observability, _cfg.sentry_dsn)
        print(
            f"[batch] sponsors: sentry={'on' if _cfg.sentry_dsn else 'off'} "
        )

    print(f"[batch] running {len(prompts)} prompts (mode={runtime.STATE['mode']})")
    results: list[dict] = []

    for i, row in enumerate(prompts, 1):
        qid = row["id"]
        question = row["question"]
        print(f"\n[{i}/{len(prompts)}] {qid}")
        print(f"  Q: {question[:100]}{'...' if len(question) > 100 else ''}")

        if args.strict:
            runtime.refresh_pod_health(_cfg)
            if runtime.STATE["mode"] != "real" or not runtime.STATE.get("pod_reachable", True):
                print(
                    f"[batch] pod not ready (mode={runtime.STATE['mode']}, "
                    f"reachable={runtime.STATE.get('pod_reachable')})",
                    file=sys.stderr,
                )
                sys.exit(1)

        t0 = time.time()
        answer, event, _perf = analyze_turn(
            [{"role": "user", "content": question}],
            message_id=qid,
            strict=args.strict,
        )
        elapsed = time.time() - t0
        payload = event.model_dump()
        top_labels = [f["label"] for f in payload["features"][:5]]

        print(f"  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"  features ({len(payload['features'])}): {top_labels}")
        print(f"  elapsed: {elapsed:.1f}s")

        if not args.dry_run:
            try:
                fanout(payload, obs=_cfg.observability, probes=_cfg.probes)
            except Exception as e:  # noqa: BLE001
                print(f"  fanout error: {e}")

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "top_features": top_labels,
                "elapsed_s": round(elapsed, 1),
            }
        )

    out_path = pathlib.Path("batch_medqa_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[batch] wrote {out_path}")

    if not args.dry_run:
        time.sleep(3)  # flush Phoenix OTLP batch exporter
        print("[batch] done — check Phoenix at http://localhost:6006 and Sentry Issues")


if __name__ == "__main__":
    main()
