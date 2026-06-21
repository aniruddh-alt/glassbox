"""See Sentry + Phoenix light up WITHOUT the GPU/model.

Fires the sample cognition_event (one flagged, one clean) through fanout().

  1. (optional) put SENTRY_DSN in .env
  2. (optional) start Phoenix:   bash scripts/run_phoenix.sh   # -> http://localhost:6006
  3. run:                        python -m backend.smoke_fanout   # from repo root

Then check: Sentry Issues for "Confident-wrong medical answer", and Phoenix at :6006
for a 'chat-turn' LLM span with cognition.* attributes.
"""

from __future__ import annotations

import json
import pathlib
import time

try:
    from dotenv import load_dotenv  # pip install python-dotenv (optional convenience)

    load_dotenv()
except Exception:
    pass  # rely on exported env vars instead

from . import fanout

_FIX = json.loads(
    (
        pathlib.Path(__file__).parents[1] / "fixtures" / "cognition_event.sample.json"
    ).read_text()
)


def main() -> None:
    fanout.init_sponsors()

    flagged = _FIX  # the sample is already a confident-wrong event
    clean = {
        **_FIX,
        "message_id": "clean-0001",
        "flag": False,
        "severity": "info",
        "uncertainty": 0.11,
        "io": {"user_msg": "What is the capital of France?", "response": "Paris."},
    }

    for ev in (flagged, clean):
        fanout.fanout(ev)
        print(f"emitted: flag={ev['flag']} uncertainty={ev['uncertainty']}")

    time.sleep(3)  # let Phoenix's batched OTLP exporter flush before the process exits
    print("done — check Sentry Issues + Phoenix at http://localhost:6006")


if __name__ == "__main__":
    main()
