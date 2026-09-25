#!/usr/bin/env python3
"""Export the latest canonical SignalEvent records without recomputing any factors."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.serialization import CANONICAL_SIGNAL_FIELDS


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/paper_signals_latest.json")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"ERROR: signal file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    signals = payload.get("signals")
    if not isinstance(signals, list):
        raise SystemExit("ERROR: canonical payload has no signals array")

    missing = []
    for idx, signal in enumerate(signals):
        absent = [field for field in CANONICAL_SIGNAL_FIELDS if field not in signal]
        if absent:
            missing.append((idx, absent))
    if missing:
        raise SystemExit(f"ERROR: non-canonical signal schema: {missing[:3]}")

    out = sys.stdout if args.output == "-" else Path(args.output).open("w", encoding="utf-8", newline="")
    close = out is not sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=list(CANONICAL_SIGNAL_FIELDS))
        writer.writeheader()
        for signal in signals:
            writer.writerow({field: signal[field] for field in CANONICAL_SIGNAL_FIELDS})
    finally:
        if close:
            out.close()


if __name__ == "__main__":
    main()
