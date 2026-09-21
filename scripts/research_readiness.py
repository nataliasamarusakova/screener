#!/usr/bin/env python3
"""Report whether the collected research data is sufficient for each stage."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import math
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/research/features.jsonl")
    parser.add_argument("--wfa-train-days", type=int, default=30)
    parser.add_argument("--wfa-validation-days", type=int, default=7)
    parser.add_argument("--wfa-test-days", type=int, default=7)
    parser.add_argument("--min-outcomes", type=int, default=100)
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        print(json.dumps({"status": "DATA_REQUIRED", "path": str(path)}, indent=2))
        return
    from engine.research import ResearchDataset
    dataset = ResearchDataset.from_jsonl(path) if path.suffix.lower() == ".jsonl" else ResearchDataset.from_csv(path)
    timestamps = sorted({bar.timestamp_ms for symbol in dataset.symbols() for bar in dataset.bars(symbol)})
    bars_needed = (args.wfa_train_days + args.wfa_validation_days + args.wfa_test_days) * 24 * 12 + 24
    calendar_days = ((timestamps[-1] - timestamps[0]) / 86_400_000) if timestamps else 0.0
    print(json.dumps({
        "status": "READY" if timestamps else "DATA_REQUIRED",
        "symbols": len(dataset.symbols()),
        "bars": sum(len(dataset.bars(s)) for s in dataset.symbols()),
        "calendar_days": calendar_days,
        "required_calendar_days_minimum": args.wfa_train_days + args.wfa_validation_days + args.wfa_test_days,
        "required_bar_count_per_contiguous_symbol_minimum": bars_needed,
        "minimum_outcomes_for_statistical_gate": args.min_outcomes,
        "note": "Outcome count is measured after backtest execution; bars alone do not establish statistical power."
    }, indent=2))


if __name__ == "__main__":
    main()
