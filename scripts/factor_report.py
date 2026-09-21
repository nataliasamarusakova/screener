#!/usr/bin/env python3
"""Report factor attribution from a completed backtest trade ledger."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from pathlib import Path

from engine.factor_research import analyze_factors, to_dicts
from engine.research import TradeRecord


def load(path: Path) -> list[TradeRecord]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(TradeRecord(**json.loads(line)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/research/backtest_trades.jsonl")
    parser.add_argument("--output", default="data/research/factor_report.json")
    args = parser.parse_args()
    trades = load(Path(args.input))
    report = {"trades": len(trades), "factors": to_dicts(analyze_factors(trades))}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
