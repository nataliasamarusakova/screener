#!/usr/bin/env python3
"""Apply portfolio risk controls to backtest trade candidates."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from dataclasses import asdict
from pathlib import Path

from engine.portfolio import PortfolioConfig, PortfolioSimulator, report_to_dict
from engine.research import ResearchDataset, TradeRecord


def load_trades(path: Path) -> list[TradeRecord]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(TradeRecord(**json.loads(line)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trades", default="data/research/backtest_trades.jsonl")
    parser.add_argument("--output", default="data/research/portfolio_report.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = ResearchDataset.from_jsonl(dataset_path) if dataset_path.suffix.lower() == ".jsonl" else ResearchDataset.from_csv(dataset_path)
    trades = load_trades(Path(args.trades))
    fills, report = PortfolioSimulator(PortfolioConfig()).run(dataset, trades)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"report": report_to_dict(report), "fills": [asdict(fill) for fill in fills]}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(report_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
