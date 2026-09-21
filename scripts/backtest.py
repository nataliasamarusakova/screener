#!/usr/bin/env python3
"""Run the point-in-time Quant Screener historical backtest.

Input dataset must contain the research schema documented in
`data/research/README.md`. The script emits a trade ledger and a compact summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from pathlib import Path

from engine.research import BacktestConfig, QuantBacktester, ResearchDataset, trade_to_dict
from engine.signals import QuantSignalEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL or CSV point-in-time research dataset")
    parser.add_argument("--output", default="data/research/backtest_trades.jsonl")
    parser.add_argument("--entry-mode", choices=["next_open", "close"], default="next_open")
    parser.add_argument("--max-holding-bars", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--output-summary", default="data/research/backtest_summary.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    dataset = ResearchDataset.from_jsonl(input_path) if input_path.suffix.lower() == ".jsonl" else ResearchDataset.from_csv(input_path)
    engine = QuantSignalEngine(strong_signal_threshold=args.threshold)
    backtester = QuantBacktester(engine, BacktestConfig(entry_mode=args.entry_mode, max_holding_bars=args.max_holding_bars))
    trades = backtester.run(dataset)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for trade in trades:
            fh.write(json.dumps(trade_to_dict(trade), separators=(",", ":")) + "\n")

    summary = {
        "symbols": len(dataset.symbols()),
        "bars": sum(len(dataset.bars(symbol)) for symbol in dataset.symbols()),
        "trades": len(trades),
        "strong_longs": sum(t.side == "LONG" for t in trades),
        "strong_shorts": sum(t.side == "SHORT" for t in trades),
        "mean_net_return": (sum(t.net_return for t in trades) / len(trades)) if trades else None,
        "total_compounded_return": None,
        "entry_mode": args.entry_mode,
        "max_holding_bars": args.max_holding_bars,
        "strong_threshold": args.threshold,
        "status": "NO_DATA" if not trades else "COMPLETED",
    }
    equity = 1.0
    for trade in trades:
        equity *= max(0.0, 1.0 + trade.net_return)
    if trades:
        summary["total_compounded_return"] = equity - 1.0
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
