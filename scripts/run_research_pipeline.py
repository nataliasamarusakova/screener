#!/usr/bin/env python3
"""Run Quant Screener research stages in order with hard evidence gates."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from pathlib import Path



def load_trades(path: Path) -> list[TradeRecord]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(TradeRecord(**json.loads(line)))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/research/features.jsonl")
    parser.add_argument("--grid", default="data/research/wfa_grid.example.json")
    parser.add_argument("--out-dir", default="data/research/run")
    parser.add_argument("--min-outcomes", type=int, default=100)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        write_json(Path(args.out_dir) / "status.json", {"status": "DATA_REQUIRED", "dataset": str(dataset_path)})
        print("DATA_REQUIRED: collect a real point-in-time research dataset first")
        return

    from engine.factor_research import analyze_factors, to_dicts
    from engine.portfolio import PortfolioConfig, PortfolioSimulator, report_to_dict
    from engine.research import BacktestConfig, QuantBacktester, ResearchDataset, TradeRecord
    from engine.signals import QuantSignalEngine
    from engine.walk_forward import WalkForwardConfig, load_parameter_grid, run_wfa, save_results

    try:
        dataset = ResearchDataset.from_jsonl(dataset_path) if dataset_path.suffix.lower() == ".jsonl" else ResearchDataset.from_csv(dataset_path)
    except Exception as exc:
        write_json(Path(args.out_dir) / "status.json", {"status": "DATA_INVALID", "reason": str(exc)})
        raise SystemExit(2)

    out_dir = Path(args.out_dir)
    signal_engine = QuantSignalEngine()
    trades = QuantBacktester(signal_engine, BacktestConfig()).run(dataset)
    trade_path = out_dir / "backtest_trades.jsonl"
    trade_path.parent.mkdir(parents=True, exist_ok=True)
    trade_path.write_text("\n".join(json.dumps(t.__dict__, separators=(",", ":")) for t in trades) + ("\n" if trades else ""), encoding="utf-8")
    n = len(trades)
    if n < args.min_outcomes:
        write_json(out_dir / "status.json", {"status": "INSUFFICIENT_OUTCOMES", "trades": n, "minimum": args.min_outcomes})
        print(f"INSUFFICIENT_OUTCOMES: trades={n} minimum={args.min_outcomes}; factor/WFA promotion is blocked")
        return

    factor_report = {"trades": n, "factors": to_dicts(analyze_factors(trades))}
    write_json(out_dir / "factor_report.json", factor_report)

    configs = load_parameter_grid(Path(args.grid))
    wfa_results = run_wfa(dataset, configs, WalkForwardConfig())
    save_results(wfa_results, out_dir / "wfa_results.json")

    fills, portfolio_report = PortfolioSimulator(PortfolioConfig()).run(dataset, trades)
    write_json(out_dir / "portfolio_report.json", {"report": report_to_dict(portfolio_report), "fills": [f.__dict__ for f in fills]})
    write_json(out_dir / "status.json", {
        "status": "COMPLETED",
        "trades": n,
        "wfa_folds": len(wfa_results),
        "portfolio": report_to_dict(portfolio_report),
    })
    print(json.dumps({"status": "COMPLETED", "trades": n, "wfa_folds": len(wfa_results), "portfolio": report_to_dict(portfolio_report)}, indent=2))


if __name__ == "__main__":
    main()
