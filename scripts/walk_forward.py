#!/usr/bin/env python3
"""Run explicit walk-forward validation from a frozen parameter grid."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

from engine.research import ResearchDataset
from engine.walk_forward import WalkForwardConfig, load_parameter_grid, run_wfa, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", default="data/research/wfa_results.json")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--embargo-bars", type=int, default=12)
    parser.add_argument("--min-train-trades", type=int, default=50)
    parser.add_argument("--min-test-trades", type=int, default=20)
    args = parser.parse_args()

    dataset = ResearchDataset.from_jsonl(Path(args.input)) if str(args.input).lower().endswith(".jsonl") else ResearchDataset.from_csv(Path(args.input))
    configs = load_parameter_grid(Path(args.grid))
    cfg = WalkForwardConfig(
        train_bars=args.train_days * 24 * 12,
        validation_bars=args.validation_days * 24 * 12,
        test_bars=args.test_days * 24 * 12,
        embargo_bars=args.embargo_bars,
        min_train_trades=args.min_train_trades,
        min_test_trades=args.min_test_trades,
    )
    results = run_wfa(dataset, configs, cfg)
    save_results(results, Path(args.output))
    print(f"folds={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
