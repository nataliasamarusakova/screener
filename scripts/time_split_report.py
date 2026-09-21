#!/usr/bin/env python3
"""Time-blocked OOS-style report for the recorded signal outcomes."""
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
from statistics import mean

HORIZONS = (5, 15, 30, 60)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r.get("timestamp_ms", 0)))


def hac_t_stat(values: list[float], max_lag: int) -> float | None:
    n = len(values)
    if n < 3:
        return None
    mu = mean(values)
    centered = [x - mu for x in values]
    gamma0 = sum(x * x for x in centered) / n
    if gamma0 <= 0.0:
        return None
    variance = gamma0
    max_lag = min(max_lag, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = sum(centered[t] * centered[t - lag] for t in range(lag, n)) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        variance += 2.0 * weight * gamma
    se = math.sqrt(max(variance, 0.0) / n)
    return mu / se if se > 0.0 else None


def stats(values: list[float], max_lag: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "t_stat_nw": None, "hit_rate": None, "profit_factor": None}
    mu = mean(values)
    t = hac_t_stat(values, max_lag=max_lag)
    gains = sum(x for x in values if x > 0.0)
    losses = -sum(x for x in values if x < 0.0)
    pf = gains / losses if losses > 0.0 else None
    return {
        "n": len(values),
        "mean": mu,
        "t_stat_nw": t,
        "hit_rate": sum(x > 0.0 for x in values) / len(values),
        "profit_factor": pf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/signal_outcomes.jsonl")
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()
    rows = load_rows(Path(args.input))
    print(json.dumps({"signals": len(rows)}, indent=2))
    if not rows:
        return

    size = max(1, math.ceil(len(rows) / max(1, args.blocks)))
    for horizon in HORIZONS:
        key = f"return_{horizon}m"
        all_values = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        max_lag = max(1, horizon // 5 - 1)
        report = {"horizon_min": horizon, "newey_west_lag": max_lag, "overall": stats(all_values, max_lag), "blocks": []}
        for i in range(0, len(rows), size):
            block = rows[i : i + size]
            vals = [float(r[key]) for r in block if r.get(key) is not None and math.isfinite(float(r[key]))]
            report["blocks"].append(stats(vals, max_lag))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
