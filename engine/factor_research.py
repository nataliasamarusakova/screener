"""Factor attribution and robustness statistics for recorded research trades."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from engine.research import ResearchDataset, TradeRecord, SUPPORTED_FORWARD_MINUTES


FACTORS = ("z_cvd", "z_fund", "z_oi", "z_micro", "z_whale")


@dataclass(frozen=True)
class FactorStats:
    factor: str
    n: int
    pearson_ic: float | None
    mean_signed_contribution: float | None
    hit_rate: float | None
    top_decile_return: float | None
    bottom_decile_return: float | None
    spread_top_minus_bottom: float | None


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        return None
    sx = float(xa.std(ddof=1))
    sy = float(ya.std(ddof=1))
    if sx <= 1e-15 or sy <= 1e-15:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def analyze_factors(trades: Sequence[TradeRecord]) -> list[FactorStats]:
    results: list[FactorStats] = []
    returns = [float(t.net_return) for t in trades if math.isfinite(float(t.net_return))]
    for factor in FACTORS:
        pairs = [
            (float(getattr(t, factor)), float(t.net_return))
            for t in trades
            if math.isfinite(float(getattr(t, factor))) and math.isfinite(float(t.net_return))
        ]
        if not pairs:
            results.append(FactorStats(factor, 0, None, None, None, None, None, None))
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        order = np.argsort(np.asarray(xs))
        decile = max(1, len(order) // 10)
        bottom = [ys[i] for i in order[:decile]]
        top = [ys[i] for i in order[-decile:]]
        directional = [x * y for x, y in pairs]
        results.append(
            FactorStats(
                factor=factor,
                n=len(pairs),
                pearson_ic=_pearson(xs, ys),
                mean_signed_contribution=_mean(directional),
                hit_rate=sum(v > 0.0 for v in ys) / len(ys),
                top_decile_return=_mean(top),
                bottom_decile_return=_mean(bottom),
                spread_top_minus_bottom=(_mean(top) - _mean(bottom)) if _mean(top) is not None and _mean(bottom) is not None else None,
            )
        )
    return results


def to_dicts(stats: Sequence[FactorStats]) -> list[dict]:
    return [asdict(item) for item in stats]


def analyze_feature_event_study(dataset: ResearchDataset) -> list[dict]:
    """Cross-sectional/event-study attribution over every signal-ready observation.

    This deliberately includes NEUTRAL observations. It uses only factor values
    recorded at the observation timestamp and future closes from the same symbol,
    so no future information enters the feature side of the analysis.
    """
    output: list[dict] = []
    for symbol in dataset.symbols():
        rows = dataset.bars(symbol)
        by_ts = {row.timestamp_ms: row for row in rows}
        for row in rows:
            recorded = row.candidate_signal or row.recorded_signal
            if not recorded or not bool(recorded):
                continue
            factors = {}
            for name in FACTORS:
                key = {
                    "z_cvd": "z_cvd_div",
                    "z_fund": "z_fund_trap",
                    "z_oi": "z_delta_oi",
                    "z_micro": "z_micro",
                    "z_whale": "z_whale_sentiment",
                }[name]
                value = recorded.get(key)
                if value is None or not math.isfinite(float(value)):
                    factors = {}
                    break
                factors[name] = float(value)
            if not factors:
                continue
            for minutes in SUPPORTED_FORWARD_MINUTES:
                future_ts = row.timestamp_ms + minutes * 60_000
                future = by_ts.get(future_ts)
                if future is None or row.close <= 0.0:
                    continue
                forward_return = future.close / row.close - 1.0
                record = {
                    "symbol": symbol,
                    "timestamp_ms": row.timestamp_ms,
                    "signal_type": recorded.get("signal_type", "NEUTRAL"),
                    "horizon_minutes": minutes,
                    "forward_return": float(forward_return),
                    **factors,
                }
                output.append(record)
    return output


def summarize_feature_event_study(rows: Sequence[dict]) -> list[dict]:
    """Aggregate event-study IC and mean signed return by factor/horizon."""
    results: list[dict] = []
    for factor in FACTORS:
        for horizon in SUPPORTED_FORWARD_MINUTES:
            pairs = [
                (float(row[factor]), float(row["forward_return"]))
                for row in rows
                if int(row.get("horizon_minutes", -1)) == horizon
                and math.isfinite(float(row.get(factor, float("nan"))))
                and math.isfinite(float(row.get("forward_return", float("nan"))))
            ]
            results.append({
                "factor": factor,
                "horizon_minutes": horizon,
                "n": len(pairs),
                "pearson_ic": _pearson([x for x, _ in pairs], [y for _, y in pairs]),
                "mean_signed_return": _mean([x * y for x, y in pairs]) if pairs else None,
            })
    return results
