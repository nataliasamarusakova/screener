"""Factor attribution and robustness statistics for recorded research trades."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from engine.research import TradeRecord


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
