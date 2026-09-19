"""
CVD vs Price Divergence and Limit Order Absorption Engine.
Detects:
1. Bullish Absorption Divergence: Price Lower Low + CVD Higher High/Low (passive limit buyers absorbing aggressive sellers).
2. Bearish Absorption Divergence: Price Higher High + CVD Lower Low/High (iceberg limit sellers absorbing aggressive buyers).
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import numba


@numba.njit(fastmath=True, nogil=True)
def detect_cvd_divergence_jit(
    prices: np.ndarray,
    cvd: np.ndarray,
    lookback: int = 12
) -> Tuple[float, int]:
    """
    Scans price and cumulative volume delta (CVD) for absorption divergences.
    Returns:
        divergence_score: float in range [-1.0, 1.0] (positive = bullish, negative = bearish)
        divergence_type: int (1 = Bullish Absorption, -1 = Bearish Absorption, 0 = None)
    """
    n = len(prices)
    if n < lookback or n < 4:
        return 0.0, 0

    curr_p = prices[-1]
    curr_cvd = cvd[-1]

    # Find recent swing high and swing low in the lookback window (excluding current bar)
    window_prices = prices[-lookback:-1]
    window_cvd = cvd[-lookback:-1]

    min_p_idx = 0
    max_p_idx = 0
    min_p = window_prices[0]
    max_p = window_prices[0]

    for i in range(1, len(window_prices)):
        p = window_prices[i]
        if p < min_p:
            min_p = p
            min_p_idx = i
        if p > max_p:
            max_p = p
            max_p_idx = i

    cvd_at_min_p = window_cvd[min_p_idx]
    cvd_at_max_p = window_cvd[max_p_idx]

    price_range = max_p - min_p
    if price_range <= 1e-12:
        return 0.0, 0

    cvd_std = np.std(window_cvd)
    if cvd_std <= 1e-12:
        cvd_std = 1.0

    # 1. Bullish Divergence: Price makes Lower Low, CVD makes Higher High/Low
    if curr_p < min_p:
        if curr_cvd > cvd_at_min_p:
            # Aggressive selling absorbed by limit buy orders
            cvd_diff_norm = (curr_cvd - cvd_at_min_p) / cvd_std
            price_drop_norm = (min_p - curr_p) / price_range
            score = min(1.0, max(0.1, 0.5 * cvd_diff_norm + 0.5 * price_drop_norm))
            return score, 1

    # 2. Bearish Divergence: Price makes Higher High, CVD makes Lower Low/High
    elif curr_p > max_p:
        if curr_cvd < cvd_at_max_p:
            # Aggressive buying absorbed by iceberg limit sell orders
            cvd_diff_norm = (cvd_at_max_p - curr_cvd) / cvd_std
            price_rally_norm = (curr_p - max_p) / price_range
            score = -min(1.0, max(0.1, 0.5 * cvd_diff_norm + 0.5 * price_rally_norm))
            return score, -1

    return 0.0, 0
