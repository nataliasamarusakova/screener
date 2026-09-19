"""
Liquidity Sweep & Reclaim Engine (Wyckoff Spring & Upthrust / Turtle Soup Pattern).
Detects stop hunts where price sweeps beyond key swing levels, triggers stop cascades,
and then immediately reclaims the level with passive/aggressive absorption.
"""
from __future__ import annotations

from typing import Optional, Tuple
import msgspec
import numpy as np
import numba


class SweepReclaimEvent(msgspec.Struct, gc=False):
    symbol: str
    pattern_type: str                   # "BULLISH_SWEEP_RECLAIM" (Spring) or "BEARISH_SWEEP_RECLAIM" (Upthrust)
    swept_level: float                  # The swing level that was probed
    reclaim_price: float                # Current price reclaiming the level
    penetration_pct: float              # How deep the sweep penetrated
    reclaim_strength_score: float       # [0.0, 1.0]
    is_confirmed: bool


@numba.njit(fastmath=True, nogil=True)
def detect_sweep_reclaim_jit(
    current_price: float,
    current_low: float,
    current_high: float,
    swing_low: float,
    swing_high: float,
    cvd_delta: float,
) -> Tuple[int, float, float]:
    """
    JIT-accelerated detection of stop hunt sweeps and reclaims.
    Returns:
        pattern_code: 1 for Bullish Spring, -1 for Bearish Upthrust, 0 for None
        swept_level: float
        penetration_pct: float
    """
    # 1. Bullish Sweep & Reclaim (Spring):
    # Low pierced below swing_low, but current price reclaimed back ABOVE swing_low
    if current_low < swing_low and current_price > swing_low:
        if cvd_delta >= 0.0:  # Absorption confirmation: buyers actively lifting offers
            penetration = ((swing_low - current_low) / swing_low) * 100.0
            # Sweep should be clean (0.05% to 2.5% sweep)
            if 0.05 <= penetration <= 2.5:
                return 1, swing_low, penetration

    # 2. Bearish Sweep & Reclaim (Upthrust):
    # High pierced above swing_high, but current price dropped back BELOW swing_high
    if current_high > swing_high and current_price < swing_high:
        if cvd_delta <= 0.0:  # Sellers actively hitting bids
            penetration = ((current_high - swing_high) / swing_high) * 100.0
            if 0.05 <= penetration <= 2.5:
                return -1, swing_high, penetration

    return 0, 0.0, 0.0


class LiquiditySweepDetector:
    """
    Detects institutional liquidity sweep and reclaim events.
    """

    def detect(
        self,
        symbol: str,
        current_price: float,
        current_high: float,
        current_low: float,
        recent_swing_high: float,
        recent_swing_low: float,
        cvd_delta: float,
    ) -> Optional[SweepReclaimEvent]:
        if recent_swing_low <= 0.0 or recent_swing_high <= 0.0 or current_price <= 0.0:
            return None

        code, level, pen_pct = detect_sweep_reclaim_jit(
            current_price=current_price,
            current_low=current_low,
            current_high=current_high,
            swing_low=recent_swing_low,
            swing_high=recent_swing_high,
            cvd_delta=cvd_delta,
        )

        if code == 1:
            return SweepReclaimEvent(
                symbol=symbol,
                pattern_type="BULLISH_SWEEP_RECLAIM",
                swept_level=round(level, 4),
                reclaim_price=round(current_price, 4),
                penetration_pct=round(pen_pct, 2),
                reclaim_strength_score=0.9,
                is_confirmed=True,
            )
        elif code == -1:
            return SweepReclaimEvent(
                symbol=symbol,
                pattern_type="BEARISH_SWEEP_RECLAIM",
                swept_level=round(level, 4),
                reclaim_price=round(current_price, 4),
                penetration_pct=round(pen_pct, 2),
                reclaim_strength_score=0.9,
                is_confirmed=True,
            )

        return None
