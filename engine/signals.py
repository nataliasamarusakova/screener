"""
Quantitative Signal Engine.
Calculates factor Z-scores and generates Composite Quantitative Scores (-100 to +100).
Rules:
- STRONG LONG (Score > +75):
    1. Funding Trap: Negative funding (Z_Fund < -2.0), rising OI.
    2. CVD Absorption: Bullish divergence (Price Lower Low, CVD Higher High).
    3. Microstructure: OBI > 0, VPIN exhaustion.
- STRONG SHORT (Score < -75):
    1. Long Trap: Overheated funding (Z_Fund > +2.5), basis falling.
    2. CVD Absorption: Bearish divergence (Price Higher High, CVD Lower Low).
    3. Microstructure: OBI < 0, supply imbalance.
- NEUTRAL / EXIT (-50 <= Score <= +50).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from contracts import SignalEvent


class QuantSignalEngine:
    """
    Computes Composite Quantitative Derivative Scores and generates SignalEvents.
    """

    def __init__(
        self,
        w_cvd: float = 25.0,
        w_fund: float = 25.0,
        w_oi: float = 15.0,
        w_micro: float = 15.0,
        w_whale: float = 20.0,
    ) -> None:
        self.w_cvd = w_cvd
        self.w_fund = w_fund
        self.w_oi = w_oi
        self.w_micro = w_micro
        self.w_whale = w_whale
        self.total_weights = w_cvd + w_fund + w_oi + w_micro + w_whale

    def compute_signal(
        self,
        symbol: str,
        current_price: float,
        funding_rate_8h: float,
        basis_spread_bps: float,
        delta_oi: float,
        oi_total: float,
        obi: float,
        vpin: float,
        cvd_divergence_score: float,
        recent_high: float,
        recent_low: float,
        timestamp_ms: Optional[int] = None,
        z_whale_sentiment: float = 0.0,
        relative_strength: float = 0.0,
        sweep_reclaim: bool = False,
        gate_status: str = "PASSED",           # Legacy single-gate (used in tests/MCP)
        gate_long_status: Optional[str] = None, # Per-direction gate (preferred in screener)
        gate_short_status: Optional[str] = None,
    ) -> SignalEvent:
        """
        Calculates individual factor Z-scores, computes the composite score,
        and generates a calibrated SignalEvent with institutional market filters.

        Gate resolution priority:
          1. If gate_long_status / gate_short_status are provided (screener path),
             apply the direction-appropriate gate AFTER signal type is determined.
          2. If only gate_status is provided (legacy / test path), apply it uniformly.
        """
        now_ms = timestamp_ms or int(time.time() * 1000)

        # 1. Z-Score of CVD Divergence [-3.0, +3.0]
        z_cvd_div = float(np.clip(cvd_divergence_score * 3.0, -3.0, 3.0))

        # 2. Z-Score of Funding Trap [-3.0, +3.0]
        norm_funding_diff = -(funding_rate_8h - 0.0001) / 0.0003
        norm_basis_diff = -basis_spread_bps / 5.0
        z_fund_trap = float(np.clip(0.7 * norm_funding_diff + 0.3 * norm_basis_diff, -3.0, 3.0))

        # 3. Z-Score of Delta OI [-3.0, +3.0]
        if oi_total > 0.0:
            delta_oi_pct = delta_oi / oi_total
            if z_fund_trap > 1.0 and delta_oi > 0:
                z_delta_oi = float(np.clip(delta_oi_pct * 50.0, 0.0, 3.0))
            elif z_fund_trap < -1.0 and delta_oi > 0:
                z_delta_oi = float(np.clip(-delta_oi_pct * 50.0, -3.0, 0.0))
            else:
                z_delta_oi = float(np.clip(delta_oi_pct * 20.0, -3.0, 3.0))
        else:
            z_delta_oi = 0.0

        # 4. Z-Score of Microstructure [-3.0, +3.0]
        vpin_factor = max(0.0, 1.0 - vpin)
        z_micro = float(np.clip(obi * 2.5 * (0.8 + 0.2 * vpin_factor), -3.0, 3.0))

        # 5. Z-Score of Whale vs Retail Sentiment Divergence [-3.0, +3.0]
        z_whale = float(np.clip(z_whale_sentiment, -3.0, 3.0))

        # 6. Composite Score Calculation [-100.0, +100.0]
        raw_weighted = (
            self.w_cvd * z_cvd_div
            + self.w_fund * z_fund_trap
            + self.w_oi * z_delta_oi
            + self.w_micro * z_micro
            + self.w_whale * z_whale
        )

        # Wyckoff Spring / Sweep Reclaim Booster
        if sweep_reclaim:
            if raw_weighted >= 0.0:
                raw_weighted += 25.0  # High-conviction bullish spring reclaim
            else:
                raw_weighted -= 25.0  # High-conviction bearish upthrust reclaim

        # Map weighted score to [-100, 100] using tanh scaling
        normalized_score = float(np.clip(100.0 * np.tanh(raw_weighted / (self.total_weights * 1.4)), -100.0, 100.0))

        # 7. Determine Signal Classification, then apply direction-correct gate
        # Use per-direction gates if provided (screener path), else fall back to single gate_status.
        if normalized_score >= 75.0:
            effective_gate = gate_long_status if gate_long_status is not None else gate_status
            if effective_gate == "PASSED":
                signal_type = "STRONG_LONG"
                final_gate = "PASSED"
            else:
                signal_type = "NEUTRAL"   # Gated by BTC dump or pre-funding payout
                normalized_score = min(normalized_score, 45.0)  # Demote score
                final_gate = effective_gate
        elif normalized_score <= -75.0:
            effective_gate = gate_short_status if gate_short_status is not None else gate_status
            if effective_gate == "PASSED":
                signal_type = "STRONG_SHORT"
                final_gate = "PASSED"
            else:
                signal_type = "NEUTRAL"   # Gated by BTC pump or pre-funding squeeze
                normalized_score = max(normalized_score, -45.0)  # Demote score
                final_gate = effective_gate
        else:
            signal_type = "NEUTRAL"
            # For neutral signals: record if any gate is active (informational only)
            if gate_long_status is not None and gate_short_status is not None:
                if gate_long_status != "PASSED" and gate_short_status != "PASSED":
                    final_gate = gate_long_status  # Both gated, show long reason
                elif gate_long_status != "PASSED":
                    final_gate = gate_long_status
                elif gate_short_status != "PASSED":
                    final_gate = gate_short_status
                else:
                    final_gate = "PASSED"
            else:
                final_gate = gate_status

        # 8. Risk Invalidation & Target Levels
        if signal_type == "STRONG_LONG":
            invalidation_price = min(recent_low * 0.995, current_price * 0.985)
            risk = max(current_price - invalidation_price, current_price * 0.005)
            target_price = current_price + (risk * 2.0)
            rrr = (target_price - current_price) / risk
        elif signal_type == "STRONG_SHORT":
            invalidation_price = max(recent_high * 1.005, current_price * 1.015)
            risk = max(invalidation_price - current_price, current_price * 0.005)
            target_price = current_price - (risk * 2.0)
            rrr = (current_price - target_price) / risk
        else:
            invalidation_price = current_price
            target_price = current_price
            rrr = 0.0

        return SignalEvent(
            symbol=symbol,
            timestamp_ms=now_ms,
            signal_type=signal_type,
            composite_score=round(normalized_score, 2),
            z_cvd_div=round(z_cvd_div, 2),
            z_fund_trap=round(z_fund_trap, 2),
            z_delta_oi=round(z_delta_oi, 2),
            z_micro=round(z_micro, 2),
            vpin=round(vpin, 4),
            obi=round(obi, 4),
            funding_8h=funding_rate_8h,
            basis_bps=round(basis_spread_bps, 2),
            price=current_price,
            invalidation_price=round(invalidation_price, 4),
            target_price=round(target_price, 4),
            risk_reward_ratio=round(rrr, 2),
            decision_timestamp_ms=now_ms,
            z_whale_sentiment=round(z_whale, 2),
            relative_strength=round(relative_strength, 2),
            sweep_reclaim=sweep_reclaim,
            gate_status=final_gate,
        )
