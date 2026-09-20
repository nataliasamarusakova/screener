"""
Quantitative Signal Engine.
Calculates empirical factor Z-scores and generates Composite Quantitative Scores (-100 to +100).

Production invariants:
- No linear scaling is mislabeled as a Z-score.
- All factor Z-scores use a point-in-time history supplied by the caller.
- Missing/insufficient distribution history fails closed instead of inventing a neutral value.
- Risk levels are derived from 5m structure + ATR; invalid levels are rejected.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple

from contracts import SignalEvent


Z_CLIP = 3.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def empirical_zscore(value: float, history: Sequence[float], min_samples: int = 24) -> float:
    """
    Compute a point-in-time sample Z-score from *previous* observations only.

    `history` must not contain the current observation. The function deliberately
    fails closed when there is not enough variance/history to estimate a distribution.
    """
    if not math.isfinite(value):
        raise ValueError("Non-finite factor value")
    if min_samples < 2:
        raise ValueError("min_samples must be >= 2")

    history_values = [float(x) for x in history]
    if not all(math.isfinite(x) for x in history_values):
        raise ValueError("Factor history contains non-finite values")
    if len(history_values) < min_samples:
        raise ValueError(
            f"Insufficient history: {len(history_values)} < {min_samples}"
        )

    mean = sum(history_values) / len(history_values)
    sum_sq = sum((x - mean) ** 2 for x in history_values)
    variance = sum_sq / (len(history_values) - 1)
    std = math.sqrt(variance)
    if std <= 1e-12:
        raise ValueError("Factor history has zero variance")

    return _clip((value - mean) / std, -Z_CLIP, Z_CLIP)


class QuantSignalEngine:
    """Computes empirical factor Z-scores and Composite SignalEvents."""

    def __init__(
        self,
        w_cvd: float = 25.0,
        w_fund: float = 25.0,
        w_oi: float = 15.0,
        w_micro: float = 15.0,
        w_whale: float = 20.0,
        funding_weight: float = 0.70,
        basis_weight: float = 0.30,
        z_history_min_samples: int = 24,
        score_tanh_scale: float = 1.40,   # Explicit calibration parameter; TODO: re-calibrate OOS.
        sweep_booster_points: float = 25.0,  # Existing model constant; TODO: re-calibrate on labelled sweeps.
        strong_signal_threshold: float = 75.0,
        swing_buffer_bps: float = 5.0,    # Explicit config; TODO: calibrate against realised invalidation slippage.
        atr_stop_multiplier: float = 1.50, # Explicit config; TODO: calibrate from 5m MAE/MFE data.
        target_risk_reward: float = 2.0,
    ) -> None:
        numeric_config = (
            w_cvd, w_fund, w_oi, w_micro, w_whale,
            funding_weight, basis_weight, score_tanh_scale,
            sweep_booster_points, strong_signal_threshold,
            swing_buffer_bps, atr_stop_multiplier, target_risk_reward,
        )
        if not all(math.isfinite(float(x)) for x in numeric_config):
            raise ValueError("Signal configuration must be finite")
        if any(float(x) < 0.0 for x in (w_cvd, w_fund, w_oi, w_micro, w_whale)):
            raise ValueError("Signal weights must be non-negative")
        self.w_cvd = float(w_cvd)
        self.w_fund = float(w_fund)
        self.w_oi = float(w_oi)
        self.w_micro = float(w_micro)
        self.w_whale = float(w_whale)
        self.total_weights = self.w_cvd + self.w_fund + self.w_oi + self.w_micro + self.w_whale
        self.funding_weight = float(funding_weight)
        self.basis_weight = float(basis_weight)
        self.z_history_min_samples = z_history_min_samples
        self.score_tanh_scale = float(score_tanh_scale)
        self.sweep_booster_points = float(sweep_booster_points)
        self.strong_signal_threshold = float(strong_signal_threshold)
        self.swing_buffer_bps = float(swing_buffer_bps)
        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.target_risk_reward = float(target_risk_reward)

        if self.total_weights <= 0.0:
            raise ValueError("Signal weights must sum to a positive value")
        if abs((self.funding_weight + self.basis_weight) - 1.0) > 1e-9:
            raise ValueError("funding_weight + basis_weight must equal 1")
        if z_history_min_samples < 2:
            raise ValueError("z_history_min_samples must be >= 2")
        if self.score_tanh_scale <= 0.0 or self.target_risk_reward <= 0.0:
            raise ValueError("score_tanh_scale and target_risk_reward must be positive")
        if not (0.0 < self.strong_signal_threshold <= 100.0):
            raise ValueError("strong_signal_threshold must be in (0, 100]")
        if self.swing_buffer_bps < 0.0 or self.atr_stop_multiplier <= 0.0:
            raise ValueError("Risk-level configuration has invalid bounds")

    def calculate_factor_zscores(
        self,
        *,
        funding_rate_8h: float,
        basis_spread_bps: float,
        delta_oi_pct: float,
        obi: float,
        vpin: float,
        cvd_divergence_score: float,
        whale_divergence_score: float,
        funding_history: Sequence[float],
        basis_history: Sequence[float],
        delta_oi_pct_history: Sequence[float],
        micro_factor_history: Sequence[float],
        cvd_history: Sequence[float],
        whale_history: Sequence[float],
    ) -> Tuple[float, float, float, float, float]:
        """Build all factor Z-scores from prior point-in-time observations."""
        micro_factor = obi * (1.0 - vpin)
        if not math.isfinite(micro_factor):
            raise ValueError("Non-finite microstructure factor")

        z_cvd = empirical_zscore(cvd_divergence_score, cvd_history, self.z_history_min_samples)
        z_funding = empirical_zscore(funding_rate_8h, funding_history, self.z_history_min_samples)
        z_basis = empirical_zscore(basis_spread_bps, basis_history, self.z_history_min_samples)
        z_fund_trap = _clip(
            -(self.funding_weight * z_funding + self.basis_weight * z_basis),
            -Z_CLIP,
            Z_CLIP,
        )
        z_delta_oi = empirical_zscore(delta_oi_pct, delta_oi_pct_history, self.z_history_min_samples)
        z_micro = empirical_zscore(micro_factor, micro_factor_history, self.z_history_min_samples)
        z_whale = empirical_zscore(whale_divergence_score, whale_history, self.z_history_min_samples)
        return z_cvd, z_fund_trap, z_delta_oi, z_micro, z_whale

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
        gate_status: str = "PASSED",
        gate_long_status: Optional[str] = None,
        gate_short_status: Optional[str] = None,
        z_cvd_override: Optional[float] = None,
        z_fund_override: Optional[float] = None,
        z_delta_oi_override: Optional[float] = None,
        z_micro_override: Optional[float] = None,
        z_whale_override: Optional[float] = None,
        sweep_pattern: str = "NONE",
        atr_pct: Optional[float] = None,
    ) -> SignalEvent:
        """Generate a signal from already validated, point-in-time factor Z-scores."""
        now_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        if (
            z_cvd_override is None
            or z_fund_override is None
            or z_delta_oi_override is None
            or z_micro_override is None
            or z_whale_override is None
        ):
            raise ValueError("All empirical factor Z-score overrides are required")

        values = (
            current_price,
            funding_rate_8h,
            basis_spread_bps,
            delta_oi,
            oi_total,
            obi,
            vpin,
            cvd_divergence_score,
            recent_high,
            recent_low,
            relative_strength,
            z_whale_sentiment,
            z_cvd_override,
            z_fund_override,
            z_delta_oi_override,
            z_micro_override,
            z_whale_override,
        )
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Non-finite signal input")
        if current_price <= 0.0 or oi_total <= 0.0:
            raise ValueError("Signal input requires positive current_price and oi_total")
        if not (0.0 <= vpin <= 1.0):
            raise ValueError("VPIN must be in [0, 1]")

        z_cvd_div = _clip(float(z_cvd_override), -Z_CLIP, Z_CLIP)
        z_fund_trap = _clip(float(z_fund_override), -Z_CLIP, Z_CLIP)
        z_delta_oi = _clip(float(z_delta_oi_override), -Z_CLIP, Z_CLIP)
        z_micro = _clip(float(z_micro_override), -Z_CLIP, Z_CLIP)
        z_whale = _clip(float(z_whale_override), -Z_CLIP, Z_CLIP)

        raw_weighted = (
            self.w_cvd * z_cvd_div
            + self.w_fund * z_fund_trap
            + self.w_oi * z_delta_oi
            + self.w_micro * z_micro
            + self.w_whale * z_whale
        )

        if sweep_reclaim:
            if sweep_pattern == "BULLISH_SWEEP_RECLAIM":
                raw_weighted += self.sweep_booster_points
            elif sweep_pattern == "BEARISH_SWEEP_RECLAIM":
                raw_weighted -= self.sweep_booster_points
            else:
                raise ValueError("sweep_reclaim=True requires an explicit sweep_pattern")
        elif sweep_pattern != "NONE":
            raise ValueError("sweep_pattern must be NONE when sweep_reclaim=False")

        normalized_score = _clip(
            100.0 * math.tanh(raw_weighted / (self.total_weights * self.score_tanh_scale)),
            -100.0,
            100.0,
        )

        if normalized_score >= self.strong_signal_threshold:
            effective_gate = gate_long_status if gate_long_status is not None else gate_status
            if effective_gate == "PASSED":
                signal_type = "STRONG_LONG"
                final_gate = "PASSED"
            else:
                signal_type = "NEUTRAL"
                normalized_score = min(normalized_score, self.strong_signal_threshold - 30.0)
                final_gate = effective_gate
        elif normalized_score <= -self.strong_signal_threshold:
            effective_gate = gate_short_status if gate_short_status is not None else gate_status
            if effective_gate == "PASSED":
                signal_type = "STRONG_SHORT"
                final_gate = "PASSED"
            else:
                signal_type = "NEUTRAL"
                normalized_score = max(normalized_score, -self.strong_signal_threshold + 30.0)
                final_gate = effective_gate
        else:
            signal_type = "NEUTRAL"
            final_gate = gate_status
            if gate_long_status is not None and gate_short_status is not None:
                if gate_long_status != "PASSED":
                    final_gate = gate_long_status
                elif gate_short_status != "PASSED":
                    final_gate = gate_short_status
                else:
                    final_gate = "PASSED"

        if signal_type in ("STRONG_LONG", "STRONG_SHORT"):
            if atr_pct is None or not math.isfinite(atr_pct) or atr_pct <= 0.0:
                raise ValueError("ATR is required for strong-signal risk levels")
            if recent_low <= 0.0 or recent_high <= 0.0:
                raise ValueError("5m swing levels are required for strong-signal risk levels")

            buffer = self.swing_buffer_bps / 10000.0
            atr_distance = current_price * atr_pct * self.atr_stop_multiplier
            if signal_type == "STRONG_LONG":
                structural_stop = recent_low * (1.0 - buffer)
                atr_stop = current_price - atr_distance
                invalidation_price = min(structural_stop, atr_stop)
                if invalidation_price <= 0.0 or invalidation_price >= current_price:
                    raise ValueError("Invalid long invalidation level")
                risk = current_price - invalidation_price
                target_price = current_price + risk * self.target_risk_reward
                rrr = (target_price - current_price) / risk
            else:
                structural_stop = recent_high * (1.0 + buffer)
                atr_stop = current_price + atr_distance
                invalidation_price = max(structural_stop, atr_stop)
                if invalidation_price <= current_price:
                    raise ValueError("Invalid short invalidation level")
                risk = invalidation_price - current_price
                target_price = current_price - risk * self.target_risk_reward
                if target_price <= 0.0:
                    raise ValueError("Invalid short target level")
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
            sweep_pattern=sweep_pattern,
        )
