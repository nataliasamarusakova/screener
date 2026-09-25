"""Canonical serialization for SignalEvent across all persistence surfaces."""
from __future__ import annotations

from typing import Any

from contracts import SignalEvent

CANONICAL_SIGNAL_FIELDS = (
    "symbol", "timestamp_ms", "signal_type", "composite_score",
    "z_cvd_div", "z_fund_trap", "z_delta_oi", "z_micro",
    "vpin", "obi", "funding_8h", "basis_bps", "price",
    "invalidation_price", "target_price", "risk_reward_ratio",
    "decision_timestamp_ms", "z_whale_sentiment", "relative_strength",
    "sweep_reclaim", "gate_status", "sweep_pattern",
    "suggested_position_usd", "suggested_leverage",
    "trailing_stop_activation_pct", "trailing_stop_distance_pct",
    "applied_friction_rt_pct",
)


def signal_to_dict(signal: SignalEvent) -> dict[str, Any]:
    """Return the canonical 27-field SignalEvent contract with stable names."""
    return {
        "symbol": signal.symbol,
        "timestamp_ms": int(signal.timestamp_ms),
        "signal_type": signal.signal_type,
        "composite_score": float(signal.composite_score),
        "z_cvd_div": float(signal.z_cvd_div),
        "z_fund_trap": float(signal.z_fund_trap),
        "z_delta_oi": float(signal.z_delta_oi),
        "z_micro": float(signal.z_micro),
        "vpin": float(signal.vpin),
        "obi": float(signal.obi),
        "funding_8h": float(signal.funding_8h),
        "basis_bps": float(signal.basis_bps),
        "price": float(signal.price),
        "invalidation_price": float(signal.invalidation_price),
        "target_price": float(signal.target_price),
        "risk_reward_ratio": float(signal.risk_reward_ratio),
        "decision_timestamp_ms": int(signal.decision_timestamp_ms),
        "z_whale_sentiment": float(signal.z_whale_sentiment),
        "relative_strength": float(signal.relative_strength),
        "sweep_reclaim": bool(signal.sweep_reclaim),
        "gate_status": signal.gate_status,
        "sweep_pattern": signal.sweep_pattern,
        "suggested_position_usd": float(signal.suggested_position_usd),
        "suggested_leverage": int(signal.suggested_leverage),
        "trailing_stop_activation_pct": float(signal.trailing_stop_activation_pct),
        "trailing_stop_distance_pct": float(signal.trailing_stop_distance_pct),
        "applied_friction_rt_pct": float(signal.applied_friction_rt_pct),
    }
