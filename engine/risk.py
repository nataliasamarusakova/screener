"""
Dynamic Risk Management & Position Sizing Engine.

Production invariants:
- Invalid directional stop/target inputs are rejected, never silently rewritten.
- Explicit zero/NaN/Inf values are never substituted with configuration defaults.
- Liquidity caps are applied only when real liquidity input is supplied.
"""
from __future__ import annotations

import math
from typing import Optional

import msgspec


class PositionSizeRecommendation(msgspec.Struct, gc=False):
    symbol: str
    side: str
    entry_price: float
    invalidation_price: float
    target_price: float
    stop_distance_pct: float
    target_distance_pct: float
    risk_reward_ratio: float
    account_capital_usdt: float
    risk_per_trade_usdt: float
    recommended_quantity: float
    recommended_notional_usdt: float
    effective_leverage: float
    max_allowed_leverage: float
    liquidity_constraint_applied: bool


class DynamicRiskEngine:
    """Computes position sizing from validated structural invalidation levels."""

    def __init__(
        self,
        default_account_capital: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        max_leverage: float = 10.0,
        max_liquidity_impact_pct: float = 1.5,
    ) -> None:
        self.default_account_capital = default_account_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.max_liquidity_impact_pct = max_liquidity_impact_pct

    def calculate_sizing(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        invalidation_price: float,
        target_price: float,
        available_liquidity_usdt: Optional[float] = None,
        account_capital: Optional[float] = None,
    ) -> Optional[PositionSizeRecommendation]:
        """Calculate position size; return None for any invalid risk contract."""
        if signal_type not in ("STRONG_LONG", "STRONG_SHORT"):
            return None

        values = (
            entry_price,
            invalidation_price,
            target_price,
            self.risk_per_trade_pct,
            self.max_leverage,
            self.max_liquidity_impact_pct,
        )
        if not all(math.isfinite(float(x)) for x in values):
            return None
        if entry_price <= 0.0 or invalidation_price <= 0.0 or target_price <= 0.0:
            return None

        capital = self.default_account_capital if account_capital is None else float(account_capital)
        if not math.isfinite(capital) or capital <= 0.0:
            return None
        if not (0.0 < self.risk_per_trade_pct <= 100.0):
            return None
        if self.max_leverage <= 0.0:
            return None
        if self.max_liquidity_impact_pct <= 0.0:
            return None

        if signal_type == "STRONG_LONG":
            side = "LONG"
            if invalidation_price >= entry_price or target_price <= entry_price:
                return None
        else:
            side = "SHORT"
            if invalidation_price <= entry_price or target_price >= entry_price:
                return None

        if available_liquidity_usdt is not None:
            if not math.isfinite(float(available_liquidity_usdt)) or available_liquidity_usdt <= 0.0:
                return None

        risk_budget = capital * (self.risk_per_trade_pct / 100.0)
        stop_distance_dollar = abs(entry_price - invalidation_price)
        target_distance_dollar = abs(target_price - entry_price)
        if stop_distance_dollar <= 0.0 or target_distance_dollar <= 0.0:
            return None

        stop_pct = (stop_distance_dollar / entry_price) * 100.0
        target_pct = (target_distance_dollar / entry_price) * 100.0
        rrr = target_distance_dollar / stop_distance_dollar

        unconstrained_qty = risk_budget / stop_distance_dollar
        unconstrained_notional = unconstrained_qty * entry_price
        max_permitted_notional = capital * self.max_leverage
        notional = min(unconstrained_notional, max_permitted_notional)

        liquidity_applied = False
        if available_liquidity_usdt is not None:
            max_liquidity_notional = available_liquidity_usdt * (self.max_liquidity_impact_pct / 100.0)
            if not math.isfinite(max_liquidity_notional) or max_liquidity_notional <= 0.0:
                return None
            if notional > max_liquidity_notional:
                notional = max_liquidity_notional
                liquidity_applied = True

        final_qty = notional / entry_price
        effective_leverage = notional / capital

        return PositionSizeRecommendation(
            symbol=symbol,
            side=side,
            entry_price=round(entry_price, 4),
            invalidation_price=round(invalidation_price, 4),
            target_price=round(target_price, 4),
            stop_distance_pct=round(stop_pct, 2),
            target_distance_pct=round(target_pct, 2),
            risk_reward_ratio=round(rrr, 2),
            account_capital_usdt=round(capital, 2),
            risk_per_trade_usdt=round(risk_budget, 2),
            recommended_quantity=round(final_qty, 6),
            recommended_notional_usdt=round(notional, 2),
            effective_leverage=round(effective_leverage, 2),
            max_allowed_leverage=self.max_leverage,
            liquidity_constraint_applied=liquidity_applied,
        )
