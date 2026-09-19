"""
Dynamic Risk Management & Position Sizing Engine.
Models:
1. Invalidation-based Position Sizing (Fixed Fractional Risk / Volatility Sizing).
2. Liquidity Constraints & Maximum Market Impact Limits.
3. Dynamic Leverage calculation bounded by risk parameters.
"""
from __future__ import annotations

import math
from typing import Optional
import msgspec


class PositionSizeRecommendation(msgspec.Struct, gc=False):
    symbol: str
    side: str                            # "LONG" or "SHORT"
    entry_price: float
    invalidation_price: float
    target_price: float
    stop_distance_pct: float             # abs(entry - invalidation) / entry * 100
    target_distance_pct: float           # abs(target - entry) / entry * 100
    risk_reward_ratio: float
    account_capital_usdt: float
    risk_per_trade_usdt: float
    recommended_quantity: float
    recommended_notional_usdt: float
    effective_leverage: float
    max_allowed_leverage: float
    liquidity_constraint_applied: bool


class DynamicRiskEngine:
    """
    Computes mathematically rigorous position sizing based on structural invalidation levels.
    """

    def __init__(
        self,
        default_account_capital: float = 10000.0,
        risk_per_trade_pct: float = 1.0,         # 1% equity at risk per trade
        max_leverage: float = 10.0,              # Maximum margin leverage
        max_liquidity_impact_pct: float = 1.5,   # Maximum % of 5m taker volume
    ) -> None:
        self.default_account_capital = default_account_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_leverage = max_leverage
        self.max_liquidity_impact_pct = max_liquidity_impact_pct

    def calculate_sizing(
        self,
        symbol: str,
        signal_type: str,                        # "STRONG_LONG" or "STRONG_SHORT"
        entry_price: float,
        invalidation_price: float,
        target_price: float,
        available_liquidity_usdt: Optional[float] = None,
        account_capital: Optional[float] = None,
    ) -> Optional[PositionSizeRecommendation]:
        """
        Calculates exact position sizing based on risk-to-invalidation distance.
        """
        if entry_price <= 0.0 or invalidation_price <= 0.0:
            return None

        capital = account_capital or self.default_account_capital
        risk_budget = capital * (self.risk_per_trade_pct / 100.0)

        side = "LONG" if signal_type == "STRONG_LONG" else "SHORT"

        # Validate directional consistency
        if side == "LONG":
            if invalidation_price >= entry_price:
                # Invalidation must be below entry for longs
                invalidation_price = entry_price * 0.985
            if target_price <= entry_price:
                target_price = entry_price * 1.03
        else:
            if invalidation_price <= entry_price:
                # Invalidation must be above entry for shorts
                invalidation_price = entry_price * 1.015
            if target_price >= entry_price:
                target_price = entry_price * 0.97

        stop_distance_dollar = abs(entry_price - invalidation_price)
        target_distance_dollar = abs(target_price - entry_price)

        stop_pct = (stop_distance_dollar / entry_price) * 100.0
        target_pct = (target_distance_dollar / entry_price) * 100.0
        rrr = (target_distance_dollar / stop_distance_dollar) if stop_distance_dollar > 0 else 0.0

        # Unconstrained Quantity: Risk Budget / Stop Distance per contract
        unconstrained_qty = risk_budget / stop_distance_dollar if stop_distance_dollar > 0 else 0.0
        unconstrained_notional = unconstrained_qty * entry_price

        # Check Maximum Leverage Cap
        max_permitted_notional = capital * self.max_leverage
        notional = min(unconstrained_notional, max_permitted_notional)

        # Check Liquidity Impact Cap
        liquidity_applied = False
        if available_liquidity_usdt is not None and available_liquidity_usdt > 0.0:
            max_liquidity_notional = available_liquidity_usdt * (self.max_liquidity_impact_pct / 100.0)
            if notional > max_liquidity_notional:
                notional = max_liquidity_notional
                liquidity_applied = True

        final_qty = notional / entry_price
        effective_leverage = notional / capital if capital > 0 else 0.0

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
