"""
Anti-Spoofing, Liquidity & Quality Gate.
Filters out:
1. Illiquid tokens (< $15M 24h volume) prone to fake orderbook spoofing.
2. Wide-spread pairs (> 3.5 bps spread) that destroy edge via slippage.
3. Distressed tokens with extreme funding rate caps (e.g. -2.0% delisting anomalies).
"""
from __future__ import annotations

import math
import msgspec


class QualityGateResult(msgspec.Struct, gc=False):
    symbol: str
    is_valid: bool
    quote_volume_24h: float
    spread_bps: float
    funding_rate_8h: float
    rejection_reason: str


class QualityFilter:
    def __init__(
        self,
        min_24h_volume_usdt: float = 15_000_000.0,  # $15M minimum 24h quote volume
        max_spread_bps: float = 3.5,                 # 3.5 bps maximum spread
        max_abs_funding_rate_8h: float = 0.015,      # 1.5% maximum absolute 8h funding
    ) -> None:
        config_values = (min_24h_volume_usdt, max_spread_bps, max_abs_funding_rate_8h)
        if not all(math.isfinite(float(x)) for x in config_values):
            raise ValueError("Quality filter configuration must be finite")
        if min_24h_volume_usdt < 0.0 or max_spread_bps < 0.0 or max_abs_funding_rate_8h <= 0.0:
            raise ValueError("Quality filter configuration has invalid bounds")

        self.min_24h_volume_usdt = float(min_24h_volume_usdt)
        self.max_spread_bps = float(max_spread_bps)
        self.max_abs_funding_rate_8h = float(max_abs_funding_rate_8h)

    def evaluate(
        self,
        symbol: str,
        quote_volume_24h: float,
        spread_bps: float,
        funding_rate_8h: float,
    ) -> QualityGateResult:
        """
        Validates whether a contract meets institutional liquidity and quality standards.
        """
        if not all(math.isfinite(float(x)) for x in (quote_volume_24h, spread_bps, funding_rate_8h)):
            return QualityGateResult(
                symbol=symbol,
                is_valid=False,
                quote_volume_24h=quote_volume_24h,
                spread_bps=spread_bps,
                funding_rate_8h=funding_rate_8h,
                rejection_reason="NON_FINITE_INPUT",
            )
        if quote_volume_24h < 0.0 or spread_bps < 0.0:
            return QualityGateResult(
                symbol=symbol,
                is_valid=False,
                quote_volume_24h=quote_volume_24h,
                spread_bps=spread_bps,
                funding_rate_8h=funding_rate_8h,
                rejection_reason="INVALID_MARKET_DATA",
            )

        # 1. Volume Gate
        if quote_volume_24h < self.min_24h_volume_usdt:
            return QualityGateResult(
                symbol=symbol,
                is_valid=False,
                quote_volume_24h=quote_volume_24h,
                spread_bps=spread_bps,
                funding_rate_8h=funding_rate_8h,
                rejection_reason=f"LOW_VOLUME: ${quote_volume_24h:,.0f} < ${self.min_24h_volume_usdt:,.0f}",
            )

        # 2. Spread Gate
        if spread_bps > self.max_spread_bps:
            return QualityGateResult(
                symbol=symbol,
                is_valid=False,
                quote_volume_24h=quote_volume_24h,
                spread_bps=spread_bps,
                funding_rate_8h=funding_rate_8h,
                rejection_reason=f"WIDE_SPREAD: {spread_bps:.2f} bps > {self.max_spread_bps:.2f} bps",
            )

        # 3. Extreme Funding Anomaly Gate (Delisting / Cap Trap)
        if abs(funding_rate_8h) >= self.max_abs_funding_rate_8h:
            return QualityGateResult(
                symbol=symbol,
                is_valid=False,
                quote_volume_24h=quote_volume_24h,
                spread_bps=spread_bps,
                funding_rate_8h=funding_rate_8h,
                rejection_reason=f"FUNDING_ANOMALY: {funding_rate_8h*100:+.2f}% exceeds safe limit",
            )

        return QualityGateResult(
            symbol=symbol,
            is_valid=True,
            quote_volume_24h=quote_volume_24h,
            spread_bps=spread_bps,
            funding_rate_8h=funding_rate_8h,
            rejection_reason="PASSED",
        )
