"""
Synthetic Liquidation Reconstruction Engine.
Binance throttles @forceOrder WebSocket streams to at most 1 order/sec.
This module continuously cross-references delta Open Interest (ΔOI) against taker aggressive volume (CVD)
and price volatility shocks to reconstruct hidden liquidations.
"""
from __future__ import annotations

import math
import time
from typing import Optional
from contracts import SyntheticLiquidation


class SyntheticLiquidationDetector:
    """
    Reconstructs hidden liquidation cascades from ΔOI and taker volume anomalies.
    """

    def __init__(
        self,
        anomaly_ratio_threshold: float = 1.3,
        min_oi_drop_threshold: float = 10.0,
        volatility_shock_pct: float = 0.25,
    ) -> None:
        self.anomaly_ratio_threshold = anomaly_ratio_threshold
        self.min_oi_drop_threshold = min_oi_drop_threshold
        self.volatility_shock_pct = volatility_shock_pct

    def detect(
        self,
        symbol: str,
        current_price: float,
        price_change_pct: float,
        delta_oi: float,
        taker_buy_vol: float,
        taker_sell_vol: float,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[SyntheticLiquidation]:
        """
        Evaluate if a delta OI drop constitutes a synthetic liquidation.
        Returns SyntheticLiquidation event if anomaly detected, else None.
        """
        values = (current_price, price_change_pct, delta_oi, taker_buy_vol, taker_sell_vol)
        if not all(math.isfinite(float(x)) for x in values):
            return None
        if current_price <= 0.0 or taker_buy_vol < 0.0 or taker_sell_vol < 0.0:
            return None

        now_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Liquidation cascades MUST result in net open interest destruction (ΔOI < 0)
        if delta_oi >= 0.0 or abs(delta_oi) < self.min_oi_drop_threshold:
            return None

        abs_oi_drop = abs(delta_oi)
        total_taker_vol = taker_buy_vol + taker_sell_vol
        # Zero visible taker volume is an unknown denominator, not a tiny volume.
        if total_taker_vol <= 0.0:
            return None

        # Anomaly ratio: how severely the OI depletion outstrips visible taker volume.
        anomaly_ratio = abs_oi_drop / total_taker_vol

        # Case 1: Long Liquidation Cascade (Price dumped + massive OI reduction)
        if price_change_pct <= -self.volatility_shock_pct:
            # Drop in OI exceeds taker sell volume significantly
            if abs_oi_drop > taker_sell_vol * self.anomaly_ratio_threshold:
                # Estimate only the OI reduction not already represented by the
                # directionally relevant visible taker flow. This is an observable
                # residual, not an uncalibrated nonlinear volume heuristic.
                estimated_liq = abs_oi_drop - taker_sell_vol
                return SyntheticLiquidation(
                    symbol=symbol,
                    timestamp_ms=now_ms,
                    price=current_price,
                    delta_oi=delta_oi,
                    taker_volume=taker_sell_vol,
                    side="LONG_LIQUIDATION",
                    estimated_liquidation_volume=estimated_liq,
                    anomaly_ratio=anomaly_ratio,
                    is_synthetic=True,
                )

        # Case 2: Short Squeeze Liquidation Cascade (Price pumped + massive OI reduction)
        elif price_change_pct >= self.volatility_shock_pct:
            # Drop in OI exceeds taker buy volume significantly
            if abs_oi_drop > taker_buy_vol * self.anomaly_ratio_threshold:
                # Same residual accounting for a short-squeeze: OI reduction
                # minus visible aggressive buy flow.
                estimated_liq = abs_oi_drop - taker_buy_vol
                return SyntheticLiquidation(
                    symbol=symbol,
                    timestamp_ms=now_ms,
                    price=current_price,
                    delta_oi=delta_oi,
                    taker_volume=taker_buy_vol,
                    side="SHORT_LIQUIDATION",
                    estimated_liquidation_volume=estimated_liq,
                    anomaly_ratio=anomaly_ratio,
                    is_synthetic=True,
                )

        return None
