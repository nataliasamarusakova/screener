"""
BTC Market Regime & Relative Strength (Beta) Filter.
Prevents buying altcoin longs during market-wide BTC liquidation dumps,
and prevents shorting during BTC impulse rallies.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import msgspec


class BTCRegime(msgspec.Struct, gc=False):
    status: str                         # "IMPULSE_DUMP", "IMPULSE_PUMP", "NEUTRAL_RANGING"
    btc_price: float
    btc_change_5m_pct: float
    btc_change_24h_pct: float
    allow_alt_longs: bool
    allow_alt_shorts: bool


class MarketRegimeEngine:
    """
    Evaluates Bitcoin's macro price action and gates altcoin trading.
    """

    def __init__(
        self,
        dump_threshold_5m_pct: float = -0.30,   # BTC dropping >= 0.3% in 5m triggers DUMP gate
        pump_threshold_5m_pct: float = +0.30,   # BTC rising >= 0.3% in 5m triggers PUMP gate
        beta_alt_btc: float = 1.6,             # Typical altcoin beta to BTC
        min_decoupled_rs_pct: float = 1.2,     # Minimum Relative Strength to bypass dump gate
    ) -> None:
        self.dump_threshold_5m_pct = dump_threshold_5m_pct
        self.pump_threshold_5m_pct = pump_threshold_5m_pct
        self.beta_alt_btc = beta_alt_btc
        self.min_decoupled_rs_pct = min_decoupled_rs_pct

    def evaluate_btc_regime(
        self,
        btc_last_price: float,
        btc_prev_price: float,
        btc_change_24h_pct: float,
    ) -> BTCRegime:
        """
        Determines the current market regime based on BTC short-term momentum.
        """
        if btc_prev_price <= 0.0 or btc_last_price <= 0.0:
            return BTCRegime(
                status="NEUTRAL_RANGING",
                btc_price=btc_last_price,
                btc_change_5m_pct=0.0,
                btc_change_24h_pct=btc_change_24h_pct,
                allow_alt_longs=True,
                allow_alt_shorts=True,
            )

        change_5m_pct = ((btc_last_price - btc_prev_price) / btc_prev_price) * 100.0

        if change_5m_pct <= self.dump_threshold_5m_pct:
            return BTCRegime(
                status="IMPULSE_DUMP",
                btc_price=btc_last_price,
                btc_change_5m_pct=round(change_5m_pct, 2),
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=False,  # Block alt longs during BTC dumps
                allow_alt_shorts=True,
            )
        elif change_5m_pct >= self.pump_threshold_5m_pct:
            return BTCRegime(
                status="IMPULSE_PUMP",
                btc_price=btc_last_price,
                btc_change_5m_pct=round(change_5m_pct, 2),
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=True,
                allow_alt_shorts=False,  # Block alt shorts during BTC pumps
            )
        else:
            return BTCRegime(
                status="NEUTRAL_RANGING",
                btc_price=btc_last_price,
                btc_change_5m_pct=round(change_5m_pct, 2),
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=True,
                allow_alt_shorts=True,
            )

    def calculate_relative_strength(
        self,
        alt_change_5m_pct: float,
        btc_change_5m_pct: float,
    ) -> float:
        """
        Calculates Beta-adjusted Relative Strength (RS):
        RS = Delta P_alt - beta * Delta P_btc.
        Positive RS indicates the altcoin is outperforming BTC.
        """
        return alt_change_5m_pct - (self.beta_alt_btc * btc_change_5m_pct)

    def check_signal_gate(
        self,
        symbol: str,
        signal_type: str,
        btc_regime: BTCRegime,
        alt_change_5m_pct: float,
    ) -> Tuple[bool, str]:
        """
        Gates signals based on BTC correlation.
        Returns: (is_allowed, reason)
        """
        if symbol.upper() in ("BTCUSDT", "BTCUSDT_240628"):
            return True, "BTC_SELF"

        rs = self.calculate_relative_strength(alt_change_5m_pct, btc_regime.btc_change_5m_pct)

        if signal_type == "STRONG_LONG":
            if not btc_regime.allow_alt_longs:
                # BTC is dumping. Allow ONLY if alt shows extreme decoupled relative strength
                if rs >= self.min_decoupled_rs_pct:
                    return True, f"DECOUPLED_STRENGTH (RS={rs:+.2f}%)"
                return False, f"BLOCKED_BY_BTC_DUMP (BTC 5m={btc_regime.btc_change_5m_pct:+.2f}%, RS={rs:+.2f}%)"

        elif signal_type == "STRONG_SHORT":
            if not btc_regime.allow_alt_shorts:
                # BTC is pumping. Allow ONLY if alt is severely weak
                if rs <= -self.min_decoupled_rs_pct:
                    return True, f"DECOUPLED_WEAKNESS (RS={rs:+.2f}%)"
                return False, f"BLOCKED_BY_BTC_PUMP (BTC 5m={btc_regime.btc_change_5m_pct:+.2f}%, RS={rs:+.2f}%)"

        return True, "PASSED"
