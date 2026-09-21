"""
BTC Market Regime & Relative Strength (Beta) Filter.
Prevents buying altcoin longs during market-wide BTC liquidation dumps,
and prevents shorting during BTC impulse rallies.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional
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
        beta_alt_btc: Optional[float] = None,  # Legacy compatibility only; production path requires rolling beta.
        min_decoupled_rs_pct: float = 1.2,     # Minimum Relative Strength to bypass dump gate
    ) -> None:
        threshold_values = (dump_threshold_5m_pct, pump_threshold_5m_pct, min_decoupled_rs_pct)
        if not all(math.isfinite(float(x)) for x in threshold_values):
            raise ValueError("Market regime configuration must be finite")
        if dump_threshold_5m_pct >= pump_threshold_5m_pct or min_decoupled_rs_pct <= 0.0:
            raise ValueError("Market regime configuration has invalid bounds")
        if beta_alt_btc is not None and (not math.isfinite(beta_alt_btc) or beta_alt_btc <= 0.0):
            raise ValueError("beta_alt_btc must be finite and positive when supplied")
        self.dump_threshold_5m_pct = float(dump_threshold_5m_pct)
        self.pump_threshold_5m_pct = float(pump_threshold_5m_pct)
        self.beta_alt_btc = beta_alt_btc
        self.min_decoupled_rs_pct = float(min_decoupled_rs_pct)

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
                btc_change_5m_pct=change_5m_pct,
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=False,  # Block alt longs during BTC dumps
                allow_alt_shorts=True,
            )
        elif change_5m_pct >= self.pump_threshold_5m_pct:
            return BTCRegime(
                status="IMPULSE_PUMP",
                btc_price=btc_last_price,
                btc_change_5m_pct=change_5m_pct,
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=True,
                allow_alt_shorts=False,  # Block alt shorts during BTC pumps
            )
        else:
            return BTCRegime(
                status="NEUTRAL_RANGING",
                btc_price=btc_last_price,
                btc_change_5m_pct=change_5m_pct,
                btc_change_24h_pct=round(btc_change_24h_pct, 2),
                allow_alt_longs=True,
                allow_alt_shorts=True,
            )

    def calculate_relative_strength(
        self,
        alt_change_5m_pct: float,
        btc_change_5m_pct: float,
        beta: Optional[float] = None,
    ) -> float:
        """
        Calculates Beta-adjusted Relative Strength (RS):
        RS = Delta P_alt - beta * Delta P_btc.
        Positive RS indicates the altcoin is outperforming BTC.
        """
        beta_value = beta if beta is not None else self.beta_alt_btc
        if beta_value is None:
            raise ValueError("rolling beta is required for relative-strength calculation")
        if not math.isfinite(beta_value) or beta_value <= 0.0:
            raise ValueError("beta must be finite and positive")
        return alt_change_5m_pct - (beta_value * btc_change_5m_pct)

    @staticmethod
    def calculate_rolling_beta(
        alt_times_ms: Sequence[int],
        alt_prices: Sequence[float],
        btc_times_ms: Sequence[int],
        btc_prices: Sequence[float],
        min_samples: int = 24,
    ) -> Optional[float]:
        """Estimate beta from aligned historical 5m closes, excluding the current bar."""
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        alt_map = {int(t): float(p) for t, p in zip(alt_times_ms, alt_prices) if float(p) > 0.0 and math.isfinite(float(p))}
        btc_map = {int(t): float(p) for t, p in zip(btc_times_ms, btc_prices) if float(p) > 0.0 and math.isfinite(float(p))}
        common = sorted(set(alt_map) & set(btc_map))
        if len(common) < min_samples + 1:
            return None

        common = common[-(min_samples + 1):]
        step_ms = 5 * 60 * 1000
        if any(cur_t - prev_t != step_ms for prev_t, cur_t in zip(common[:-1], common[1:])):
            return None

        alt_rets = []
        btc_rets = []
        for prev_t, cur_t in zip(common[:-1], common[1:]):
            alt_rets.append((alt_map[cur_t] / alt_map[prev_t]) - 1.0)
            btc_rets.append((btc_map[cur_t] / btc_map[prev_t]) - 1.0)
        if len(alt_rets) < min_samples:
            return None

        alt_mean = sum(alt_rets) / len(alt_rets)
        btc_mean = sum(btc_rets) / len(btc_rets)
        cov = sum((a - alt_mean) * (b - btc_mean) for a, b in zip(alt_rets, btc_rets))
        var_btc = sum((b - btc_mean) ** 2 for b in btc_rets)
        if var_btc <= 1e-16:
            return None
        beta = cov / var_btc
        return beta if math.isfinite(beta) else None

    def check_signal_gate(
        self,
        symbol: str,
        signal_type: str,
        btc_regime: BTCRegime,
        alt_change_5m_pct: float,
        beta: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Gates signals based on BTC correlation.
        Returns: (is_allowed, reason)
        """
        if symbol.upper() in ("BTCUSDT", "BTCUSDT_240628"):
            return True, "BTC_SELF"
        if beta is None:
            return False, "MISSING_ROLLING_BETA"

        rs = self.calculate_relative_strength(alt_change_5m_pct, btc_regime.btc_change_5m_pct, beta=beta)

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
