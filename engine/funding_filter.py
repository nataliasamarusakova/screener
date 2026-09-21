"""
Funding Settlement Epoch Proximity & Countdown Filter.
Protects against pre-funding frontrunning dumps and paying adverse funding fees
in the final 20 minutes before 8h/4h settlement.
"""
from __future__ import annotations

import math
import time
from typing import Optional
import msgspec


# FIX: raised from 0.0003 to 0.0010 based on academic literature:
# extreme funding is > 0.10% per 8h (≈0.3%/day, ≈110%/year), not 0.03%.
FUNDING_EXTREME_THRESHOLD_8H = 0.0010


class FundingGateResult(msgspec.Struct, gc=False):
    symbol: str
    minutes_to_settlement: float
    is_in_epoch_window: bool
    allow_long: bool
    allow_short: bool
    gate_reason: str
    is_favorable_for_long: bool = False    # NEW: negative funding = long receives payment
    is_favorable_for_short: bool = False   # NEW: positive funding = short receives payment


class FundingFilterEngine:
    """
    Evaluates time remaining until funding settlement and filters adverse positioning.
    """

    def __init__(
        self,
        proximity_threshold_minutes: float = 20.0,
        extreme_threshold_8h: float = FUNDING_EXTREME_THRESHOLD_8H,
    ) -> None:
        if not math.isfinite(proximity_threshold_minutes) or proximity_threshold_minutes <= 0.0:
            raise ValueError("proximity_threshold_minutes must be finite and positive")
        if not math.isfinite(extreme_threshold_8h) or extreme_threshold_8h <= 0.0:
            raise ValueError("extreme_threshold_8h must be finite and positive")
        self.proximity_threshold_minutes = float(proximity_threshold_minutes)
        self.extreme_threshold_8h = float(extreme_threshold_8h)

    def evaluate_funding_gate(
        self,
        symbol: str,
        signal_type: str,
        funding_rate_8h: float,
        next_funding_time_ms: int,
        current_time_ms: Optional[int] = None,
    ) -> FundingGateResult:
        """
        Evaluates whether a trade is safe to execute relative to the next funding epoch.
        Symmetric logic:
        - Positive extreme funding: blocks LONG, favors SHORT.
        - Negative extreme funding: blocks SHORT, favors LONG.
        """
        now_ms = current_time_ms if current_time_ms is not None else int(time.time() * 1000)

        try:
            funding_rate_8h = float(funding_rate_8h)
        except (TypeError, ValueError) as exc:
            raise ValueError("funding_rate_8h must be numeric") from exc
        if not math.isfinite(funding_rate_8h):
            return FundingGateResult(
                symbol=symbol, minutes_to_settlement=0.0, is_in_epoch_window=False,
                allow_long=False, allow_short=False, gate_reason="INVALID_FUNDING_RATE",
            )

        if next_funding_time_ms <= 0:
            return FundingGateResult(
                symbol=symbol, minutes_to_settlement=0.0, is_in_epoch_window=False,
                allow_long=False, allow_short=False, gate_reason="UNKNOWN_FUNDING_TIME",
            )

        if next_funding_time_ms <= now_ms:
            return FundingGateResult(
                symbol=symbol, minutes_to_settlement=0.0, is_in_epoch_window=False,
                allow_long=False, allow_short=False, gate_reason="STALE_FUNDING_TIME",
            )

        minutes_to_settlement = (next_funding_time_ms - now_ms) / (60.0 * 1000.0)
        is_in_window = minutes_to_settlement <= self.proximity_threshold_minutes

        allow_long = True
        allow_short = True
        reason = "PASSED"
        favorable_long = False
        favorable_short = False

        # Favorable funding flags (informational, not blocking)
        if funding_rate_8h <= -self.extreme_threshold_8h:
            favorable_long = True
        elif funding_rate_8h >= self.extreme_threshold_8h:
            favorable_short = True

        if is_in_window:
            # Positive extreme: longs pay, shorts receive. Block LONG, do not block SHORT.
            if funding_rate_8h >= self.extreme_threshold_8h:
                if signal_type == "STRONG_LONG":
                    allow_long = False
                    reason = (
                        f"BLOCKED_PRE_FUNDING_PAYOUT: Long pays {funding_rate_8h*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m (Pre-settlement dumping risk)"
                    )
                elif signal_type == "STRONG_SHORT":
                    # FIX: short receives payment, this is favorable, do not block
                    reason = (
                        f"FAVORABLE_FUNDING_SHORT: Short receives {funding_rate_8h*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m"
                    )
            # Negative extreme: shorts pay, longs receive. Block SHORT, do not block LONG.
            elif funding_rate_8h <= -self.extreme_threshold_8h:
                if signal_type == "STRONG_SHORT":
                    allow_short = False
                    reason = (
                        f"BLOCKED_PRE_FUNDING_PAYOUT: Short pays {abs(funding_rate_8h)*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m (Pre-settlement squeeze risk)"
                    )
                elif signal_type == "STRONG_LONG":
                    reason = (
                        f"FAVORABLE_FUNDING_LONG: Long receives {abs(funding_rate_8h)*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m"
                    )

        return FundingGateResult(
            symbol=symbol,
            minutes_to_settlement=round(minutes_to_settlement, 1),
            is_in_epoch_window=is_in_window,
            allow_long=allow_long,
            allow_short=allow_short,
            gate_reason=reason,
            is_favorable_for_long=favorable_long,
            is_favorable_for_short=favorable_short,
        )
