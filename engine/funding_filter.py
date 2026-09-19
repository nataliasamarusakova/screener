"""
Funding Settlement Epoch Proximity & Countdown Filter.
Protects against pre-funding frontrunning dumps and paying adverse funding fees
in the final 15-25 minutes before 8h/4h settlement.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple
import msgspec


class FundingGateResult(msgspec.Struct, gc=False):
    symbol: str
    minutes_to_settlement: float
    is_in_epoch_window: bool             # True if < threshold minutes to settlement
    allow_long: bool
    allow_short: bool
    gate_reason: str


class FundingFilterEngine:
    """
    Evaluates time remaining until funding settlement and filters adverse positioning.
    """

    def __init__(self, proximity_threshold_minutes: float = 20.0) -> None:
        self.proximity_threshold_minutes = proximity_threshold_minutes

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
        """
        now_ms = current_time_ms if current_time_ms is not None else int(time.time() * 1000)

        try:
            funding_rate_8h = float(funding_rate_8h)
        except (TypeError, ValueError) as exc:
            raise ValueError("funding_rate_8h must be numeric") from exc
        if not math.isfinite(funding_rate_8h):
            return FundingGateResult(
                symbol=symbol,
                minutes_to_settlement=0.0,
                is_in_epoch_window=False,
                allow_long=False,
                allow_short=False,
                gate_reason="INVALID_FUNDING_RATE",
            )

        if next_funding_time_ms <= 0:
            return FundingGateResult(
                symbol=symbol,
                minutes_to_settlement=0.0,
                is_in_epoch_window=False,
                allow_long=False,
                allow_short=False,
                gate_reason="UNKNOWN_FUNDING_TIME",
            )

        if next_funding_time_ms <= now_ms:
            return FundingGateResult(
                symbol=symbol,
                minutes_to_settlement=0.0,
                is_in_epoch_window=False,
                allow_long=False,
                allow_short=False,
                gate_reason="STALE_FUNDING_TIME",
            )

        minutes_to_settlement = (next_funding_time_ms - now_ms) / (60.0 * 1000.0)
        is_in_window = minutes_to_settlement <= self.proximity_threshold_minutes

        allow_long = True
        allow_short = True
        reason = "PASSED"

        if is_in_window:
            # Overheated positive funding: Longs pay Shorts
            if funding_rate_8h >= 0.0003:  # +0.03%
                if signal_type == "STRONG_LONG":
                    allow_long = False
                    reason = (
                        f"BLOCKED_PRE_FUNDING_PAYOUT: Long pays {funding_rate_8h*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m (Pre-settlement dumping risk)"
                    )
            # Heavy negative funding: Shorts pay Longs
            elif funding_rate_8h <= -0.0003:  # -0.03%
                if signal_type == "STRONG_SHORT":
                    allow_short = False
                    reason = (
                        f"BLOCKED_PRE_FUNDING_PAYOUT: Short pays {abs(funding_rate_8h)*100:+.3f}% "
                        f"in {minutes_to_settlement:.1f}m (Pre-settlement squeeze risk)"
                    )

        return FundingGateResult(
            symbol=symbol,
            minutes_to_settlement=round(minutes_to_settlement, 1),
            is_in_epoch_window=is_in_window,
            allow_long=allow_long,
            allow_short=allow_short,
            gate_reason=reason,
        )
