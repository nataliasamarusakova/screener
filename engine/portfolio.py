"""Portfolio-level simulation over event-driven trade candidates."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Sequence, Any

import numpy as np

from engine.research import ResearchDataset, TradeRecord


@dataclass(frozen=True)
class PortfolioConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    max_leverage: float = 3.0
    max_aggregate_risk_pct: float = 0.03
    max_strong_per_direction: int = 3
    max_portfolio_correlation: float = 0.80
    correlation_lookback_bars: int = 288
    max_drawdown_pct: float = 0.20
    daily_loss_limit_pct: float = 0.08

    def __post_init__(self) -> None:
        if self.initial_equity <= 0 or not 0 < self.risk_per_trade_pct < 1:
            raise ValueError("invalid portfolio capital/risk configuration")
        if self.max_leverage < 1 or self.max_aggregate_risk_pct <= 0:
            raise ValueError("invalid portfolio caps")
        if not 0.0 < self.max_drawdown_pct < 1.0 or not 0.0 < self.daily_loss_limit_pct < 1.0:
            raise ValueError("invalid drawdown/daily loss limits")


@dataclass
class PortfolioFill:
    signal_timestamp_ms: int
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    symbol: str
    side: str
    notional_usd: float
    stop_risk_usd: float
    pnl_usd: float
    equity_before: float
    equity_after: float
    blocked: bool = False
    block_reason: str = ""


@dataclass
class PortfolioReport:
    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    profit_factor: float | None
    trades_executed: int
    trades_blocked: int
    block_reasons: dict[str, int]
    long_trades: int
    short_trades: int


def _returns_for_symbol(dataset: ResearchDataset, symbol: str, timestamp_ms: int, lookback: int) -> np.ndarray:
    rows = dataset.bars(symbol)
    eligible = [r for r in rows if r.timestamp_ms <= timestamp_ms]
    if len(eligible) < 3:
        return np.asarray([], dtype=float)
    closes = np.asarray([r.close for r in eligible[-(lookback + 1):]], dtype=float)
    return closes[1:] / closes[:-1] - 1.0


def rolling_correlation(dataset: ResearchDataset, a: str, b: str, timestamp_ms: int, lookback: int) -> float | None:
    ra = _returns_for_symbol(dataset, a, timestamp_ms, lookback)
    rb = _returns_for_symbol(dataset, b, timestamp_ms, lookback)
    n = min(len(ra), len(rb))
    if n < 20:
        return None
    ra = ra[-n:]
    rb = rb[-n:]
    if float(np.std(ra, ddof=1)) <= 1e-15 or float(np.std(rb, ddof=1)) <= 1e-15:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


class PortfolioSimulator:
    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()

    def run(self, dataset: ResearchDataset, candidates: Sequence[TradeRecord]) -> tuple[list[PortfolioFill], PortfolioReport]:
        ordered = sorted(candidates, key=lambda t: (t.entry_timestamp_ms, t.exit_timestamp_ms, t.symbol))
        open_positions: dict[str, PortfolioFill] = {}
        fills: list[PortfolioFill] = []
        equity = self.config.initial_equity
        peak = equity
        max_dd = 0.0
        block_reasons: dict[str, int] = defaultdict(int)
        daily_start: dict[str, float] = {}

        def close_positions(now: int) -> None:
            nonlocal equity, peak, max_dd
            due = [symbol for symbol, pos in open_positions.items() if pos.exit_timestamp_ms <= now]
            for symbol in sorted(due):
                pos = open_positions.pop(symbol)
                equity += pos.pnl_usd
                pos.equity_after = equity
                peak = max(peak, equity)
                dd = 1.0 - equity / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

        for trade in ordered:
            close_positions(trade.entry_timestamp_ms)
            day_key = str(trade.entry_timestamp_ms // 86_400_000)
            day_start_equity = daily_start.setdefault(day_key, equity)
            peak_drawdown = 1.0 - equity / peak if peak > 0.0 else 0.0
            day_loss = 1.0 - equity / day_start_equity if day_start_equity > 0.0 else 0.0
            if peak_drawdown >= self.config.max_drawdown_pct:
                reason = "MAX_DRAWDOWN_HALT"
            elif day_loss >= self.config.daily_loss_limit_pct:
                reason = "DAILY_LOSS_HALT"
            elif trade.symbol in open_positions:
                reason = "SYMBOL_ALREADY_OPEN"
            else:
                active = list(open_positions.values())
                same_direction = sum(1 for p in active if p.side == trade.side)
                if same_direction >= self.config.max_strong_per_direction:
                    reason = "DIRECTION_CAP"
                else:
                    stop_distance_pct = abs(trade.entry_price - trade.invalidation_price) / trade.entry_price
                    if stop_distance_pct <= 0:
                        reason = "INVALID_STOP_DISTANCE"
                    else:
                        risk_usd = equity * self.config.risk_per_trade_pct
                        existing_risk = sum(p.stop_risk_usd for p in active)
                        if existing_risk + risk_usd > equity * self.config.max_aggregate_risk_pct + 1e-9:
                            reason = "AGGREGATE_RISK_CAP"
                        else:
                            reason = ""
                            for pos in active:
                                corr = rolling_correlation(dataset, pos.symbol, trade.symbol, trade.entry_timestamp_ms, self.config.correlation_lookback_bars)
                                # Only positive correlation represents concentration; negative correlation is diversification.
                                if corr is not None and corr > self.config.max_portfolio_correlation:
                                    reason = f"CORRELATION_{corr:.2f}_WITH_{pos.symbol}"
                                    break
                            if not reason:
                                notional = min(risk_usd / stop_distance_pct, equity * self.config.max_leverage)
                                pnl = notional * trade.net_return
                                fill = PortfolioFill(
                                    signal_timestamp_ms=trade.signal_timestamp_ms,
                                    entry_timestamp_ms=trade.entry_timestamp_ms,
                                    exit_timestamp_ms=trade.exit_timestamp_ms,
                                    symbol=trade.symbol,
                                    side=trade.side,
                                    notional_usd=notional,
                                    stop_risk_usd=notional * stop_distance_pct,
                                    pnl_usd=pnl,
                                    equity_before=equity,
                                    equity_after=equity + pnl,
                                )
                                open_positions[trade.symbol] = fill
                                fills.append(fill)
                                continue
            block_reasons[reason] += 1
            fills.append(
                PortfolioFill(
                    signal_timestamp_ms=trade.signal_timestamp_ms,
                    entry_timestamp_ms=trade.entry_timestamp_ms,
                    exit_timestamp_ms=trade.exit_timestamp_ms,
                    symbol=trade.symbol,
                    side=trade.side,
                    notional_usd=0.0,
                    stop_risk_usd=0.0,
                    pnl_usd=0.0,
                    equity_before=equity,
                    equity_after=equity,
                    blocked=True,
                    block_reason=reason,
                )
            )

        close_positions(2**63 - 1)
        executed = [f for f in fills if not f.blocked]
        pnls = [f.pnl_usd for f in executed]
        gains = sum(x for x in pnls if x > 0)
        losses = -sum(x for x in pnls if x < 0)
        # Reconstruct equity after each executed exit in chronological order.
        eq = self.config.initial_equity
        peak_eq = eq
        max_dd = 0.0
        for fill in sorted(executed, key=lambda f: (f.exit_timestamp_ms, f.symbol)):
            eq += fill.pnl_usd
            peak_eq = max(peak_eq, eq)
            if peak_eq > 0:
                max_dd = max(max_dd, 1.0 - eq / peak_eq)
        report = PortfolioReport(
            initial_equity=self.config.initial_equity,
            final_equity=eq,
            total_return=eq / self.config.initial_equity - 1.0,
            max_drawdown=max_dd,
            profit_factor=(gains / losses if losses > 0 else None),
            trades_executed=len(executed),
            trades_blocked=len(fills) - len(executed),
            block_reasons=dict(block_reasons),
            long_trades=sum(f.side == "LONG" for f in executed),
            short_trades=sum(f.side == "SHORT" for f in executed),
        )
        return fills, report


def report_to_dict(report: PortfolioReport) -> dict[str, Any]:
    return asdict(report)
