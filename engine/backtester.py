"""
Point-in-time walk-forward backtesting with CEX frictions and Deflated Sharpe Ratio.

The backtest frequency is the frequency of the input bars (production: closed 5m bars).
Trade PnL includes both execution price impact and exchange fees.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import msgspec
import numpy as np
import polars as pl

from engine.execution_cost import CEXCostModel


class BacktestMetrics(msgspec.Struct, gc=False):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_net_pnl_pct: float
    profit_factor: float
    max_drawdown_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    deflated_sharpe_ratio: float
    is_statistically_significant: bool
    skewness: float
    kurtosis: float
    n_trials: int = 0


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    returns: np.ndarray,
    n_trials: Optional[int] = None,
    var_trials_sr: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute DSR from per-period returns.

    n_trials must be the actual number of model/parameter trials that were searched.
    The multiple-testing variance must be supplied for n_trials > 1; there is no
    defensible default in the engine.

    Returns: (DSR probability [0,1], skewness, raw kurtosis).
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    T = len(returns)
    if T < 5:
        return 0.0, 0.0, 3.0
    if not math.isfinite(observed_sr):
        return 0.0, 0.0, 3.0

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r <= 1e-12:
        return 0.0, 0.0, 3.0

    diffs = returns - mean_r
    m3 = float(np.mean(diffs ** 3))
    m4 = float(np.mean(diffs ** 4))
    skewness = m3 / (std_r ** 3)
    raw_kurtosis = m4 / (std_r ** 4)

    if n_trials is None:
        return 0.0, skewness, raw_kurtosis
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1 when supplied")

    euler_mascheroni = 0.5772156649
    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        if var_trials_sr is None or not math.isfinite(var_trials_sr) or var_trials_sr <= 0.0:
            raise ValueError("var_trials_sr is required and must be positive when n_trials > 1")
        norm_inv1 = math.sqrt(2.0) * _erfinv(2.0 * (1.0 - 1.0 / n_trials) - 1.0)
        norm_inv2 = math.sqrt(2.0) * _erfinv(2.0 * (1.0 - 1.0 / (n_trials * math.e)) - 1.0)
        sr_benchmark = math.sqrt(var_trials_sr) * (
            (1.0 - euler_mascheroni) * norm_inv1 + euler_mascheroni * norm_inv2
        )

    # Bailey/López de Prado convention uses raw kurtosis gamma4 (normal ~= 3).
    denom_term = 1.0 - skewness * observed_sr + ((raw_kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
    if denom_term <= 0.0:
        return 0.0, skewness, raw_kurtosis

    se_sr = math.sqrt(denom_term / (T - 1.0))
    z_stat = (observed_sr - sr_benchmark) / se_sr if se_sr > 0.0 else 0.0
    dsr = 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))
    return max(0.0, min(1.0, dsr)), skewness, raw_kurtosis


def _erfinv(y: float) -> float:
    """Winitzki approximation of the inverse error function."""
    y = max(-0.999999, min(0.999999, y))
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    ln1_y2 = math.log(1.0 - y * y)
    term1 = 2.0 / (math.pi * a) + ln1_y2 / 2.0
    val = term1 * term1 - ln1_y2 / a
    return sgn * math.sqrt(max(0.0, math.sqrt(val) - term1))


class QuantBacktester:
    """Backtester using closed-bar next-open entries and per-bar risk metrics."""

    def __init__(
        self,
        cost_model: Optional[CEXCostModel] = None,
        annualization_factor: float = 365.0 * 24.0 * 12.0,  # 5-minute bars/year; validated by timestamp cadence.
        market_spread_bps: float = 1.5,  # Backtest input; TODO: replace with symbol/time-varying spread observations.
    ) -> None:
        if not math.isfinite(float(annualization_factor)) or annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be finite and positive")
        if not math.isfinite(float(market_spread_bps)) or market_spread_bps < 0.0:
            raise ValueError("market_spread_bps must be finite and non-negative")
        self.cost_model = cost_model or CEXCostModel()
        self.annualization_factor = annualization_factor
        self.market_spread_bps = market_spread_bps

    @staticmethod
    def _net_return_long(entry, exit_) -> float:
        entry_fee = entry.fee_bps / 10000.0
        exit_fee = exit_.fee_bps / 10000.0
        return (
            (exit_.executed_price * (1.0 - exit_fee))
            - (entry.executed_price * (1.0 + entry_fee))
        ) / entry.executed_price

    @staticmethod
    def _net_return_short(entry, exit_) -> float:
        entry_fee = entry.fee_bps / 10000.0
        exit_fee = exit_.fee_bps / 10000.0
        return (entry.executed_price * (1.0 - entry_fee) - exit_.executed_price * (1.0 + exit_fee)) / entry.executed_price

    def run_backtest(
        self,
        df: pl.DataFrame,
        score_column: str = "composite_score",
        price_column: str = "close",
        open_price_column: str = "open",
        timestamp_column: str = "timestamp_ms",
        long_threshold: float = 75.0,
        short_threshold: float = -75.0,
        holding_bars: int = 6,
        n_trials: Optional[int] = None,
        var_trials_sr: Optional[float] = None,
    ) -> BacktestMetrics:
        """
        Signal on bar i executes at bar i+1 OPEN and exits at a later CLOSED bar.
        The input must contain strictly increasing, gap-free 5m timestamps because
        annualization_factor is explicitly the 5m bars/year convention.
        Returns are fixed-notional per-bar PnL, including execution costs and fees;
        inactive bars are zero returns.
        """
        required = (score_column, price_column, open_price_column, timestamp_column)
        if df.is_empty() or any(col not in df.columns for col in required):
            if timestamp_column not in df.columns:
                raise ValueError(f"Backtest requires {timestamp_column!r} with 5m timestamps")
            return self._empty_metrics()
        if holding_bars < 1:
            return self._empty_metrics()

        timestamps = df[timestamp_column].cast(pl.Int64).to_numpy()
        prices = df[price_column].cast(pl.Float64).to_numpy()
        opens = df[open_price_column].cast(pl.Float64).to_numpy()
        scores = df[score_column].cast(pl.Float64).to_numpy()
        n_bars = len(prices)
        if n_bars < holding_bars + 2:
            return self._empty_metrics()
        if not (np.isfinite(prices).all() and np.isfinite(opens).all()):
            return self._empty_metrics()
        if np.any(prices <= 0.0) or np.any(opens <= 0.0):
            return self._empty_metrics()
        if len(timestamps) != n_bars or n_bars < 2:
            return self._empty_metrics()
        timestamp_diffs = np.diff(timestamps)
        if not np.all(timestamp_diffs == 5 * 60 * 1000):
            raise ValueError("Backtest timestamps must be strictly contiguous 5m bars")

        bar_returns = np.zeros(n_bars, dtype=np.float64)
        trade_returns: list[float] = []
        i = 0

        while i < n_bars - holding_bars - 1:
            score = scores[i]
            if not math.isfinite(float(score)):
                i += 1
                continue

            if score >= long_threshold or score <= short_threshold:
                side = "BUY" if score >= long_threshold else "SELL"
                entry_idx = i + 1
                exit_idx = min(entry_idx + holding_bars, n_bars - 1)
                entry_ref = float(opens[entry_idx])
                exit_ref = float(prices[exit_idx])
                if entry_ref <= 0.0 or exit_ref <= 0.0:
                    i += 1
                    continue

                entry_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT",
                    side=side,
                    reference_price=entry_ref,
                    spread_bps=self.market_spread_bps,
                    is_market_order=True,
                )
                exit_side = "SELL" if side == "BUY" else "BUY"
                exit_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT",
                    side=exit_side,
                    reference_price=exit_ref,
                    spread_bps=self.market_spread_bps,
                    is_market_order=True,
                )

                net_return = (
                    self._net_return_long(entry_exec, exit_exec)
                    if side == "BUY"
                    else self._net_return_short(entry_exec, exit_exec)
                )
                if not math.isfinite(net_return):
                    i = exit_idx
                    continue

                # Decompose the exact execution return into non-overlapping 5m marks.
                # The denominator is the executed entry price, matching _net_return_*.
                entry_fee = entry_exec.fee_bps / 10000.0
                exit_fee = exit_exec.fee_bps / 10000.0
                entry_exec_price = entry_exec.executed_price
                if side == "BUY":
                    bar_returns[entry_idx] += (float(prices[entry_idx]) - entry_exec_price * (1.0 + entry_fee)) / entry_exec_price
                    for j in range(entry_idx + 1, exit_idx):
                        bar_returns[j] += (float(prices[j]) - float(prices[j - 1])) / entry_exec_price
                    bar_returns[exit_idx] += (exit_exec.executed_price * (1.0 - exit_fee) - float(prices[exit_idx - 1])) / entry_exec_price
                else:
                    bar_returns[entry_idx] += (entry_exec_price * (1.0 - entry_fee) - float(prices[entry_idx])) / entry_exec_price
                    for j in range(entry_idx + 1, exit_idx):
                        bar_returns[j] += (float(prices[j - 1]) - float(prices[j])) / entry_exec_price
                    bar_returns[exit_idx] += (float(prices[exit_idx - 1]) - exit_exec.executed_price * (1.0 + exit_fee)) / entry_exec_price

                trade_returns.append(net_return)
                i = exit_idx
            else:
                i += 1

        if not trade_returns:
            return self._empty_metrics()

        r_arr = np.asarray(trade_returns, dtype=np.float64)
        wins = r_arr[r_arr > 0.0]
        losses = r_arr[r_arr < 0.0]
        total_trades = len(r_arr)
        win_trades = len(wins)
        loss_trades = len(losses)

        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = abs(float(np.sum(losses)))
        profit_factor = gross_profit / max(gross_loss, 1e-12)

        equity_curve = 1.0 + np.cumsum(bar_returns)
        if not np.isfinite(equity_curve).all() or equity_curve[-1] <= 0.0:
            return self._empty_metrics()
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_dd_pct = float(np.max(drawdowns)) * 100.0
        total_net_pnl_pct = (equity_curve[-1] - 1.0) * 100.0

        mean_bar = float(np.mean(bar_returns))
        std_bar = float(np.std(bar_returns, ddof=1)) if n_bars > 1 else 0.0
        sharpe_per_period = mean_bar / std_bar if std_bar > 1e-12 else 0.0
        sharpe_annualized = sharpe_per_period * math.sqrt(self.annualization_factor)

        downside = np.minimum(bar_returns, 0.0)
        downside_deviation = math.sqrt(float(np.mean(downside ** 2)))
        sortino = (mean_bar / downside_deviation) * math.sqrt(self.annualization_factor) if downside_deviation > 1e-12 else 0.0

        dsr, skew, kurt = compute_deflated_sharpe_ratio(
            observed_sr=sharpe_per_period,
            returns=bar_returns,
            n_trials=n_trials,
            var_trials_sr=var_trials_sr,
        )

        annualized_return_pct = float(total_net_pnl_pct * (self.annualization_factor / n_bars))
        annualized_volatility_pct = std_bar * math.sqrt(self.annualization_factor) * 100.0

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=win_trades,
            losing_trades=loss_trades,
            win_rate_pct=round((win_trades / total_trades) * 100.0, 2),
            total_net_pnl_pct=round(total_net_pnl_pct, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            annualized_return_pct=round(annualized_return_pct, 2),
            annualized_volatility_pct=round(annualized_volatility_pct, 2),
            sharpe_ratio=round(sharpe_annualized, 2),
            sortino_ratio=round(sortino, 2),
            deflated_sharpe_ratio=round(dsr, 4),
            is_statistically_significant=(dsr >= 0.95) if n_trials is not None else False,
            skewness=round(skew, 2),
            kurtosis=round(kurt, 2),
            n_trials=int(n_trials or 0),
        )

    def _empty_metrics(self) -> BacktestMetrics:
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            total_net_pnl_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            annualized_return_pct=0.0,
            annualized_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            deflated_sharpe_ratio=0.0,
            is_statistically_significant=False,
            skewness=0.0,
            kurtosis=3.0,
            n_trials=0,
        )
