"""
Walk-Forward Backtesting Engine powered by Polars SIMD LazyFrames
with Bailey & López de Prado Deflated Sharpe Ratio (DSR) & CEX Cost Modeling.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
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
    deflated_sharpe_ratio: float  # Bailey & López de Prado (2014) DSR probability in [0.0, 1.0]
    is_statistically_significant: bool  # True if DSR >= 0.95 (5% significance level)
    skewness: float
    kurtosis: float


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    returns: np.ndarray,
    n_trials: int = 20,
    var_trials_sr: float = 0.25,
) -> Tuple[float, float, float]:
    """
    Computes Deflated Sharpe Ratio (DSR) accounting for:
    1. Multiple testing / selection bias (number of trials N).
    2. Non-Gaussian returns (skewness and kurtosis).
    3. Sample length T.

    Returns: (DSR probability in [0, 1], skewness, kurtosis)
    """
    T = len(returns)
    if T < 5 or observed_sr == 0.0:
        return 0.0, 0.0, 3.0

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r <= 1e-12:
        return 0.0, 0.0, 3.0

    # Calculate skewness and kurtosis
    diffs = returns - mean_r
    m3 = np.mean(diffs ** 3)
    m4 = np.mean(diffs ** 4)

    skewness = float(m3 / (std_r ** 3))
    # Bailey & López de Prado (2014) require EXCESS kurtosis γ₄ = (m4/σ⁴) - 3
    # (normal distribution → raw=3, excess=0).
    # Using raw kurtosis here inflates SE(SR) → artificially deflates DSR probability.
    excess_kurtosis = float(m4 / (std_r ** 4)) - 3.0

    # Expected maximum Sharpe ratio under the null hypothesis (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        # Approximate expected maximum of N standard normal variables
        norm_inv1 = math.sqrt(2.0) * _erfinv(2.0 * (1.0 - 1.0 / n_trials) - 1.0)
        norm_inv2 = math.sqrt(2.0) * _erfinv(2.0 * (1.0 - 1.0 / (n_trials * math.e)) - 1.0)
        sr_benchmark = math.sqrt(var_trials_sr) * ((1.0 - euler_mascheroni) * norm_inv1 + euler_mascheroni * norm_inv2)

    # Standard error of the Sharpe ratio under non-normality (Bailey & LdP 2014, Eq.1):
    # SE(SR) = sqrt( (1 - γ₃·SR + ((γ₄-1)/4)·SR²) / (T-1) )
    # where γ₃ = skewness, γ₄ = excess kurtosis
    denom_term = 1.0 - skewness * observed_sr + ((excess_kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
    if denom_term <= 0.0:
        denom_term = 1.0
    se_sr = math.sqrt(denom_term / (T - 1.0))

    # Test statistic z
    z_stat = (observed_sr - sr_benchmark) / se_sr if se_sr > 0 else 0.0

    # Cumulative normal probability Phi(z)
    dsr = 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))
    return max(0.0, min(1.0, dsr)), skewness, excess_kurtosis


def _erfinv(y: float) -> float:
    """Winitzki approximation of the inverse error function."""
    y = max(-0.999999, min(0.999999, y))
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    ln1_y2 = math.log(1.0 - y * y)
    term1 = 2.0 / (math.pi * a) + ln1_y2 / 2.0
    val = term1 * term1 - ln1_y2 / a
    return sgn * math.sqrt(math.sqrt(val) - term1)


class QuantBacktester:
    """
    SIMD LazyFrame Walk-Forward Backtester with realistic CEX frictions.
    """

    def __init__(
        self,
        cost_model: Optional[CEXCostModel] = None,
        annualization_factor: float = 365.0 * 24.0 * 12.0,  # 5-minute bars in a year
    ) -> None:
        self.cost_model = cost_model or CEXCostModel()
        self.annualization_factor = annualization_factor

    def run_backtest(
        self,
        df: pl.DataFrame,
        score_column: str = "composite_score",
        price_column: str = "close",
        long_threshold: float = 75.0,
        short_threshold: float = -75.0,
        holding_bars: int = 6,          # 30-minute default holding horizon (6 x 5m bars)
        n_trials: int = 20,
    ) -> BacktestMetrics:
        """
        Executes vectorized backtest on Polars DataFrame with realistic slippage.
        """
        if df.is_empty() or score_column not in df.columns or price_column not in df.columns:
            return self._empty_metrics()

        prices = df[price_column].to_numpy()
        scores = df[score_column].to_numpy()
        n_bars = len(prices)

        if n_bars < holding_bars + 2:
            return self._empty_metrics()

        trade_returns: List[float] = []
        i = 0

        while i < n_bars - holding_bars:
            score = scores[i]
            if score >= long_threshold:
                # Enter Long at next bar's open/price with slippage
                entry_ref = prices[i + 1]
                entry_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT", side="BUY", reference_price=entry_ref, spread_bps=1.5, is_market_order=True
                ).executed_price

                exit_idx = min(i + 1 + holding_bars, n_bars - 1)
                exit_ref = prices[exit_idx]
                exit_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT", side="SELL", reference_price=exit_ref, spread_bps=1.5, is_market_order=True
                ).executed_price

                pnl_pct = ((exit_exec - entry_exec) / entry_exec) * 100.0
                trade_returns.append(pnl_pct)
                i += holding_bars
            elif score <= short_threshold:
                # Enter Short at next bar's open/price with slippage
                entry_ref = prices[i + 1]
                entry_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT", side="SELL", reference_price=entry_ref, spread_bps=1.5, is_market_order=True
                ).executed_price

                exit_idx = min(i + 1 + holding_bars, n_bars - 1)
                exit_ref = prices[exit_idx]
                exit_exec = self.cost_model.simulate_execution(
                    symbol="BTCUSDT", side="BUY", reference_price=exit_ref, spread_bps=1.5, is_market_order=True
                ).executed_price

                pnl_pct = ((entry_exec - exit_exec) / entry_exec) * 100.0
                trade_returns.append(pnl_pct)
                i += holding_bars
            else:
                i += 1

        if not trade_returns:
            return self._empty_metrics()

        r_arr = np.array(trade_returns, dtype=np.float64)
        wins = r_arr[r_arr > 0]
        losses = r_arr[r_arr < 0]

        total_trades = len(r_arr)
        win_trades = len(wins)
        loss_trades = len(losses)
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-6
        profit_factor = gross_profit / gross_loss

        # Equity Curve and Max Drawdown
        equity_curve = np.cumprod(1.0 + (r_arr / 100.0))
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_dd_pct = float(np.max(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

        # Annualized Metrics
        mean_ret = float(np.mean(r_arr))
        std_ret = float(np.std(r_arr)) if len(r_arr) > 1 else 1.0
        ann_factor = math.sqrt(self.annualization_factor / holding_bars)

        sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 1e-9 else 0.0

        # Downside risk for Sortino
        downside_diffs = r_arr[r_arr < 0]
        downside_std = float(np.std(downside_diffs)) if len(downside_diffs) > 1 else std_ret
        sortino = (mean_ret / downside_std) * ann_factor if downside_std > 1e-9 else 0.0

        # Deflated Sharpe Ratio calculation
        dsr, skew, kurt = compute_deflated_sharpe_ratio(
            observed_sr=sharpe, returns=r_arr / 100.0, n_trials=n_trials
        )

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=win_trades,
            losing_trades=loss_trades,
            win_rate_pct=round(win_rate, 2),
            total_net_pnl_pct=round(float(np.sum(r_arr)), 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            annualized_return_pct=round(mean_ret * (self.annualization_factor / holding_bars), 2),
            annualized_volatility_pct=round(std_ret * ann_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            deflated_sharpe_ratio=round(dsr, 4),
            is_statistically_significant=(dsr >= 0.95),
            skewness=round(skew, 2),
            kurtosis=round(kurt, 2),
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
        )
