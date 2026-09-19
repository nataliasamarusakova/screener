"""
Core zero-GC data contracts for Quantitative Crypto Derivatives Platform.
Powered strictly by msgspec.Struct(gc=False) for zero-copy deserialization and zero GC pressure.
"""
from __future__ import annotations
from typing import Optional
import msgspec


class NormalizedTrade(msgspec.Struct, gc=False):
    """
    Normalized taker trade event from crypto derivative exchanges.
    Strictly zero GC overhead.
    """
    symbol: str
    price: float
    quantity: float
    quote_quantity: float
    side: str              # "BUY" (taker bought, maker sold) or "SELL" (taker sold, maker bought)
    timestamp_ms: int
    is_buyer_maker: bool   # True = sell taker trade (buyer was maker), False = buy taker trade
    trade_id: int


class OrderBookLevel(msgspec.Struct, gc=False):
    """Single price-quantity level in L2 order book."""
    price: float
    qty: float


class OrderBookSnapshot(msgspec.Struct, gc=False):
    """
    Synchronized L2 OrderBook snapshot with microstructure metrics.
    Zero GC allocation.
    """
    symbol: str
    last_update_id: int
    timestamp_ms: int
    bids: tuple[tuple[float, float], ...]  # Tuple of (price, qty)
    asks: tuple[tuple[float, float], ...]  # Tuple of (price, qty)
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_bps: float
    obi_depth5: float       # Order Book Imbalance across top 5 levels: (BidQty - AskQty) / (BidQty + AskQty)
    obi_depth10: float      # Order Book Imbalance across top 10 levels


class NormalizedFunding(msgspec.Struct, gc=False):
    """
    Normalized funding rate brought to a unified 8-hour basis and annualized scale.
    Includes Perp vs Spot Index basis spread.
    """
    symbol: str
    raw_rate: float
    interval_hours: float
    normalized_8h_rate: float   # Normalized to 8h: (1 + raw)^(8 / interval_h) - 1
    annualized_rate: float      # Annualized percentage (normalized_8h_rate * 3 * 365 * 100)
    mark_price: float
    index_price: float
    basis_spread: float         # mark_price - index_price
    basis_spread_bps: float     # ((mark_price - index_price) / index_price) * 10000
    next_funding_time_ms: int
    timestamp_ms: int


class SyntheticLiquidation(msgspec.Struct, gc=False):
    """
    Reconstructed synthetic liquidation event.
    Detects hidden liquidation cascades throttled by exchange WebSocket endpoints.
    Triggered when -DeltaOI anomalously outstrips market taker volume during volatility spikes.
    """
    symbol: str
    timestamp_ms: int
    price: float
    delta_oi: float                     # Negative drop in Open Interest
    taker_volume: float                 # Recorded market taker volume in the same interval
    side: str                           # "LONG_LIQUIDATION" (forced sell) or "SHORT_LIQUIDATION" (forced buy)
    estimated_liquidation_volume: float # Reconstructed volume
    anomaly_ratio: float                # abs(delta_oi) / taker_volume
    is_synthetic: bool                  # True if reconstructed, False if from official forceOrder stream


class SignalEvent(msgspec.Struct, gc=False):
    """
    Composite Quantitative Signal (-100 to +100) with breakdown of factors and risk parameters.
    """
    symbol: str
    timestamp_ms: int
    signal_type: str            # "STRONG_LONG", "STRONG_SHORT", "NEUTRAL"
    composite_score: float      # Score in range [-100.0, +100.0]
    z_cvd_div: float            # Z-Score of Price vs CVD divergence
    z_fund_trap: float          # Z-Score of funding trap detection
    z_delta_oi: float           # Z-Score of Open Interest accumulation/depletion
    z_micro: float              # Z-Score of Microstructure (OBI, VPIN, Spread)
    vpin: float                 # Volume-Synchronized Probability of Informed Trading
    obi: float                  # Order Book Imbalance [-1.0, 1.0]
    funding_8h: float           # 8h normalized funding rate
    basis_bps: float            # Perp vs Index basis spread in bps
    price: float                # Current reference price
    invalidation_price: float   # Hard stop / invalidation level
    target_price: float         # Model target price based on volatility / structure
    risk_reward_ratio: float    # Projected R:R ratio
    decision_timestamp_ms: int  # Point-in-time timestamp (must be < execution_timestamp_ms)
    z_whale_sentiment: float = 0.0 # Z-Score of Smart Money vs Retail positioning divergence
    relative_strength: float = 0.0 # Beta-adjusted Relative Strength vs BTC (%)
    sweep_reclaim: bool = False    # True if Wyckoff Spring / Upthrust liquidity sweep confirmed
    gate_status: str = "PASSED"    # "PASSED", "GATED_BTC_DUMP", "GATED_PRE_FUNDING", etc.


class MarketStateSnapshot(msgspec.Struct, gc=False):
    """
    State cache representation for 5-minute cron persistence.
    Allows zero-docker cron execution without losing historical delta context.
    """
    symbol: str
    timestamp_ms: int
    last_price: float
    open_interest: float
    delta_oi_5m: float
    cumulative_cvd_5m: float
    funding_rate_8h: float
    basis_bps: float
    vpin_estimate: float
    obi_score: float
    composite_score: float
    low_24h: float = 0.0       # 24h low — used as swing low reference for Wyckoff sweep detection
    high_24h: float = 0.0      # 24h high — used as swing high reference for Wyckoff sweep detection
    whale_sentiment_z: float = 0.0
    # FIX [C1]: History buffers for CVD divergence detection (need 4-12 points minimum)
    cvd_history_5m: tuple = ()  # Tuple of last N CVD values (5m snapshots)
    price_history_5m: tuple = ()  # Tuple of last N prices (5m snapshots)
