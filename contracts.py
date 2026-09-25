"""
Core typed data contracts for Quantitative Crypto Derivatives Platform.
Uses msgspec.Struct(gc=False) for bounded object-model overhead.
"""
from __future__ import annotations
from typing import Optional
import msgspec


class NormalizedTrade(msgspec.Struct, gc=False):
    """Normalized taker trade event from crypto derivative exchanges."""
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
    """Synchronized L2 OrderBook snapshot with microstructure metrics."""
    symbol: str
    last_update_id: int
    timestamp_ms: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_bps: float
    obi_depth5: float
    obi_depth10: float


class NormalizedFunding(msgspec.Struct, gc=False):
    """Normalized funding rate brought to a unified 8-hour basis."""
    symbol: str
    raw_rate: float
    interval_hours: float
    normalized_8h_rate: float
    annualized_rate: float
    mark_price: float
    index_price: float
    basis_spread: float
    basis_spread_bps: float
    next_funding_time_ms: int
    timestamp_ms: int


class SyntheticLiquidation(msgspec.Struct, gc=False):
    """Reconstructed synthetic liquidation event."""
    symbol: str
    timestamp_ms: int
    price: float
    delta_oi: float
    taker_volume: float
    side: str
    estimated_liquidation_volume: float
    anomaly_ratio: float
    is_synthetic: bool


class SignalEvent(msgspec.Struct, gc=False):
    """Composite Quantitative Signal (-100 to +100) with breakdown of factors and risk parameters."""
    symbol: str
    timestamp_ms: int          # completed 5m candle close timestamp
    signal_type: str            # "STRONG_LONG", "STRONG_SHORT", "NEUTRAL"
    composite_score: float
    z_cvd_div: float
    z_fund_trap: float
    z_delta_oi: float
    z_micro: float
    vpin: float
    obi: float
    funding_8h: float
    basis_bps: float
    price: float
    invalidation_price: float
    target_price: float
    risk_reward_ratio: float    # Net R:R (friction-adjusted). 0.0 for NEUTRAL.
    decision_timestamp_ms: int # wall-clock signal generation timestamp
    z_whale_sentiment: float = 0.0
    relative_strength: float = 0.0
    sweep_reclaim: bool = False
    gate_status: str = "PASSED"
    sweep_pattern: str = "NONE"
    suggested_position_usd: float = 0.0
    suggested_leverage: int = 1
    # NEW: Trailing stop parameters (activates when price reaches activation threshold)
    trailing_stop_activation_pct: float = 0.0   # In fraction of price (e.g. 0.005 = 0.5%)
    trailing_stop_distance_pct: float = 0.0     # Distance from peak (e.g. 0.003 = 0.3%)
    # NEW: Friction model that was actually applied
    applied_friction_rt_pct: float = 0.0


class MarketStateSnapshot(msgspec.Struct, gc=False):
    """State cache representation for 5-minute cron persistence."""
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
    low_24h: float = 0.0
    high_24h: float = 0.0
    whale_sentiment_z: float = 0.0
    cvd_history_5m: tuple = ()
    price_history_5m: tuple = ()
    funding_history_5m: tuple = ()
    basis_history_5m: tuple = ()
    delta_oi_pct_history_5m: tuple = ()
    micro_factor_history_5m: tuple = ()
    whale_divergence_history_5m: tuple = ()
    cvd_divergence_history_5m: tuple = ()
    candle_open_time_ms: int = 0
    candle_open_times_5m: tuple = ()
    candle_high_5m: float = 0.0
    candle_low_5m: float = 0.0
    signal_ready: bool = False
    strategy_revision: str = "LEGACY_UNKNOWN"
    code_revision: str = "UNKNOWN"
    config_fingerprint: str = "UNKNOWN"
    research_schema_version: int = 0
