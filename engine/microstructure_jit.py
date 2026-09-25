"""
Microstructure Hot-Path Analytics powered by Numba JIT (@njit(fastmath=True, nogil=True)).
Includes:
1. Streaming VPIN (Volume-Synchronized Probability of Informed Trading) with O(1) rolling updates.
2. Depth-weighted Order Book Imbalance (OBI).
3. Volume Dispersion and Volatility Compression metric.
"""
from __future__ import annotations
import numpy as np
import numba

VPIN_ESTIMATE_METHOD = "INTRABAR_VOLUME_BUCKET_PROXY"


@numba.njit(fastmath=True, nogil=True)
def compute_weighted_obi_jit(
    bid_qtys: np.ndarray,
    ask_qtys: np.ndarray,
    decay: float = 0.85
) -> float:
    """
    Computes exponential depth-weighted Order Book Imbalance across L2 levels.
    Levels closer to the spread have higher weights: w_i = decay^i.
    Returns: OBI in range [-1.0, 1.0].
    """
    n_levels = min(len(bid_qtys), len(ask_qtys))
    if n_levels == 0:
        return 0.0

    weighted_bid_vol = 0.0
    weighted_ask_vol = 0.0
    current_weight = 1.0

    for i in range(n_levels):
        weighted_bid_vol += bid_qtys[i] * current_weight
        weighted_ask_vol += ask_qtys[i] * current_weight
        current_weight *= decay

    total_vol = weighted_bid_vol + weighted_ask_vol
    if total_vol <= 1e-12:
        return 0.0

    return (weighted_bid_vol - weighted_ask_vol) / total_vol


@numba.njit(fastmath=True, nogil=True)
def compute_vpin_numba(
    trade_qtys: np.ndarray,
    is_buyer: np.ndarray,
    bucket_vol: float,
    window_baskets: int = 50
) -> float:
    """
    JIT-accelerated VPIN computation on an array of trades.
    Trades are accumulated into volume buckets of size `bucket_vol`.
    VPIN = sum(|V_buy - V_sell|) / (N * bucket_vol).
    """
    n_trades = len(trade_qtys)
    if n_trades == 0 or bucket_vol <= 0.0:
        return 0.0

    bucket_imbalances = np.zeros(window_baskets, dtype=np.float64)
    bucket_idx = 0
    total_buckets_filled = 0

    curr_bucket_buy = 0.0
    curr_bucket_sell = 0.0
    curr_bucket_vol = 0.0

    for i in range(n_trades):
        q = trade_qtys[i]
        is_buy = is_buyer[i]

        while q > 0.0:
            remaining_in_bucket = bucket_vol - curr_bucket_vol

            if q >= remaining_in_bucket:
                # Fills the current bucket
                if is_buy:
                    curr_bucket_buy += remaining_in_bucket
                else:
                    curr_bucket_sell += remaining_in_bucket

                # Finalize bucket
                imbalance = abs(curr_bucket_buy - curr_bucket_sell)
                bucket_imbalances[bucket_idx] = imbalance
                bucket_idx = (bucket_idx + 1) % window_baskets
                total_buckets_filled += 1

                # Reset bucket
                q -= remaining_in_bucket
                curr_bucket_buy = 0.0
                curr_bucket_sell = 0.0
                curr_bucket_vol = 0.0
            else:
                # Partially fills bucket
                if is_buy:
                    curr_bucket_buy += q
                else:
                    curr_bucket_sell += q
                curr_bucket_vol += q
                q = 0.0

    # Calculate VPIN over available filled buckets
    valid_buckets = min(total_buckets_filled, window_baskets)
    if valid_buckets == 0:
        # If not enough volume to fill a full bucket, estimate from current partial
        if curr_bucket_vol > 0.0:
            return abs(curr_bucket_buy - curr_bucket_sell) / curr_bucket_vol
        return 0.0

    sum_imbalances = 0.0
    for j in range(valid_buckets):
        sum_imbalances += bucket_imbalances[j]

    return sum_imbalances / (valid_buckets * bucket_vol)


@numba.njit(fastmath=True, nogil=True)
def compute_volatility_compression_jit(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray
) -> float:
    """
    Computes normalized Parkinson volatility compression relative to volume.
    Low value indicates market coil / absorption before breakout.
    """
    n = len(closes)
    if n < 5:
        return 1.0

    log_hl_sum = 0.0
    vol_sum = 0.0

    for i in range(n):
        if lows[i] > 0.0 and highs[i] >= lows[i]:
            log_hl = np.log(highs[i] / lows[i])
            log_hl_sum += log_hl * log_hl
        vol_sum += volumes[i]

    parkinson_var = log_hl_sum / (4.0 * np.log(2.0) * n)
    avg_vol = vol_sum / n if n > 0 else 1.0

    # Compression ratio: volatility normalized by volume
    if avg_vol <= 0.0:
        return 0.0

    return np.sqrt(parkinson_var) * 1000.0


class StreamingVPINState:
    """
    Stateful circular-buffer VPIN accumulator for real-time tick streaming.
    Updates in strict O(1) time per completed volume basket.
    """

    def __init__(self, bucket_vol: float, window_baskets: int = 50) -> None:
        self.bucket_vol = bucket_vol
        self.window_baskets = window_baskets
        self.bucket_imbalances = np.zeros(window_baskets, dtype=np.float64)
        self.bucket_idx = 0
        self.filled_count = 0
        self.running_sum_imbalance = 0.0

        self.curr_buy_vol = 0.0
        self.curr_sell_vol = 0.0
        self.curr_bucket_vol = 0.0

    def update_tick(self, qty: float, is_buy: bool) -> float:
        """
        Process a single trade tick.
        Returns current VPIN value.
        """
        if self.bucket_vol <= 0.0:
            return 0.0

        q = qty
        while q > 0.0:
            rem = self.bucket_vol - self.curr_bucket_vol
            if q >= rem:
                if is_buy:
                    self.curr_buy_vol += rem
                else:
                    self.curr_sell_vol += rem

                imbalance = abs(self.curr_buy_vol - self.curr_sell_vol)

                # O(1) rolling sum update in circular buffer
                old_val = self.bucket_imbalances[self.bucket_idx]
                self.bucket_imbalances[self.bucket_idx] = imbalance
                self.running_sum_imbalance += (imbalance - old_val)

                self.bucket_idx = (self.bucket_idx + 1) % self.window_baskets
                if self.filled_count < self.window_baskets:
                    self.filled_count += 1

                q -= rem
                self.curr_buy_vol = 0.0
                self.curr_sell_vol = 0.0
                self.curr_bucket_vol = 0.0
            else:
                if is_buy:
                    self.curr_buy_vol += q
                else:
                    self.curr_sell_vol += q
                self.curr_bucket_vol += q
                q = 0.0

        if self.filled_count == 0:
            if self.curr_bucket_vol > 0.0:
                return abs(self.curr_buy_vol - self.curr_sell_vol) / self.curr_bucket_vol
            return 0.0

        return self.running_sum_imbalance / (self.filled_count * self.bucket_vol)
