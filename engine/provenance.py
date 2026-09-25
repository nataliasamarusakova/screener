"""Runtime provenance for reproducible signal/research records."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping

SCHEMA_VERSION = 4
STRATEGY_REVISION = "2026-09-25-cleanstart-hardened-v4"

# Parameters which materially affect signal/risk semantics. Values come from the
# running process so the fingerprint is stable for identical configurations.
CONFIG_DEFAULTS = {
    "TOP_N_SYMBOLS": "100",
    "W_CVD": "25.0",
    "W_FUND": "25.0",
    "W_OI": "15.0",
    "W_MICRO": "15.0",
    "W_WHALE": "20.0",
    "FUNDING_WEIGHT": "0.70",
    "BASIS_WEIGHT": "0.30",
    "SCORE_TANH_SCALE": "1.40",
    "STRONG_SIGNAL_THRESHOLD": "75.0",
    "SWING_BUFFER_BPS": "15.0",
    "TARGET_RISK_REWARD": "2.0",
    "TRAILING_ACTIVATION_R": "0.5",
    "TRAILING_DISTANCE_R": "0.7",
    "SIGNAL_HISTORY_BARS": "60",
    "SIGNAL_Z_MIN_SAMPLES": "24",
    "CVD_LOOKBACK_BARS": "8",
    "BETA_MIN_SAMPLES": "24",
    "ATR_LOOKBACK_BARS": "12",
    "ATR_STOP_MULTIPLIER": "1.50",
    "MAX_ATR_MULTIPLIER": "2.50",
    "MAX_PORTFOLIO_CORRELATION": "0.80",
    "MAX_AGGREGATE_RISK_PCT": "0.03",
    "MAX_GROSS_LEVERAGE": "3.0",
    "MAX_STRONG_PER_DIRECTION": "3",
    "FUNDING_PROXIMITY_MINUTES": "20",
    "FUNDING_EXTREME_THRESHOLD_8H": "0.0010",
    "OBI_DECAY": "0.85",
    "MAX_DRAWDOWN_PCT": "0.20",
    "DAILY_LOSS_LIMIT_PCT": "0.08",
    "MAX_EQUITY_STATE_AGE_SEC": "600",
    "RISK_PER_TRADE_PCT": "0.01",
    "MAX_LEVERAGE": "3",
    "SWEEP_BOOSTER_POINTS": "0.0",
    "FRICTION_RT_PCT": "0.0018",
    "TAKER_COMMISSION_RT_PCT": "0.0010",
    "MIN_EFFECTIVE_RRR": "1.30",
    "VPIN_WINDOW_BASKETS": "10",
    "MIN_24H_VOLUME_USDT": "10000000",
    "MAX_SPREAD_BPS": "2.5",
    "SLIPPAGE_ATR_FRACTION": "0.15",
    "MAX_SLIPPAGE_RT_PCT": "0.0030",
    "ACCOUNT_EQUITY_USDT": "10000.0",
}


CONFIG_ENV_KEYS = tuple(CONFIG_DEFAULTS.keys())



def git_revision() -> str:
    return os.getenv("GITHUB_SHA") or os.getenv("GIT_COMMIT") or "UNKNOWN"


def config_fingerprint(env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    payload = {key: str(source.get(key, CONFIG_DEFAULTS[key])) for key in CONFIG_ENV_KEYS}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def provenance() -> dict[str, str | int]:
    return {
        "research_schema_version": SCHEMA_VERSION,
        "strategy_revision": STRATEGY_REVISION,
        "code_revision": git_revision(),
        "config_fingerprint": config_fingerprint(),
    }
