from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


# P1 thresholds are kept with runtime code so GitHub Actions needs no external config file.
DEFAULT_P1_CONFIG: dict[str, Any] = {
    "universe": {
        "expected_contract_type": "PERPETUAL",
        "allow_unknown_contract_type_for_alerts": False,
    },
    "thresholds": {
        "strong_flow_quantile": 0.15,
        "strong_flow_min_rows": 20,
        "oi_4h_alt_pct": 1.0,
        "oi_4h_major_pct": 0.3,
        "oi_4h_small_usd": 2_000_000.0,
        "funding_hot_pct": 0.03,
        "funding_negative_pct": -0.005,
        "ls_long_extreme": 2.5,
        "ls_short_extreme": 0.9,
        "quality_vol_mktcap_max": 2.0,
        "quality_oi_vol_min": 0.1,
        "quality_oi_min_usd": 40_000_000.0,
        # Initial guardrails, not statistically calibrated thresholds. Require
        # either meaningful USD size or a material share of open interest.
        "liquidation_min_side_usd": 500_000.0,
        "liquidation_min_side_oi_ratio": 0.005,
        # Guardrail only: do not treat liquidation coverage as causal on tiny OI changes.
        "oi_24h_min_drop_usd_for_liq_coverage": 1_000_000.0,
        "oi_24h_min_drop_pct_for_short_d": -1.0,
    },
    "scenario": {
        # P1 does not quantify "significant" short liquidations; keep trigger UNKNOWN.
        "short_squeeze_min_short_liqs_usd": 0.0,
        "spot_accumulation_required_snapshots": 3,
        "snapshot_interval_hours": 4.0,
        # SHORT E keeps its original strict spacing tolerance.
        "snapshot_interval_tolerance_minutes": 10,
        # LONG B spot-accumulation snapshots can arrive late on Actions;
        # this tolerance is one-sided and never allows early/overlapping samples.
        "spot_accumulation_snapshot_tolerance_minutes": 30,
    },
    "structure": {
        "swing_left_bars": 2,
        "swing_right_bars": 2,
        "pivot_lookback_bars": 60,
        # A higher-low trigger must use a swing formed within the last 24 closed 1H bars.
        "higher_low_max_age_bars": 24,
        # Initial ATR guardrails from the audit; verify against forward results
        # before tightening or relaxing these defaults.
        "higher_low_min_rise_atr": 0.25,
        "higher_low_max_extension_atr": 2.0,
        "retest_lookback_bars": 8,
        "retest_atr_tolerance": 0.25,
        "retest_max_extension_atr": 2.0,
        "weak_rejection_wick_body_ratio": 0.75,
        "weak_rejection_atr_tolerance": 0.5,
        # A failed breakout is a short-lived event, not a level that can remain
        # valid after the market has accepted price above resistance.
        "failed_breakout_max_age_bars": 3,
    },
    "scanner": {
        "max_chart_candidates": 100,
        "fetch_15m_context": False,
        "market_context_max_candidates": 20,
        "market_context_oi_limit": 500,
        "market_context_funding_limit": 200,
        "market_context_kline_limit": 300,
    },
}


@dataclass(slots=True)
class Settings:
    coinalyze_url: str
    coinalyze_p_sid: str
    coinalyze_chat_sid: str
    coinalyze_max_pages: int
    coinalyze_headless: bool
    coinalyze_timeout_ms: int
    binance_base_url: str
    binance_http_proxy: str
    binance_vpn_enabled: bool
    binance_kline_limit: int
    binance_request_interval_sec: float
    binance_fetch_orderbook: bool
    binance_orderbook_limit: int
    execution_notional_usd: float
    telegram_bot_token: str
    telegram_chat_ids: list[str]
    data_dir: Path
    log_level: str
    dry_run: bool
    p1: dict[str, Any]
    telegram_storage_hash_key: str = ""
    coinalyze_page_attempts: int = 3
    coinalyze_retry_backoff_sec: float = 3.0
    # Explicit shadow execution assumptions. Rates are per side and applied to
    # traded notional, never margin. The $1,000 default is a research reference,
    # not an order size or an estimate of the user's future position size.
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    shadow_entry_execution_type: str = "TAKER"
    shadow_exit_execution_type: str = "TAKER"
    shadow_notional_usd: float = 1_000.0


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _text_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer") from exc


def load_settings(root: Path = ROOT) -> Settings:
    """Load runtime settings exclusively from GitHub Actions environment variables."""
    strategy = DEFAULT_P1_CONFIG

    chat_ids = [
        item.strip()
        for item in os.getenv("TELEGRAM_CHAT_IDS", "").replace(";", ",").split(",")
        if item.strip()
    ]
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    obsolete_fee_value = os.getenv("TRADING_FEE_BPS_PER_SIDE", "").strip()
    if obsolete_fee_value:
        raise ValueError(
            "TRADING_FEE_BPS_PER_SIDE is no longer supported. Remove this obsolete Actions variable; "
            "configure MAKER_FEE_BPS and TAKER_FEE_BPS instead."
        )

    def fee_setting(env_name: str, default: float) -> float:
        value = _float_env(env_name, default)
        if value < 0:
            raise ValueError(f"{env_name} must be a finite non-negative number of basis points")
        return value

    entry_execution_type = _text_env("SHADOW_ENTRY_EXECUTION_TYPE", "TAKER").upper()
    exit_execution_type = _text_env("SHADOW_EXIT_EXECUTION_TYPE", "TAKER").upper()
    if entry_execution_type not in {"MAKER", "TAKER"}:
        raise ValueError("SHADOW_ENTRY_EXECUTION_TYPE must be MAKER or TAKER")
    if exit_execution_type not in {"MAKER", "TAKER"}:
        raise ValueError("SHADOW_EXIT_EXECUTION_TYPE must be MAKER or TAKER")
    shadow_notional = _float_env("SHADOW_NOTIONAL_USD", 1_000.0)
    if not math.isfinite(shadow_notional) or shadow_notional <= 0:
        raise ValueError("SHADOW_NOTIONAL_USD must be a finite positive number")

    settings = Settings(
        coinalyze_url=os.getenv("COINALYZE_URL", "").strip(),
        coinalyze_p_sid=os.getenv("COINALYZE_P_SID", "").strip(),
        coinalyze_chat_sid=os.getenv("COINALYZE_CHAT_SID", "").strip(),
        coinalyze_max_pages=max(1, min(100, _int_env("COINALYZE_MAX_PAGES", 20))),
        coinalyze_headless=_bool_env("COINALYZE_HEADLESS", True),
        coinalyze_timeout_ms=max(5000, min(120_000, _int_env("COINALYZE_TIMEOUT_MS", 45_000))),
        binance_base_url=_text_env("BINANCE_BASE_URL", "https://fapi.binance.com").rstrip("/"),
        binance_http_proxy=os.getenv("BINANCE_HTTP_PROXY", "").strip(),
        binance_vpn_enabled=_bool_env("BINANCE_VPN_ENABLED", False),
        binance_kline_limit=max(100, min(1500, _int_env("BINANCE_KLINE_LIMIT", 300))),
        binance_request_interval_sec=max(0.0, _float_env("BINANCE_REQUEST_INTERVAL_SEC", 0.12)),
        binance_fetch_orderbook=_bool_env("BINANCE_FETCH_ORDERBOOK", False),
        binance_orderbook_limit=max(5, min(1000, _int_env("BINANCE_ORDERBOOK_LIMIT", 100))),
        execution_notional_usd=max(0.0, _float_env("EXECUTION_NOTIONAL_USD", 0.0)),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_ids=chat_ids,
        data_dir=data_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        dry_run=_bool_env("DRY_RUN", True),
        p1={section: values.copy() for section, values in strategy.items()},
        telegram_storage_hash_key=os.getenv("TELEGRAM_STORAGE_HASH_KEY", "").strip(),
        coinalyze_page_attempts=max(1, min(5, _int_env("COINALYZE_PAGE_ATTEMPTS", 3))),
        coinalyze_retry_backoff_sec=max(
            0.0, min(30.0, _float_env("COINALYZE_RETRY_BACKOFF_SEC", 3.0))
        ),
        maker_fee_bps=fee_setting("MAKER_FEE_BPS", 2.0),
        taker_fee_bps=fee_setting("TAKER_FEE_BPS", 5.0),
        shadow_entry_execution_type=entry_execution_type,
        shadow_exit_execution_type=exit_execution_type,
        shadow_notional_usd=shadow_notional,
    )

    settings.p1.setdefault("scanner", {})["fetch_15m_context"] = _bool_env(
        "BINANCE_FETCH_15M_CONTEXT", False
    )
    if settings.telegram_bot_token and settings.telegram_chat_ids and not settings.telegram_storage_hash_key:
        log.warning(
            "TELEGRAM_STORAGE_HASH_KEY is unset; recipient identity falls back to TELEGRAM_BOT_TOKEN. "
            "Rotating the bot token can orphan persisted active-notification state; configure a stable dedicated key."
        )
    if not settings.coinalyze_url:
        raise ValueError("Missing GitHub Secret COINALYZE_URL")
    if not settings.coinalyze_p_sid or not settings.coinalyze_chat_sid:
        raise ValueError("Missing GitHub Secrets COINALYZE_P_SID and/or COINALYZE_CHAT_SID")
    if settings.binance_vpn_enabled and not settings.binance_http_proxy:
        raise ValueError("BINANCE_VPN_ENABLED=true requires BINANCE_HTTP_PROXY")
    if settings.binance_http_proxy and not settings.binance_http_proxy.startswith(("http://", "https://")):
        raise ValueError("BINANCE_HTTP_PROXY must use http:// or https://")
    return settings
