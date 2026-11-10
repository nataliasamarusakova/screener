from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


# P1 thresholds are kept with runtime code so GitHub Actions needs no external config file.
DEFAULT_P1_CONFIG: dict[str, Any] = {
    "universe": {
        "expected_contract_type": "PERPETUAL",
        "allow_unknown_contract_type_for_alerts": False,
    },
    "thresholds": {
        "strong_flow_quantile": 0.15,
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
    },
    "scenario": {
        # P1 does not quantify "significant" short liquidations; keep trigger UNKNOWN.
        "short_squeeze_min_short_liqs_usd": 0.0,
        "spot_accumulation_required_snapshots": 3,
        "snapshot_interval_hours": 4.0,
        "snapshot_interval_tolerance_minutes": 10,
    },
    "structure": {
        "swing_left_bars": 2,
        "swing_right_bars": 2,
        "pivot_lookback_bars": 60,
        "retest_lookback_bars": 8,
        "retest_atr_tolerance": 0.25,
        "weak_rejection_wick_body_ratio": 0.75,
        "weak_rejection_atr_tolerance": 0.5,
    },
    "scanner": {
        "max_chart_candidates": 100,
        "fetch_15m_context": True,
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
    trading_fee_bps_per_side: float | None
    telegram_bot_token: str
    telegram_chat_ids: list[str]
    telegram_cooldown_hours: float
    telegram_send_wait_states: bool
    data_dir: Path
    log_level: str
    dry_run: bool
    p1: dict[str, Any]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    fee_value = os.getenv("TRADING_FEE_BPS_PER_SIDE", "").strip()
    settings = Settings(
        coinalyze_url=os.getenv("COINALYZE_URL", "").strip(),
        coinalyze_p_sid=os.getenv("COINALYZE_P_SID", "").strip(),
        coinalyze_chat_sid=os.getenv("COINALYZE_CHAT_SID", "").strip(),
        coinalyze_max_pages=max(1, min(100, int(os.getenv("COINALYZE_MAX_PAGES", "20")))),
        coinalyze_headless=_bool_env("COINALYZE_HEADLESS", True),
        coinalyze_timeout_ms=max(5000, int(os.getenv("COINALYZE_TIMEOUT_MS", "30000"))),
        binance_base_url=os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com").rstrip("/"),
        binance_http_proxy=os.getenv("BINANCE_HTTP_PROXY", "").strip(),
        binance_vpn_enabled=_bool_env("BINANCE_VPN_ENABLED", False),
        binance_kline_limit=max(100, min(1500, int(os.getenv("BINANCE_KLINE_LIMIT", "300")))),
        binance_request_interval_sec=max(0.0, float(os.getenv("BINANCE_REQUEST_INTERVAL_SEC", "0.12"))),
        binance_fetch_orderbook=_bool_env("BINANCE_FETCH_ORDERBOOK", True),
        binance_orderbook_limit=max(5, min(1000, int(os.getenv("BINANCE_ORDERBOOK_LIMIT", "100")))),
        execution_notional_usd=max(0.0, float(os.getenv("EXECUTION_NOTIONAL_USD", "0"))),
        trading_fee_bps_per_side=max(0.0, float(fee_value)) if fee_value else None,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_ids=chat_ids,
        telegram_cooldown_hours=max(0.0, float(os.getenv("TELEGRAM_COOLDOWN_HOURS", "4"))),
        telegram_send_wait_states=_bool_env("TELEGRAM_SEND_WAIT_STATES", True),
        data_dir=data_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        dry_run=_bool_env("DRY_RUN", True),
        p1={section: values.copy() for section, values in strategy.items()},
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
