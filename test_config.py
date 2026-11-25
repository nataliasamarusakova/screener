from __future__ import annotations

from pathlib import Path

import pytest

from screener.config import load_settings


def _set_required_secrets(monkeypatch) -> None:
    monkeypatch.setenv("COINALYZE_URL", "https://coinalyze.net/?columns=x&filter=y&contract_type=perpetual")
    monkeypatch.setenv("COINALYZE_P_SID", "p-session")
    monkeypatch.setenv("COINALYZE_CHAT_SID", "chat-session")


def test_settings_read_credentials_only_from_process_environment(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("TG_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("TG_CHAT_IDS", "12345,67890")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "12345,67890")

    settings = load_settings(tmp_path)

    assert settings.coinalyze_url.startswith("https://coinalyze.net/")
    assert settings.coinalyze_p_sid == "p-session"
    assert settings.coinalyze_chat_sid == "chat-session"
    assert settings.telegram_bot_token == "telegram-token"
    assert settings.telegram_chat_ids == ["12345", "67890"]
    assert settings.data_dir == tmp_path / "data"


def test_settings_require_coinalyze_github_secrets(tmp_path: Path, monkeypatch):
    for key in ("COINALYZE_URL", "COINALYZE_P_SID", "COINALYZE_CHAT_SID"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="COINALYZE_URL"):
        load_settings(tmp_path)


def test_settings_require_proxy_when_wireguard_mode_is_enabled(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("BINANCE_VPN_ENABLED", "true")
    monkeypatch.delenv("BINANCE_HTTP_PROXY", raising=False)

    with pytest.raises(ValueError, match="BINANCE_HTTP_PROXY"):
        load_settings(tmp_path)


def test_settings_do_not_have_an_output_reports_directory(tmp_path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    settings = load_settings(tmp_path)
    assert not hasattr(settings, "output_dir")


def test_settings_read_optional_telegram_storage_hash_key(tmp_path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("TELEGRAM_STORAGE_HASH_KEY", "stable-hash-secret")

    settings = load_settings(tmp_path)

    assert settings.telegram_storage_hash_key == "stable-hash-secret"


def test_expensive_optional_binance_context_is_opt_in(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.delenv("BINANCE_FETCH_ORDERBOOK", raising=False)

    settings = load_settings(tmp_path)

    assert settings.binance_fetch_orderbook is False
    assert settings.p1["scanner"]["fetch_15m_context"] is False

    monkeypatch.setenv("BINANCE_FETCH_15M_CONTEXT", "true")
    settings = load_settings(tmp_path)
    assert settings.p1["scanner"]["fetch_15m_context"] is True

    monkeypatch.setenv("BINANCE_FETCH_ORDERBOOK", "true")
    settings = load_settings(tmp_path)
    assert settings.binance_fetch_orderbook is True


def test_live_telegram_configuration_warns_when_storage_hash_key_is_not_stable(tmp_path, monkeypatch, caplog):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "12345")
    monkeypatch.delenv("TELEGRAM_STORAGE_HASH_KEY", raising=False)

    with caplog.at_level("WARNING"):
        settings = load_settings(tmp_path)

    assert settings.telegram_storage_hash_key == ""
    assert "Rotating the bot token can orphan" in caplog.text


def test_coinalyze_timeout_and_retry_settings_are_configurable(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.delenv("COINALYZE_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("COINALYZE_PAGE_ATTEMPTS", raising=False)
    monkeypatch.delenv("COINALYZE_RETRY_BACKOFF_SEC", raising=False)

    defaults = load_settings(tmp_path)
    assert defaults.coinalyze_timeout_ms == 45_000
    assert defaults.coinalyze_page_attempts == 3
    assert defaults.coinalyze_retry_backoff_sec == 3.0

    monkeypatch.setenv("COINALYZE_TIMEOUT_MS", "60000")
    monkeypatch.setenv("COINALYZE_PAGE_ATTEMPTS", "4")
    monkeypatch.setenv("COINALYZE_RETRY_BACKOFF_SEC", "4.5")
    configured = load_settings(tmp_path)
    assert configured.coinalyze_timeout_ms == 60_000
    assert configured.coinalyze_page_attempts == 4
    assert configured.coinalyze_retry_backoff_sec == 4.5


def test_coinalyze_retry_settings_are_clamped_to_safe_ranges(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("COINALYZE_TIMEOUT_MS", "999999")
    monkeypatch.setenv("COINALYZE_PAGE_ATTEMPTS", "99")
    monkeypatch.setenv("COINALYZE_RETRY_BACKOFF_SEC", "999")

    settings = load_settings(tmp_path)
    assert settings.coinalyze_timeout_ms == 120_000
    assert settings.coinalyze_page_attempts == 5
    assert settings.coinalyze_retry_backoff_sec == 30.0


def test_shadow_fee_defaults_and_side_specific_overrides(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.delenv("TRADING_FEE_BPS_PER_SIDE", raising=False)
    for name in (
        "MAKER_FEE_BPS", "TAKER_FEE_BPS",
        "SHADOW_ENTRY_EXECUTION_TYPE", "SHADOW_EXIT_EXECUTION_TYPE", "SHADOW_NOTIONAL_USD",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = load_settings(tmp_path)
    assert defaults.maker_fee_bps == 2.0
    assert defaults.taker_fee_bps == 5.0
    assert defaults.shadow_entry_execution_type == "TAKER"
    assert defaults.shadow_exit_execution_type == "TAKER"
    assert defaults.shadow_notional_usd == 1_000.0

    monkeypatch.setenv("MAKER_FEE_BPS", "1.5")
    monkeypatch.setenv("TAKER_FEE_BPS", "4.5")
    monkeypatch.setenv("SHADOW_ENTRY_EXECUTION_TYPE", "maker")
    monkeypatch.setenv("SHADOW_EXIT_EXECUTION_TYPE", "taker")
    monkeypatch.setenv("SHADOW_NOTIONAL_USD", "2500")
    configured = load_settings(tmp_path)
    assert configured.maker_fee_bps == 1.5
    assert configured.taker_fee_bps == 4.5
    assert configured.shadow_entry_execution_type == "MAKER"
    assert configured.shadow_exit_execution_type == "TAKER"
    assert configured.shadow_notional_usd == 2_500.0


def test_obsolete_single_fee_setting_is_rejected(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("TRADING_FEE_BPS_PER_SIDE", "5")

    with pytest.raises(ValueError, match="TRADING_FEE_BPS_PER_SIDE is no longer supported"):
        load_settings(tmp_path)


def test_blank_numeric_environment_values_use_clean_defaults(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("SHADOW_NOTIONAL_USD", "")
    monkeypatch.setenv("MAKER_FEE_BPS", "")
    monkeypatch.setenv("TAKER_FEE_BPS", "")
    monkeypatch.setenv("COINALYZE_MAX_PAGES", "")
    monkeypatch.setenv("COINALYZE_TIMEOUT_MS", "")
    monkeypatch.setenv("COINALYZE_PAGE_ATTEMPTS", "")
    monkeypatch.setenv("COINALYZE_RETRY_BACKOFF_SEC", "")
    monkeypatch.setenv("BINANCE_KLINE_LIMIT", "")
    monkeypatch.setenv("BINANCE_REQUEST_INTERVAL_SEC", "")
    monkeypatch.setenv("BINANCE_ORDERBOOK_LIMIT", "")
    monkeypatch.setenv("EXECUTION_NOTIONAL_USD", "")

    settings = load_settings(tmp_path)

    assert settings.shadow_notional_usd == 1_000.0
    assert settings.maker_fee_bps == 2.0
    assert settings.taker_fee_bps == 5.0
    assert settings.coinalyze_max_pages == 20
    assert settings.coinalyze_timeout_ms == 45_000
    assert settings.coinalyze_page_attempts == 3
    assert settings.coinalyze_retry_backoff_sec == 3.0
    assert settings.binance_kline_limit == 300
    assert settings.binance_request_interval_sec == 0.12
    assert settings.binance_orderbook_limit == 100
    assert settings.execution_notional_usd == 0.0
    assert not hasattr(settings, "trading_fee_bps_per_side")


def test_shadow_fee_configuration_rejects_invalid_execution_type_and_notional(tmp_path: Path, monkeypatch):
    _set_required_secrets(monkeypatch)
    monkeypatch.setenv("SHADOW_ENTRY_EXECUTION_TYPE", "UNKNOWN")
    with pytest.raises(ValueError, match="SHADOW_ENTRY_EXECUTION_TYPE"):
        load_settings(tmp_path)

    monkeypatch.setenv("SHADOW_ENTRY_EXECUTION_TYPE", "TAKER")
    monkeypatch.setenv("SHADOW_NOTIONAL_USD", "0")
    with pytest.raises(ValueError, match="SHADOW_NOTIONAL_USD"):
        load_settings(tmp_path)
