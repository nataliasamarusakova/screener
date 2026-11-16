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
