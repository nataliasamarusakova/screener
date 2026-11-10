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
