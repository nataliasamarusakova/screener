from types import SimpleNamespace
from datetime import timedelta

from screener.metrics import analyze_universe
from screener.models import Check, ScenarioResult, MarketRow
from screener.service import _collect_notifications


class FakeStore:
    def __init__(self):
        self.exists_calls = []
        self.due_calls = []

    def notification_exists(self, key, chat_id=None):
        self.exists_calls.append((key, chat_id))
        return True

    def notification_due(self, key, cooldown, chat_id=None):
        self.due_calls.append((key, cooldown, chat_id))
        return True



def test_unchanged_table_suppresses_duplicate_notification_without_name_error():
    market = MarketRow(
        symbol="ETHUSDT_PERP", name="Ethereum", price=2000, chg24_pct=2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=1, oi_chg24_usd=1_000_000, oi_chg4h_pct=2, oi_chg4h_usd=3_000_000,
        fr_avg_pct=0.001, fr_oiw_pct=0.002, pfr_oiw_pct=0.003,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=1.2, futures_imbalance_4h=10, spot_imbalance_4h=10,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A", direction="LONG",
        checks=[Check("complete", True, "yes", "yes")], quality_flags=[], status="OBSERVATION",
    )
    settings = SimpleNamespace(
        telegram_bot_token="", telegram_chat_ids=[], telegram_send_wait_states=True,
        telegram_cooldown_hours=4,
    )
    store = FakeStore()

    notifications = _collect_notifications(
        [result], enriched, settings, store,
        contract_alerts_allowed=True, dry_run=True, no_new_data=True,
    )

    assert notifications == []
    assert store.exists_calls == [("ETHUSDT_PERP|LONG A|OBSERVATION", None)]
    assert store.due_calls == []


def test_telegram_alert_matches_compact_observation_format_and_keeps_disclaimer():
    from screener.telegram import format_alert

    market = MarketRow(
        symbol="AEROUSDT_PERP.A", name="AERO", price=0.42, chg24_pct=1.69,
        oi_chg4h_pct=3.92, oi_chg4h_usd=1_200_000,
        fr_oiw_pct=-0.0104, ls_ratio_1d=1.20,
        futures_imbalance_4h=17.4027, spot_imbalance_4h=22.4889,
        btc_corr7d=0.82,
        market_cap_usd=500_000_000, volume24_usd=50_000_000, oi_usd=100_000_000,
        oi_vol_ratio=2.0, oi_mktcap_ratio=0.2, vol_mktcap_ratio=0.1,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol,
        name=market.name,
        scenario="LONG A — Trend Continuation",
        direction="LONG",
        checks=[Check(f"check-{i}", True, "pass", "rule") for i in range(6)],
        quality_flags=[],
        status="OBSERVATION",
        technical={
            "context_4h": {
                "ema20": 0.39, "ema50": 0.41,
                "last_close": 0.38, "last_confirmed_support": 0.40,
            },
            "higher_low": {"confirmed": False, "reason": "Latest confirmed swing low is not higher"},
            "trend_retest_long": {"confirmed": False, "reason": "No 1H close + subsequent retest found"},
        },
    )

    message = format_alert(result, enriched[0])

    assert "AERO — LONG A" in message
    assert "OBSERVATION · 6/6" in message
    assert "F = <code>+17,4027</code> · S = <code>+22,4889</code>" in message
    assert "OI 4H: <code>+3,92%</code>" in message
    assert "Цена 24H: <code>+1,69%</code>" in message
    assert "Funding OI_W: <code>-0,0104%</code>" in message
    assert "L/S 1D: <code>1,20</code>" in message
    assert "BTC Corr. 7D: <code>+0,82</code>" in message
    assert "4H EMA20 ниже EMA50." in message
    assert "Последнее закрытие 4H ниже подтверждённой поддержки." in message
    assert "Higher Low отсутствует." in message
    assert "LONG retest не подтверждён." in message
    assert "графическая структура против LONG. Не входить." in message
    assert "check-0" not in message
    assert "правило:" not in message
    assert message.endswith("<i>Наблюдение по P1, не торговый сигнал и не команда на вход. Вход требует проверки графика и риска.</i>")


def test_telegram_message_does_not_expand_internal_check_dump():
    from screener.telegram import format_alert

    market = MarketRow(symbol="BTCUSDT_PERP", name="Bitcoin", price=70000, chg24_pct=1)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol,
        name=market.name,
        scenario="LONG A — Trend Continuation",
        direction="LONG",
        checks=[Check("условие " + ("x" * 100), None, "значение " + ("y" * 100), "правило") for _ in range(80)],
        quality_flags=[],
        status="OBSERVATION",
    )

    message = format_alert(result, enriched[0])

    assert len(message) < 1500
    assert "условие " not in message
    assert message.endswith("<i>Наблюдение по P1, не торговый сигнал и не команда на вход. Вход требует проверки графика и риска.</i>")


def test_scan_pipeline_dry_run_and_unchanged_snapshot_suppression(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import screener.service as service
    from screener.coinalyze import ScrapeResult
    from screener.config import Settings
    from screener.storage import Store

    markets = []
    for index in range(20):
        flow = 50.0 if index == 0 else 0.5
        markets.append(MarketRow(
            symbol=f"ALT{index}USDT_PERP.A", name=f"ALT{index}", price=1.0 + index,
            chg24_pct=2.0, market_cap_usd=1_000_000_000, volume24_usd=100_000_000,
            oi_usd=100_000_000, oi_chg24_pct=1.0, oi_chg24_usd=1_000_000,
            oi_chg4h_pct=2.0, oi_chg4h_usd=3_000_000, oi_vol_ratio=1.0,
            oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.1, fr_avg_pct=0.01,
            pfr_avg_pct=0.01, fr_oiw_pct=0.01, pfr_oiw_pct=0.01,
            liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
            ls_ratio_1h=1.2, ls_ratio_1d=1.2,
            futures_imbalance_4h=flow, spot_imbalance_4h=flow,
        ))

    from screener.config import DEFAULT_P1_CONFIG
    p1 = {section: values.copy() for section, values in DEFAULT_P1_CONFIG.items()}
    settings = Settings(
        coinalyze_url="https://coinalyze.net/", coinalyze_p_sid="", coinalyze_chat_sid="",
        coinalyze_max_pages=20, coinalyze_headless=True, coinalyze_timeout_ms=30_000,
        binance_base_url="https://fapi.binance.com", binance_http_proxy="",
        binance_vpn_enabled=False, binance_kline_limit=300,
        binance_request_interval_sec=0.0, binance_fetch_orderbook=False,
        binance_orderbook_limit=100, execution_notional_usd=0.0,
        trading_fee_bps_per_side=None, telegram_bot_token="", telegram_chat_ids=[],
        telegram_cooldown_hours=4.0, telegram_send_wait_states=True,
        data_dir=tmp_path / "data",
        log_level="INFO", dry_run=True, p1=p1,
    )
    fixed_time = datetime.now(timezone.utc)
    scraped = ScrapeResult(
        rows=markets, fetched_at=fixed_time, contract_type="PERPETUAL", pages_loaded=1,
        headers_found=[], missing_columns=[], source_url=settings.coinalyze_url,
    )
    monkeypatch.setattr(service, "fetch_table", lambda **_kwargs: scraped)
    monkeypatch.setattr(service, "_technical_for_symbol", lambda *_args: {
        "trend_retest_long": {"confirmed": True, "reason": "test fixture: closed 1H retest"},
        "higher_low": {"confirmed": False, "reason": "not applicable"},
        "weak_bounce_rejection": {"confirmed": False},
        "execution_cost_assumptions": {"taker_fee_bps_per_side": None},
    })

    first = service.run_scan_once(settings, dry_run=True)
    assert first["status"] == "OK"
    assert first["rows"] == 20
    assert any(item.status == "OBSERVATION" for item in first["results"])
    assert "output_paths" not in first
    assert not (tmp_path / "reports").exists()
    assert (settings.data_dir / "snapshots.jsonl").exists()
    assert (settings.data_dir / "market_rows.jsonl").exists()
    assert (settings.data_dir / "runs.jsonl").exists()

    store = Store(settings.data_dir)
    store.record_notification("ALT0USDT_PERP.A|LONG A — Trend Continuation|OBSERVATION", "fixture")
    scraped.fetched_at = fixed_time + timedelta(minutes=30)
    second = service.run_scan_once(settings, dry_run=True)
    assert second["no_new_coinalyze_data"] is True
    assert second["notifications"] == []
    # Polling an identical table must not move the observation timestamp forward.
    latest_history = store.recent_symbol_history("ALT0USDT_PERP.A", limit=5)
    assert latest_history[-1]["snapshot_at"] == fixed_time.isoformat()


def test_partial_telegram_delivery_retries_only_recipients_without_success(tmp_path, monkeypatch):
    import screener.service as service
    from screener.storage import Store

    market = MarketRow(
        symbol="ETHUSDT_PERP", name="Ethereum", price=2000, chg24_pct=2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=1, oi_chg24_usd=1_000_000, oi_chg4h_pct=2, oi_chg4h_usd=3_000_000,
        fr_avg_pct=0.001, fr_oiw_pct=0.002, pfr_oiw_pct=0.003,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=1.2, futures_imbalance_4h=10, spot_imbalance_4h=10,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A", direction="LONG",
        checks=[Check("complete", True, "yes", "yes")], quality_flags=[], status="OBSERVATION",
    )
    settings = SimpleNamespace(
        telegram_bot_token="token", telegram_chat_ids=["chat-a", "chat-b"],
        telegram_send_wait_states=True, telegram_cooldown_hours=4,
    )
    store = Store(tmp_path / "data")
    calls = []

    class PartialTelegram:
        def __init__(self, token, chat_ids):
            self.chat_ids = chat_ids

        @property
        def configured(self):
            return True

        def send(self, text, chat_ids=None):
            recipients = list(self.chat_ids if chat_ids is None else chat_ids)
            calls.append(recipients)
            return [
                {
                    "chat_id": chat_id,
                    "sent": not (chat_id == "chat-b" and len(calls) == 1),
                    "message_id": f"{chat_id}-message",
                }
                for chat_id in recipients
            ]

    monkeypatch.setattr(service, "TelegramClient", PartialTelegram)
    key = "ETHUSDT_PERP|LONG A|OBSERVATION"

    first = _collect_notifications(
        [result], enriched, settings, store,
        contract_alerts_allowed=True, dry_run=False, no_new_data=False,
    )
    assert first[0]["sent"] is False
    assert calls == [["chat-a", "chat-b"]]
    assert store.notification_exists(key, "chat-a") is True
    assert store.notification_exists(key, "chat-b") is False

    second = _collect_notifications(
        [result], enriched, settings, store,
        contract_alerts_allowed=True, dry_run=False, no_new_data=True,
    )
    assert second[0]["sent"] is True
    assert calls == [["chat-a", "chat-b"], ["chat-b"]]
    assert store.notification_exists(key, "chat-b") is True

    third = _collect_notifications(
        [result], enriched, settings, store,
        contract_alerts_allowed=True, dry_run=False, no_new_data=True,
    )
    assert third == []
    assert calls == [["chat-a", "chat-b"], ["chat-b"]]
