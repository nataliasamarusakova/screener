from types import SimpleNamespace
from datetime import timedelta

from screener.metrics import analyze_universe
from screener.models import Check, ScenarioResult, MarketRow
from screener.service import _collect_notifications


class FakeStore:
    def __init__(self):
        self.exists_calls = []
        self.due_calls = []

    def notification_exists(self, key):
        self.exists_calls.append(key)
        return True

    def notification_due(self, key, cooldown):
        self.due_calls.append((key, cooldown))
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
    assert store.exists_calls == ["ETHUSDT_PERP|LONG A|OBSERVATION"]
    assert store.due_calls == []


def test_telegram_alert_truncation_keeps_disclaimer():
    from screener.telegram import format_alert

    market = MarketRow(symbol="BTCUSDT_PERP", name="Bitcoin", price=70000, chg24_pct=1)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol,
        name=market.name,
        scenario="LONG A",
        direction="LONG",
        checks=[Check("условие " + ("x" * 100), None, "значение " + ("y" * 100), "правило") for _ in range(80)],
        quality_flags=[],
        status="OBSERVATION",
    )

    message = format_alert(result, enriched[0])

    assert len(message) <= 3900
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
        output_dir=tmp_path / "output", log_level="INFO", dry_run=True, p1=p1,
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
    assert all(path.exists() for path in first["output_paths"])

    store = Store(settings.data_dir)
    store.record_notification("ALT0USDT_PERP.A|LONG A — Trend Continuation|OBSERVATION", "fixture")
    scraped.fetched_at = fixed_time + timedelta(minutes=30)
    second = service.run_scan_once(settings, dry_run=True)
    assert second["no_new_coinalyze_data"] is True
    assert second["notifications"] == []
    # Polling an identical table must not move the observation timestamp forward.
    latest_history = store.recent_symbol_history("ALT0USDT_PERP.A", limit=5)
    assert latest_history[-1]["snapshot_at"] == fixed_time.isoformat()


def test_price_oi_snapshot_classification_marks_overlapping_windows_preliminary(tmp_path):
    from datetime import datetime, timedelta, timezone
    from screener.service import _snapshot_deltas
    from screener.storage import Store

    first_time = datetime.now(timezone.utc) - timedelta(hours=3)
    second_time = first_time + timedelta(hours=3)
    first = MarketRow(
        symbol="BTCUSDT_PERP", name="Bitcoin", price=100, oi_usd=100_000_000,
        futures_imbalance_4h=1, spot_imbalance_4h=2, fr_oiw_pct=0.001,
    )
    second = MarketRow(
        symbol="BTCUSDT_PERP", name="Bitcoin", price=101, oi_usd=102_000_000,
        futures_imbalance_4h=2, spot_imbalance_4h=3, fr_oiw_pct=0.002,
    )
    store = Store(tmp_path / "history-data")
    store.add_snapshot(first_time, "PERPETUAL", [first])
    store.add_snapshot(second_time, "PERPETUAL", [second])
    enriched, _ = analyze_universe([second], {"thresholds": {}})
    delta = _snapshot_deltas(store, enriched, second_time)["BTCUSDT_PERP"]
    assert delta["price_change_pct"] > 0
    assert delta["oi_change_pct"] > 0
    assert delta["price_x_oi_classification"].startswith("preliminary_overlapping_windows:")
