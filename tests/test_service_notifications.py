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



def test_waiting_and_observation_scenarios_are_not_delivered():
    market = MarketRow(
        symbol="ETHUSDT_PERP", name="Ethereum", price=2000, chg24_pct=2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=1, oi_chg24_usd=1_000_000, oi_chg4h_pct=2, oi_chg4h_usd=3_000_000,
        fr_avg_pct=0.001, fr_oiw_pct=0.002, pfr_oiw_pct=0.003,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=1.2, futures_imbalance_4h=10, spot_imbalance_4h=10,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    settings = SimpleNamespace(telegram_bot_token="", telegram_chat_ids=[])
    store = FakeStore()
    for status in ("OBSERVATION", "WAIT_BOUNCE", "PREPARATION", "TRIGGER", "NOT_MET"):
        result = ScenarioResult(
            symbol=market.symbol, name=market.name, scenario="LONG A", direction="LONG",
            checks=[Check("complete", True, "yes", "yes")], quality_flags=[], status=status,
        )
        assert _collect_notifications([result], enriched, settings, store, True, False) == []


def test_telegram_header_uses_direction_emoji_and_full_contract_ticker_for_short():
    from screener.telegram import format_alert

    market = MarketRow(symbol="INITUSDT_PERP.A", name="Initia Token", price=1.0, chg24_pct=-1.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="SHORT D — Long Liquidation Cascade",
        direction="SHORT", checks=[Check("cascade", True, "yes", "rule")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "INITUSDT",
            "weak_bounce_rejection": {"confirmed": True, "resistance": 1.02, "rejection_open_time_ms": 1_790_000_000_000},
        },
    )
    message = format_alert(result, enriched[0], result.technical)
    first, second = message.splitlines()[:2]
    assert first == "🔴 <b>SHORT</b> · Initia Token (<code>INITUSDT</code>)"
    assert second == "<b>SHORT D · CONFIRMED · P1 1/1</b>"
    assert "Initia Token" in first and "<code>INITUSDT</code>" in first


def test_telegram_header_uses_direction_emoji_and_full_contract_ticker_for_long():
    from screener.telegram import format_alert

    market = MarketRow(symbol="PARTIUSDT_PERP.A", name="Particle Network", price=0.5, chg24_pct=2.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "yes", "rule")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "PARTIUSDT",
            "trend_retest_long": {"confirmed": True, "level": 0.48, "retest_open_time_ms": 1_790_000_000_000},
        },
    )
    message = format_alert(result, enriched[0], result.technical)
    first, second = message.splitlines()[:2]
    assert first == "🟢 <b>LONG</b> · Particle Network (<code>PARTIUSDT</code>)"
    assert second == "<b>LONG A · CONFIRMED · P1 1/1</b>"


def test_contract_ticker_keeps_quote_asset_and_removes_only_market_suffix():
    from screener.telegram import _contract_symbol

    assert _contract_symbol("INITUSDT_PERP.A") == "INITUSDT"
    assert _contract_symbol("PARTIUSDT_PERP") == "PARTIUSDT"
    assert _contract_symbol("BTCUSDT") == "BTCUSDT"
    assert _contract_symbol("FOO-USDT-SWAP") == "FOO-USDT"


def test_telegram_alert_is_compact_and_has_no_repeated_disclaimer():
    from screener.telegram import format_alert

    market = MarketRow(symbol="BTCUSDT_PERP", name="Bitcoin", price=70000, chg24_pct=1)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("condition", True, "yes", "rule")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "BTCUSDT",
            "trend_retest_long": {"confirmed": True, "level": 69_000.0, "retest_open_time_ms": 1_790_000_000_000},
        },
    )
    message = format_alert(result, enriched[0], result.technical)
    assert len(message) <= 3900
    assert "P1 не проходили backtest" not in message
    assert "команда на вход" not in message


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
        "binance_symbol": "ALT0USDT",
        "trend_retest_long": {"confirmed": True, "level": 1.0, "retest_open_time_ms": 1_790_000_000_000, "reason": "test fixture: closed 1H retest"},
        "higher_low": {"confirmed": False, "reason": "not applicable"},
        "weak_bounce_rejection": {"confirmed": False},
        "execution_cost_assumptions": {"taker_fee_bps_per_side": None},
    })

    first = service.run_scan_once(settings, dry_run=True)
    assert first["status"] == "OK"
    assert first["rows"] == 20
    assert first["snapshot_saved"] is True
    assert first["notifications"] == []
    assert not any(item.status in {"OBSERVATION", "WAIT_BOUNCE", "PREPARATION", "TRIGGER"} for item in first["results"])
    assert "output_paths" not in first
    assert not (tmp_path / "reports").exists()
    assert (settings.data_dir / "snapshots.jsonl").exists()
    assert (settings.data_dir / "market_rows.jsonl").exists()
    assert (settings.data_dir / "notifications.jsonl").exists()
    assert (settings.data_dir / "runs.jsonl").exists()

    store = Store(settings.data_dir)
    scraped.fetched_at = fixed_time + timedelta(minutes=30)
    second = service.run_scan_once(settings, dry_run=True)
    assert second["no_new_coinalyze_data"] is True
    assert second["snapshot_saved"] is False
    assert second["notifications"] == []
    # Polling in the same UTC slot does not create a duplicate independent observation.
    latest_history = store.recent_symbol_history("ALT0USDT_PERP.A", limit=5)
    assert latest_history[-1]["snapshot_at"] == fixed_time.isoformat()

    # The first identical poll in the next four-hour UTC slot is a distinct scheduled observation.
    scraped.fetched_at = fixed_time + timedelta(hours=4)
    third = service.run_scan_once(settings, dry_run=True)
    assert third["no_new_coinalyze_data"] is True
    assert third["snapshot_saved"] is True
    latest_history = Store(settings.data_dir).recent_symbol_history("ALT0USDT_PERP.A", limit=5)
    assert len(latest_history) == 2
    assert latest_history[-1]["snapshot_at"] == (fixed_time + timedelta(hours=4)).isoformat()


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


def test_telegram_alert_shows_market_data_and_minimal_graph_context():
    from screener.telegram import format_alert

    market = MarketRow(
        symbol="GRASSUSDT_PERP.A", name="GRASS", price=1.0, chg24_pct=4.92,
        oi_usd=80_000_000, oi_chg24_pct=5.21, oi_chg24_usd=3_000_000,
        oi_chg4h_pct=1.50, oi_chg4h_usd=900_000,
        fr_oiw_pct=0.0096, pfr_oiw_pct=-0.0049, ls_ratio_1d=1.08,
        futures_imbalance_4h=7.6293, spot_imbalance_4h=4.2696,
        btc_corr7d=0.88, liq_long24_usd=50_000, liq_short24_usd=50_000,
        liq_total24_usd=100_000,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check(f"C{i}", True, "ok", "rule") for i in range(6)],
        quality_flags=[], status="CONFIRMED_SIGNAL_WITH_WARNING",
        technical={
            "binance_symbol": "GRASSUSDT",
            "context_4h": {"ema20": 1.0, "ema50": 1.1, "last_close": 1.02, "last_confirmed_support": 0.98},
            "trend_retest_long": {"confirmed": True, "level": 0.98, "retest_open_time_ms": 1_790_000_000_000},
        },
    )
    message = format_alert(result, enriched[0], result.technical)
    assert "GRASS (<code>GRASSUSDT</code>)" in message
    assert "LONG A · CONFIRMED · P1 6/6 · WARNING" in message
    assert "Buy F/S: <code>53,81%</code> / <code>52,13%</code>" in message
    assert "Flow Gap (S−F): <code>-3,3597</code>" in message
    assert "OI 4H: <code>+1,50%</code> / <code>+$900,0k</code>" in message
    assert "OI 24H: <code>+5,21%</code> / <code>+$3,0m</code>" in message
    assert "FR / PFR / Δ:" in message
    assert "BTC Corr. 7D: <code>+0,88</code>" in message
    assert "Trigger candle:" in message and "Invalidation:" in message
    assert "PFR OI_W меняет знак" in message
    assert "15m" not in message


def test_funding_sign_reversal_is_warning_only_after_chart_confirmation(tmp_path):
    from screener.telegram import format_alert
    from screener.scenarios import evaluate_scenarios
    from screener.config import DEFAULT_P1_CONFIG

    market = MarketRow(
        symbol="GRASSUSDT_PERP.A", name="GRASS", price=1.0, chg24_pct=5.09,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=5.21, oi_chg24_usd=5_000_000,
        oi_chg4h_pct=1.67, oi_chg4h_usd=901_200,
        oi_vol_ratio=0.5, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.2,
        fr_avg_pct=0.0096, fr_oiw_pct=0.0096, pfr_oiw_pct=-0.0049,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=1.08, futures_imbalance_4h=7.6293, spot_imbalance_4h=4.2696,
        btc_corr7d=0.88,
    )
    enriched, stats = analyze_universe([market], DEFAULT_P1_CONFIG)
    technical = {
        "trend_retest_long": {"confirmed": True, "level": 0.98, "retest_open_time_ms": 1_790_000_000_000},
    }
    results = evaluate_scenarios(enriched[0], stats, DEFAULT_P1_CONFIG, technical=technical)
    long_a = next(item for item in results if item.scenario.startswith("LONG A"))
    assert long_a.passed_count == 6 and long_a.total_count == 6
    assert long_a.status == "CONFIRMED_SIGNAL_WITH_WARNING"
    technical.update({
        "binance_symbol": "GRASSUSDT",
        "context_4h": {"ema20": 1.0, "ema50": 1.1, "last_close": 1.02, "last_confirmed_support": 0.98},
    })
    message = format_alert(long_a, enriched[0], technical)
    assert "CONFIRMED · P1 6/6 · WARNING" in message
    assert "PFR OI_W меняет знак относительно FR OI_W" in message


def test_short_d_is_blocked_when_liquidations_cover_less_than_ten_percent_of_oi_drop():
    from screener.scenarios import evaluate_scenarios

    market = MarketRow(
        symbol="XRPUSDT_PERP.A", name="XRP", price=0.5, chg24_pct=-0.68,
        oi_usd=1_052_513_986, oi_chg24_pct=-5.72, oi_chg24_usd=-60_203_800,
        oi_chg4h_pct=-0.64, oi_chg4h_usd=-1_000_000,
        fr_oiw_pct=0.0093, pfr_oiw_pct=0.0090, ls_ratio_1d=3.39,
        futures_imbalance_4h=-2.0372, spot_imbalance_4h=-7.8957,
        btc_corr7d=0.91, liq_long24_usd=2_900_000, liq_short24_usd=586_800,
        liq_total24_usd=3_486_800,
    )
    enriched, stats = analyze_universe([market], {"thresholds": {}})
    results = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}},
        technical={"weak_bounce_rejection": {"confirmed": True, "resistance": 0.51, "rejection_open_time_ms": 1_790_000_000_000}},
    )
    short_d = next(item for item in results if item.scenario.startswith("SHORT D"))
    assert abs(enriched[0].liquidation_coverage_of_oi_drop_pct - 5.79) < 0.01
    assert short_d.status == "BLOCKED_CAUSALITY"
    coverage = next(check for check in short_d.checks if check.name == "Liquidations cover OI decline")
    assert coverage.passed is False


def test_market_summary_formatter_is_not_part_of_telegram_runtime():
    import screener.telegram as telegram_module
    assert not hasattr(telegram_module, "format_market_summary")


def test_market_summary_event_key_is_not_part_of_notification_runtime():
    import screener.service as service
    assert not hasattr(service, "_market_summary_event_key")


def test_pons_long_a_five_of_six_is_not_delivered():
    from screener.config import DEFAULT_P1_CONFIG
    from screener.scenarios import evaluate_scenarios

    markets = []
    for index in range(75):
        symbol = "PONSUSDT_PERP.A" if index == 11 else f"ALT{index:02d}USDT_PERP.A"
        markets.append(MarketRow(
            symbol=symbol, name=symbol, price=1.0, chg24_pct=5.07,
            market_cap_usd=1_000_000_000, volume24_usd=500_000_000, oi_usd=50_000_000,
            oi_chg24_pct=1.0, oi_chg24_usd=1_000_000,
            oi_chg4h_pct=2.62, oi_chg4h_usd=3_000_000,
            oi_vol_ratio=0.5, oi_mktcap_ratio=0.05, vol_mktcap_ratio=0.5,
            fr_avg_pct=0.01, fr_oiw_pct=0.0090, pfr_oiw_pct=0.0085,
            liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
            ls_ratio_1d=1.74, futures_imbalance_4h=float(100 - index),
            spot_imbalance_4h=None if index >= 72 else float(100 - index),
        ))
    enriched, stats = analyze_universe(markets, DEFAULT_P1_CONFIG)
    pons = next(item for item in enriched if item.symbol.startswith("PONS"))
    result = next(item for item in evaluate_scenarios(pons, stats, DEFAULT_P1_CONFIG) if item.scenario.startswith("LONG A"))
    assert result.passed_count == 5 and result.total_count == 6
    assert result.status == "NOT_MET"
    settings = SimpleNamespace(telegram_bot_token="", telegram_chat_ids=[])
    outcomes = _collect_notifications([result], enriched, settings, FakeStore(), True, True)
    assert outcomes == []


def test_notification_identity_is_stable_for_metric_changes_but_changes_for_new_trigger():
    from screener.service import _notification_identity

    base = ScenarioResult(
        symbol="STRKUSDT_PERP.A", name="STRK", scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "positive", ">0")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "STRKUSDT",
            "trend_retest_long": {"confirmed": True, "level": 1.04, "retest_open_time_ms": 1_790_000_000_000},
        },
    )
    initial = _notification_identity(base)
    assert initial is not None
    dynamic_change = ScenarioResult(
        symbol=base.symbol, name=base.name, scenario=base.scenario, direction=base.direction,
        checks=[Check("flow", True, "changed F/S/OI", ">0")], quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={**base.technical, "futures_imbalance_4h": 55, "oi_chg4h_pct": 4.5},
    )
    assert _notification_identity(dynamic_change) == initial
    new_trigger = ScenarioResult(
        symbol=base.symbol, name=base.name, scenario=base.scenario, direction=base.direction,
        checks=list(base.checks), quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={"binance_symbol": "STRKUSDT", "trend_retest_long": {
            "confirmed": True, "level": 1.05, "retest_open_time_ms": 1_790_003_600_000,
        }},
    )
    assert _notification_identity(new_trigger)[1] != initial[1]


def test_market_summary_key_helper_is_not_part_of_runtime():
    import screener.service as service
    assert not hasattr(service, "_market_summary_event_key")


def test_telegram_alert_keeps_invalidation_but_omits_15m_context():
    from screener.telegram import format_alert

    market = MarketRow(symbol="STRKUSDT_PERP.A", name="STRK", price=1.0, chg24_pct=2.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "yes", "rule")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "STRKUSDT",
            "context_4h": {"trend": "EMA20 < EMA50", "ema20": 1.0, "ema50": 1.1,
                           "last_close": 1.05, "last_confirmed_support": 1.0},
            "trend_retest_long": {"confirmed": True, "level": 1.04, "retest_open_time_ms": 1_790_000_000_000},
            "context_15m": {"direction": "BULLISH", "last_closed_change_pct": 0.42},
        },
    )
    message = format_alert(result, enriched[0], result.technical)
    assert "15m" not in message
    assert "Trigger candle:" in message
    assert "Invalidation: 1H close ниже <code>1,04000000</code>" in message


def test_telegram_client_retries_rate_limit_once_then_succeeds(monkeypatch):
    from screener.telegram import TelegramClient
    import screener.telegram as telegram_module

    calls = []

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.headers = {}

        def json(self):
            return self.payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise telegram_module.requests.HTTPError(f"HTTP {self.status_code}")

    responses = [
        Response(429, {"ok": False, "parameters": {"retry_after": 1}}),
        Response(200, {"ok": True, "result": {"message_id": 77}}),
    ]

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    sleeps = []
    monkeypatch.setattr(telegram_module.requests, "post", fake_post)
    monkeypatch.setattr(telegram_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    client = TelegramClient("token", ["chat-1"])
    result = client.send("test message")

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert result == [{"chat_id": "chat-1", "sent": True, "delivery_unknown": False, "message_id": 77}]


def test_safe_truncation_fails_closed_instead_of_silently_cutting_a_message():
    import pytest
    from screener.telegram import _safe_telegram_truncate
    assert _safe_telegram_truncate("<b>LONG</b> <code>ZROUSDT</code>") == "<b>LONG</b> <code>ZROUSDT</code>"
    with pytest.raises(ValueError, match="exceeds safe limit"):
        _safe_telegram_truncate("<b>" + ("X" * 4000) + "</b>", limit=3900)


def test_active_setup_is_not_repeated_when_chart_state_updates(tmp_path):
    from screener.service import _notification_identity
    from screener.storage import Store

    first = ScenarioResult(
        symbol="STRKUSDT_PERP.A", name="STRK", scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "ok", ">0")], quality_flags=[],
        status="CONFIRMED_SIGNAL",
        technical={"binance_symbol": "STRKUSDT", "trend_retest_long": {
            "confirmed": True, "level": 1.04, "retest_open_time_ms": 1_790_000_000_000,
        }},
    )
    changed_metrics = ScenarioResult(
        symbol=first.symbol, name=first.name, scenario=first.scenario, direction=first.direction,
        checks=list(first.checks), quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={**first.technical, "futures_imbalance_4h": 18.0, "oi_chg4h_pct": 3.1},
    )
    assert _notification_identity(first) == _notification_identity(changed_metrics)
    store = Store(tmp_path / "state")
    group, event_key, fingerprint, level = _notification_identity(first)
    store.record_notification(event_key, 1, chat_id="chat-1", event_group=group,
        fingerprint=fingerprint, event_state="ACTIVE", direction="LONG",
        invalidation_level=level, pair="STRKUSDT", scenario_code="LONG A")
    assert store.notification_exists(event_key, "chat-1")
    assert len(store.active_notification_states(chat_ids=["chat-1"])) == 1


def test_technical_for_symbol_uses_closed_15m_and_derives_24h_low_from_closed_1h():
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from screener.models import Candle
    from screener.service import _technical_for_symbol

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    intervals = {"4h": 4 * 60 * 60 * 1000, "1h": 60 * 60 * 1000, "15m": 15 * 60 * 1000}

    class FakeBinance:
        def resolve_symbol(self, symbol):
            return "STRKUSDT"

        def fetch_candles(self, symbol, interval, limit):
            assert symbol == "STRKUSDT"
            interval_ms = intervals[interval]
            rows = []
            count = 100
            for index in range(count):
                open_time = now_ms - (count - index) * interval_ms
                open_price = 100.0 + index * 0.01
                close_price = open_price + (0.5 if interval == "15m" and index == count - 1 else 0.05)
                low = open_price - 0.2
                if interval == "1h" and index == count - 2:
                    low = 95.0
                high = max(open_price, close_price) + 0.2
                rows.append(Candle(
                    open_time_ms=open_time,
                    close_time_ms=open_time + interval_ms - 1,
                    open=open_price, high=high, low=low, close=close_price,
                    volume=1000, quote_volume=100000, trades=100,
                ))
            return rows

    settings = SimpleNamespace(
        p1={"structure": {}, "scanner": {"fetch_15m_context": True}},
        binance_kline_limit=100,
        binance_fetch_orderbook=False,
        execution_notional_usd=0.0,
        trading_fee_bps_per_side=5.0,
    )
    technical = _technical_for_symbol(FakeBinance(), "STRKUSDT_PERP.A", settings)
    assert technical["context_15m"]["direction"] == "BULLISH"
    assert technical["context_15m"]["last_closed_change_pct"] > 0
    assert technical["context_15m"]["source"] == "last closed Binance 15m candle"
    assert technical["range_24h_1h"]["low"] == 95.0
    assert technical["range_24h_1h"]["distance_from_low_pct"] is not None


def test_telegram_pair_is_copyable_code_and_send_has_no_inline_keyboard(monkeypatch):
    import screener.telegram as telegram_module
    from screener.telegram import TelegramClient, format_alert

    market = MarketRow(symbol="ZROUSDT_PERP.A", name="LayerZero", price=2.0, chg24_pct=4.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("P1", True, "pass", "rule")], quality_flags=[],
        status="CONFIRMED_SIGNAL",
        technical={"binance_symbol": "ZROUSDT", "trend_retest_long": {
            "confirmed": True, "level": 1.9, "retest_open_time_ms": 1_790_000_000_000,
        }},
    )
    message = format_alert(result, enriched[0], result.technical)
    assert "LayerZero (<code>ZROUSDT</code>)" in message

    posted = []
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"ok": True, "result": {"message_id": 1}}

    def fake_post(url, **kwargs):
        posted.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(telegram_module.requests, "post", fake_post)
    TelegramClient("token", ["chat-1"]).send(message)
    assert posted[0]["parse_mode"] == "HTML"
    assert "<code>ZROUSDT</code>" in posted[0]["text"]
    assert "reply_markup" not in posted[0]



def test_short_e_has_a_stable_notification_anchor_for_service_and_telegram():
    from datetime import datetime, timezone
    from screener.service import _confirmation_anchor, _has_required_confirmation, _notification_identity
    from screener.telegram import _confirmation_anchor as telegram_confirmation_anchor

    market = MarketRow(symbol="BTCUSDT_PERP", name="Bitcoin", price=99.0, chg24_pct=-1.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    anchor = 1_790_000_000_000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    close_time_ms = now_ms - 60_000
    open_time_ms = close_time_ms - (60 * 60_000 - 1)
    result = ScenarioResult(
        symbol="BTCUSDT_PERP", name="Bitcoin", scenario="SHORT E — Failed Short Squeeze",
        direction="SHORT", checks=[Check("failed breakout", True, "confirmed", "rule")],
        quality_flags=[], status="CONFIRMED_SIGNAL",
        technical={
            "binance_symbol": "BTCUSDT",
            "last_closed_1h": {
                "open_time_ms": open_time_ms,
                "close_time_ms": close_time_ms,
                "close": 99.0,
            },
            "context_4h": {
                "last_close": 99.0,
                "last_close_time_ms": close_time_ms,
                "last_confirmed_resistance": 110.0,
            },
            "failed_breakout_rejection": {
                "confirmed": True,
                "failed_level": 100.0,
                "last_close": 99.0,
                "failure_open_time_ms": anchor,
            },
        },
    )

    assert _confirmation_anchor(result) == anchor
    assert telegram_confirmation_anchor(result) == anchor
    assert _has_required_confirmation(result, enriched[0]) is True
    identity = _notification_identity(result)
    assert identity is not None
    assert identity[1].endswith(f"|{anchor}")



def test_telegram_request_errors_redact_bot_token_and_recipient_from_logs(monkeypatch, caplog):
    import logging
    import screener.telegram as telegram_module
    from screener.telegram import TelegramClient

    token = "123456:SECRET-BOT-TOKEN"

    def failed_post(url, **kwargs):
        recipient = kwargs.get("json", {}).get("chat_id", "")
        raise telegram_module.requests.ConnectionError(
            f"Connection failed for recipient {recipient} at {url}: timed out"
        )

    monkeypatch.setattr(telegram_module.requests, "post", failed_post)
    with caplog.at_level(logging.ERROR, logger="screener.telegram"):
        result = TelegramClient(token, ["chat-1"]).send("test message")

    assert result[0]["delivery_unknown"] is True
    assert token not in str(result)
    assert "[REDACTED]" in result[0]["error"]
    assert token not in result[0]["error"]
    assert "chat-1" not in result[0]["error"]
    assert token not in caplog.text
    assert "chat-1" not in caplog.text
