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
    assert len(store.exists_calls) == 1
    assert store.exists_calls[0].startswith("ETHUSDT_PERP|LONG A|")
    assert len(store.exists_calls[0].rsplit("|", 1)[1]) == 20
    assert store.due_calls == []


def test_telegram_header_uses_direction_emoji_and_full_contract_ticker_for_short():
    from screener.telegram import format_alert

    market = MarketRow(symbol="INITUSDT_PERP.A", name="Initia Token", price=1.0, chg24_pct=-1.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        # Deliberately different result label: the displayed ticker must come
        # from the MarketRow data, never from the friendly result.name.
        symbol="NOT_THE_SOURCE_TICKER", name=market.name, scenario="SHORT D — Long Liquidation Cascade",
        direction="SHORT", checks=[Check("cascade", True, "yes", "rule")],
        quality_flags=[], status="WAIT_BOUNCE",
    )

    message = format_alert(result, enriched[0])
    first, second = message.splitlines()[:2]

    assert first == "🔴 <b>SHORT</b> · <code>INITUSDT</code>"
    assert second == "<b>SHORT D · WAIT_BOUNCE · 1/1</b>"
    assert "Initia Token" not in first
    assert "INIT</code>" not in first


def test_telegram_header_uses_direction_emoji_and_full_contract_ticker_for_long():
    from screener.telegram import format_alert

    market = MarketRow(symbol="PARTIUSDT_PERP.A", name="Particle Network", price=0.5, chg24_pct=2.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "yes", "rule")],
        quality_flags=[], status="OBSERVATION",
    )

    message = format_alert(result, enriched[0])
    first, second = message.splitlines()[:2]

    assert first == "🟢 <b>LONG</b> · <code>PARTIUSDT</code>"
    assert second == "<b>LONG A · OBSERVATION · 1/1</b>"
    assert "Particle Network" not in first


def test_contract_ticker_keeps_quote_asset_and_removes_only_market_suffix():
    from screener.telegram import _contract_symbol

    assert _contract_symbol("INITUSDT_PERP.A") == "INITUSDT"
    assert _contract_symbol("PARTIUSDT_PERP") == "PARTIUSDT"
    assert _contract_symbol("BTCUSDT") == "BTCUSDT"
    assert _contract_symbol("FOO-USDT-SWAP") == "FOO-USDT"


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
    from screener.telegram import DISCLAIMER
    assert message.endswith(DISCLAIMER)


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
    assert first["snapshot_saved"] is True
    assert any(item.get("type") == "market_summary" for item in first["notifications"])
    assert any(item.status == "OBSERVATION" for item in first["results"])
    assert "output_paths" not in first
    assert not (tmp_path / "reports").exists()
    assert (settings.data_dir / "snapshots.jsonl").exists()
    assert (settings.data_dir / "market_rows.jsonl").exists()
    assert (settings.data_dir / "notifications.jsonl").exists()
    assert (settings.data_dir / "runs.jsonl").exists()

    store = Store(settings.data_dir)
    prior_result = next(item for item in first["results"] if item.status == "OBSERVATION")
    event_fingerprint = service._scenario_event_fingerprint(prior_result)
    store.record_notification(
        f"{prior_result.symbol}|{prior_result.scenario}|{event_fingerprint}", "fixture"
    )
    scraped.fetched_at = fixed_time + timedelta(minutes=30)
    second = service.run_scan_once(settings, dry_run=True)
    assert second["no_new_coinalyze_data"] is True
    assert second["snapshot_saved"] is False
    assert not any(item.get("type") in {"scenario", "market_summary"} for item in second["notifications"])
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


def test_telegram_alert_shows_oi_usd_funding_divergence_and_graph_context():
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
        quality_flags=[], status="OBSERVATION",
        technical={
            "context_4h": {"ema20": 1.0, "ema50": 1.1, "last_close": 1.02, "last_confirmed_support": 0.98},
            "higher_low": {"confirmed": True},
            "trend_retest_long": {"confirmed": True},
        },
    )
    message = format_alert(result, enriched[0])
    assert "Buy F/S: <code>53,81%</code> / <code>52,13%</code>" in message
    assert "Spread S−F: <code>-3,3597</code>" in message
    assert "OI 4H: <code>+1,50%</code> / <code>+$900,0k</code> (малое в USD)" in message
    assert "OI 24H: <code>+5,21%</code> / <code>+$3,0m</code>" in message
    assert "FR <code>+0,0096%</code>; PFR <code>-0,0049%</code>" in message
    assert "ВНИМАНИЕ: прогнозный PFR OI_W меняет знак" in message
    assert "BTC Corr. 7D: <code>+0,88</code>" in message
    assert "Higher Low подтверждён" in message
    assert "LONG retest подтверждён" in message
    assert "OI 4H изменение мало в USD" in message
    # EMA bearish context is a caution, not an automatic invalidation when support/retest hold.
    assert "контртрендовая ситуация" in message


def test_funding_sign_reversal_is_sent_as_observation_with_warning(tmp_path):
    from screener.telegram import format_alert
    from screener.scenarios import evaluate_scenarios

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
    enriched, stats = analyze_universe([market], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    long_a = next(item for item in results if item.scenario.startswith("LONG A"))

    assert long_a.passed_count == 6
    assert long_a.total_count == 6
    assert long_a.status == "OBSERVATION_WITH_WARNING"
    message = format_alert(long_a, enriched[0], {
        "context_4h": {"ema20": 1.0, "ema50": 1.1, "last_close": 1.02, "last_confirmed_support": 0.98},
        "higher_low": {"confirmed": True},
        "trend_retest_long": {"confirmed": True},
    })
    assert "OBSERVATION · WARNING · 6/6" in message
    assert "прогнозный PFR OI_W меняет знак" in message
    assert "не считать его безоговорочным наблюдением" in message

    settings = SimpleNamespace(
        telegram_bot_token="", telegram_chat_ids=[], telegram_send_wait_states=True,
        telegram_cooldown_hours=4,
    )
    from screener.storage import Store
    deliveries = _collect_notifications(
        [long_a], enriched, settings, Store(tmp_path / "funding-state"),
        contract_alerts_allowed=True, dry_run=True,
    )
    scenario_delivery = next(item for item in deliveries if item.get("type") == "scenario")
    assert "OBSERVATION · WARNING · 6/6" in scenario_delivery["text"]


def test_short_d_alert_warns_when_liquidations_cover_less_than_ten_percent_of_oi_drop():
    from screener.telegram import format_alert

    market = MarketRow(
        symbol="XRPUSDT_PERP.A", name="XRP", price=0.5, chg24_pct=-0.68,
        oi_usd=1_052_513_986, oi_chg24_pct=-5.72, oi_chg24_usd=-60_203_800,
        oi_chg4h_pct=-0.64, oi_chg4h_usd=-1_000_000,
        fr_oiw_pct=0.0093, pfr_oiw_pct=0.0090, ls_ratio_1d=3.39,
        futures_imbalance_4h=-2.0372, spot_imbalance_4h=-7.8957,
        btc_corr7d=0.91, liq_long24_usd=2_900_000, liq_short24_usd=586_800,
        liq_total24_usd=3_486_800,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name="XRP", scenario="SHORT D — Long Liquidation Cascade",
        direction="SHORT", checks=[Check(f"C{i}", True, "ok", "rule") for i in range(5)],
        quality_flags=[], status="WAIT_BOUNCE",
        technical={
            "context_4h": {"ema20": 0.49, "ema50": 0.51, "last_close": 0.48, "last_confirmed_resistance": 0.50},
            "higher_low": {"confirmed": True},
            "weak_bounce_rejection": {"confirmed": False, "reason": "No weak bounce rejection confirmed"},
        },
    )
    assert abs(enriched[0].liquidation_coverage_of_oi_drop_pct - 5.79) < 0.01
    message = format_alert(result, enriched[0])
    assert "LiqImb (Short−Long): <code>-66,34%</code>" in message
    assert "Liqs. 24H / падение OI 24H: <code>5,79%</code>" in message
    assert "ликвидации объясняют только 5,79% снижения OI" in message
    assert "Каскадная интерпретация ослаблена" in message
    assert "SHORT у минимума запрещён" in message
    assert "отказ от сопротивления после 1h-отскока не подтверждён" in message.lower()


def test_market_summary_includes_whole_market_regime_btc_eth_and_divergences():
    from datetime import datetime, timezone
    from screener.metrics import analyze_universe
    from screener.telegram import format_market_summary

    btc = MarketRow(
        symbol="BTCUSDT_PERP.A", name="Bitcoin", chg24_pct=0.32,
        oi_chg24_pct=0.08, oi_chg24_usd=21_600_000, oi_chg4h_pct=0.33,
        oi_chg4h_usd=86_800_000, fr_oiw_pct=0.0084, ls_ratio_1d=1.60,
        futures_imbalance_4h=-0.2838, spot_imbalance_4h=-0.3723, btc_corr7d=1.0,
    )
    eth = MarketRow(
        symbol="ETHUSDT_PERP.A", name="Ethereum", chg24_pct=0.06,
        oi_chg24_pct=-1.29, oi_chg24_usd=-176_400_000, oi_chg4h_pct=0.40,
        oi_chg4h_usd=53_200_000, fr_oiw_pct=0.0042, ls_ratio_1d=1.92,
        futures_imbalance_4h=-1.5127, spot_imbalance_4h=-1.4750, btc_corr7d=0.96,
    )
    rows, stats = analyze_universe([btc, eth], {"thresholds": {}})
    text = format_market_summary(rows, stats, datetime.now(timezone.utc), "PERPETUAL")
    assert "СВОДКА РЫНКА P1" in text
    assert "BTC:" in text and "ETH:" in text
    assert "F/S разного знака" in text
    assert "Цена↑/OI24↓" in text
    assert "OI 24H <code>-0,60%</code> / <code>-$77,4m</code>" in text
    assert "OI 4H <code>+0,36%</code> / <code>+$70,0m</code>" in text
    assert "P1 — стартовые гипотезы без backtest" in text
    assert len(text) <= 3900


def test_market_summary_matches_75_row_reference_counts_ranks_and_correlations():
    from datetime import datetime, timezone
    from screener.telegram import format_market_summary

    markets = []
    for index in range(75):
        if index == 0:
            symbol = "BTCUSDT_PERP.A"
            corr = 1.0
        elif index == 11:
            # This is PONS in the reference report: rank 12, outside top 15%.
            symbol = "PONSUSDT_PERP.A"
            corr = 0.88
        else:
            symbol = f"ALT{index:02d}USDT_PERP.A"
            corr = 0.80 if index < 45 else 0.30

        spot = None if index >= 72 else float(100 - index)
        price = 1.0
        oi24 = 1.0
        if index == 0:
            price, oi24 = 1.0, -1.0
        elif index == 1:
            price, oi24 = -1.0, -1.0
        elif index == 2:
            price, oi24 = -1.0, 1.0

        markets.append(MarketRow(
            symbol=symbol,
            name=symbol,
            price=10.0,
            chg24_pct=price,
            market_cap_usd=1_000_000_000,
            volume24_usd=200_000_000,
            oi_usd=80_000_000,
            oi_chg24_pct=oi24,
            oi_chg24_usd=oi24 * 1_000_000,
            oi_chg4h_pct=1.0,
            oi_chg4h_usd=1_000_000,
            oi_vol_ratio=0.4,
            oi_mktcap_ratio=0.08,
            vol_mktcap_ratio=0.2,
            fr_avg_pct=0.001,
            fr_oiw_pct=0.002,
            pfr_oiw_pct=0.002,
            liq_total24_usd=100_000,
            liq_long24_usd=50_000,
            liq_short24_usd=50_000,
            ls_ratio_1d=1.2,
            futures_imbalance_4h=(-31.0 if index == 69 else float(100 - index)),
            spot_imbalance_4h=spot,
            btc_corr7d=corr,
        ))

    rows, stats = analyze_universe(markets, {"thresholds": {"strong_flow_quantile": 0.15}})
    pons = next(item for item in rows if item.symbol.startswith("PONS"))
    assert pons.f_rank_desc == 12
    assert pons.s_rank_desc == 12
    assert pons.f_strong_buy is False
    assert pons.s_strong_buy is False

    text = format_market_summary(rows, stats, datetime.now(timezone.utc), "PERPETUAL")
    assert "F верхние 15% (n=11)" in text
    assert "F нижние 15% (n=11)" in text
    assert "S верхние 15% (n=11)" in text
    assert "S нижние 15% (n=11)" in text
    # BTC is included in the correlation statistic, but not in correlated-alt exposure count.
    assert "BTC Corr. 7D>0,7 <code>45/75</code>; связанных альтов <code>44</code>" in text
    assert "F/S разного знака <code>1/72</code>" in text
    assert "Цена↑/Spot↓ <code>0/72</code>" in text
    assert "Цена↓/Spot↑ <code>2/72</code>" in text
    assert "Цена↑/OI24↓ <code>1/75</code>" in text
    assert "Цена↓/OI24↓ <code>1/75</code>" in text
    assert "Цена↓/OI24↑ <code>1/75</code>" in text
    assert len(text) <= 3900


def test_pons_long_a_five_of_six_not_sent_as_observation_after_rank_fix():
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
    result = next(
        item for item in evaluate_scenarios(pons, stats, DEFAULT_P1_CONFIG)
        if item.scenario.startswith("LONG A")
    )
    assert result.passed_count == 5
    assert result.total_count == 6
    assert result.status == "NOT_MET"

    settings = SimpleNamespace(
        telegram_bot_token="", telegram_chat_ids=[], telegram_send_wait_states=True,
        telegram_cooldown_hours=4,
    )
    outcomes = _collect_notifications(
        [result], enriched, settings, FakeStore(), contract_alerts_allowed=True,
        dry_run=True, no_new_data=False,
    )
    assert outcomes == []


def test_scenario_notification_fingerprint_changes_on_chart_state_transition():
    from screener.service import _scenario_event_fingerprint

    base = ScenarioResult(
        symbol="STRKUSDT_PERP.A", name="STRK", scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "positive", ">0")],
        quality_flags=[], status="OBSERVATION",
        technical={
            "context_4h": {"trend": "EMA20 < EMA50", "ema20": 1.0, "ema50": 1.1,
                           "last_close": 1.05, "last_confirmed_support": 1.0},
            "trend_retest_long": {"confirmed": False, "level": 1.04},
        },
    )
    same = base
    confirmed = ScenarioResult(
        symbol=base.symbol, name=base.name, scenario=base.scenario, direction=base.direction,
        checks=list(base.checks), quality_flags=[], status="OBSERVATION",
        technical={
            "context_4h": {"trend": "EMA20 < EMA50", "ema20": 1.0, "ema50": 1.1,
                           "last_close": 1.05, "last_confirmed_support": 1.0},
            "trend_retest_long": {"confirmed": True, "level": 1.04, "retest_open_time_ms": 1790000000000},
        },
    )
    assert _scenario_event_fingerprint(base) == _scenario_event_fingerprint(same)
    assert _scenario_event_fingerprint(base) != _scenario_event_fingerprint(confirmed)


def test_market_summary_key_is_stable_per_saved_snapshot():
    from datetime import datetime, timezone
    from screener.service import _market_summary_event_key

    first = datetime(2026, 11, 10, 12, 1, tzinfo=timezone.utc)
    same_snapshot = datetime(2026, 11, 10, 12, 1, tzinfo=timezone.utc)
    next_snapshot = datetime(2026, 11, 10, 15, 55, tzinfo=timezone.utc)
    assert _market_summary_event_key("PERPETUAL", first) == _market_summary_event_key("PERPETUAL", same_snapshot)
    assert _market_summary_event_key("PERPETUAL", first) != _market_summary_event_key("PERPETUAL", next_snapshot)


def test_telegram_alert_includes_15m_timing_context_and_scenario_invalidation():
    from screener.telegram import format_alert

    market = MarketRow(symbol="STRKUSDT_PERP.A", name="STRK", price=1.0, chg24_pct=2.0)
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = ScenarioResult(
        symbol=market.symbol, name=market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[Check("flow", True, "yes", "rule")],
        quality_flags=[], status="OBSERVATION",
        technical={
            "context_4h": {"trend": "EMA20 < EMA50", "ema20": 1.0, "ema50": 1.1,
                           "last_close": 1.05, "last_confirmed_support": 1.0},
            "trend_retest_long": {"confirmed": True, "level": 1.04, "retest_open_time_ms": 1790000000000},
            "higher_low": {"confirmed": True, "latest_low": 0.98},
            "context_15m": {"direction": "BULLISH", "last_closed_change_pct": 0.42},
        },
    )
    message = format_alert(result, enriched[0])
    assert "15m контекст (последняя закрытая свеча): бычья" in message
    assert "изменение +0,42%" in message
    assert "Уровень отмены сценария" in message
    assert "закрытие 1H ниже уровня отменяет LONG" in message
    assert "уровень пробоя и ретеста 1H" in message


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
    assert result == [{"chat_id": "chat-1", "sent": True, "message_id": 77}]


def test_safe_truncation_keeps_html_well_formed_and_respects_telegram_limit():
    from html.parser import HTMLParser
    from screener.telegram import _safe_telegram_truncate, DISCLAIMER

    raw = "<b>" + ("A & B < C " * 900) + "</b>" + DISCLAIMER
    result = _safe_telegram_truncate(raw, limit=3900)
    assert len(result) <= 3900
    assert "<b>" not in result
    assert "&amp;" in result
    assert result.endswith(DISCLAIMER)
    parser = HTMLParser()
    parser.feed(result)
    parser.close()


def test_scenario_state_transitions_are_delivered_even_when_state_returns_to_previous_value(tmp_path, monkeypatch):
    import screener.service as service
    from screener.storage import Store

    market = MarketRow(
        symbol="STRKUSDT_PERP.A", name="STRK", price=1.0, chg24_pct=2.0,
        futures_imbalance_4h=8.0, spot_imbalance_4h=2.0,
        oi_chg4h_pct=1.2, oi_chg4h_usd=2_500_000,
        fr_oiw_pct=0.005, pfr_oiw_pct=0.004, ls_ratio_1d=1.2,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    settings = SimpleNamespace(
        telegram_bot_token="token", telegram_chat_ids=["chat-1"],
        telegram_send_wait_states=True, telegram_cooldown_hours=4,
    )
    store = Store(tmp_path / "state")
    sent = []

    class FakeTelegram:
        def __init__(self, token, chat_ids):
            self.chat_ids = chat_ids

        @property
        def configured(self):
            return True

        def send(self, message, chat_ids=None):
            recipients = self.chat_ids if chat_ids is None else chat_ids
            sent.append(message)
            return [{"chat_id": chat_id, "sent": True, "message_id": len(sent)} for chat_id in recipients]

    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)

    def make_result(retest_confirmed):
        return ScenarioResult(
            symbol=market.symbol, name=market.name,
            scenario="LONG A — Trend Continuation", direction="LONG",
            checks=[Check("flow", True, "ok", ">0")], quality_flags=[], status="OBSERVATION",
            technical={
                "context_4h": {"trend": "EMA20 < EMA50", "ema20": 1.0, "ema50": 1.1,
                               "last_close": 1.05, "last_confirmed_support": 1.0},
                "trend_retest_long": {"confirmed": retest_confirmed, "level": 1.04},
            },
        )

    # State A is delivered, then transition B is delivered immediately despite the four-hour cooldown.
    first = make_result(False)
    out_a = _collect_notifications(
        [first], enriched, settings, store, True, False, no_new_data=True,
    )
    assert len(sent) == 1 and out_a[0]["sent"] is True

    second = make_result(True)
    out_b = _collect_notifications(
        [second], enriched, settings, store, True, False, no_new_data=True,
    )
    assert len(sent) == 2 and out_b[0]["sent"] is True

    # Returning to a previously seen fingerprint is still a new transition relative to last state B.
    out_a_again = _collect_notifications(
        [first], enriched, settings, store, True, False, no_new_data=True,
    )
    assert len(sent) == 3 and out_a_again[0]["sent"] is True

    # Stable state A is not repeated on the next unchanged poll.
    out_stable = _collect_notifications(
        [first], enriched, settings, store, True, False, no_new_data=True,
    )
    assert out_stable == []
    assert len(sent) == 3


def test_technical_for_symbol_uses_closed_15m_and_derives_24h_low_from_closed_1h():
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from screener.models import Candle
    from screener.service import _technical_for_symbol

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    intervals = {"4h": 4 * 60 * 60 * 1000, "1h": 60 * 60 * 1000, "15m": 15 * 60 * 1000}

    class FakeBinance:
        def fetch_candles(self, symbol, interval, limit):
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
