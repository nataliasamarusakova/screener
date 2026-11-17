from screener.metrics import analyze_universe
from screener.models import MarketRow
from screener.scenarios import evaluate_scenarios


def test_long_a_requires_all_six_conditions_and_no_flags():
    rows = []
    for i in range(20):
        rows.append(MarketRow(
            symbol=f"ALT{i}", name=f"ALT{i}", price=10, chg24_pct=2,
            market_cap_usd=1_000_000_000, volume24_usd=100_000_000, oi_usd=100_000_000,
            oi_chg24_pct=2, oi_chg24_usd=2_000_000, oi_chg4h_pct=2, oi_chg4h_usd=3_000_000,
            fr_avg_pct=0.001, fr_oiw_pct=0.01, pfr_oiw_pct=0.01,
            liq_total24_usd=1_000_000, liq_long24_usd=500_000, liq_short24_usd=500_000,
            ls_ratio_1d=1.2, futures_imbalance_4h=0.5, spot_imbalance_4h=0.5,
        ))
    # Make exactly one row the upper-quintile flow leader while keeping other rows neutral/negative.
    rows[0].futures_imbalance_4h = 20
    rows[0].spot_imbalance_4h = 20
    enriched, stats = analyze_universe(rows, {"thresholds": {"strong_flow_quantile": 0.15, "quality_vol_mktcap_max": 2, "quality_oi_vol_min": 0.1, "quality_oi_min_usd": 40_000_000, "oi_4h_alt_pct": 1, "funding_hot_pct": 0.03, "ls_long_extreme": 2.5}})
    target = enriched[0]
    result = evaluate_scenarios(target, stats, {"thresholds": {"oi_4h_alt_pct": 1, "funding_hot_pct": 0.03, "ls_long_extreme": 2.5}}, technical={"trend_retest_long": {"confirmed": True}})
    long_a = next(item for item in result if item.scenario.startswith("LONG A"))
    assert long_a.complete
    assert long_a.status == "CONFIRMED_SIGNAL"


def test_missing_value_is_unknown_not_zero():
    row = MarketRow(symbol="A", name="A", futures_imbalance_4h=None, spot_imbalance_4h=2)
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    long_a = next(item for item in results if item.scenario.startswith("LONG A"))
    assert long_a.checks[0].passed is None


def test_long_a_waits_for_closed_1h_retest_before_confirmation():
    rows = []
    for i in range(20):
        rows.append(MarketRow(
            symbol=f"ALT{i}", name=f"ALT{i}", price=10, chg24_pct=2,
            market_cap_usd=1_000_000_000, volume24_usd=100_000_000, oi_usd=100_000_000,
            oi_chg24_pct=2, oi_chg24_usd=2_000_000, oi_chg4h_pct=2, oi_chg4h_usd=3_000_000,
            fr_avg_pct=0.001, fr_oiw_pct=0.01, pfr_oiw_pct=0.01,
            liq_total24_usd=1_000_000, liq_long24_usd=500_000, liq_short24_usd=500_000,
            ls_ratio_1d=1.2, futures_imbalance_4h=0.5, spot_imbalance_4h=0.5,
        ))
    rows[0].futures_imbalance_4h = 20
    rows[0].spot_imbalance_4h = 20
    enriched, stats = analyze_universe(rows, {"thresholds": {"strong_flow_quantile": 0.15, "quality_vol_mktcap_max": 2, "quality_oi_vol_min": 0.1, "quality_oi_min_usd": 40_000_000, "oi_4h_alt_pct": 1, "funding_hot_pct": 0.03, "ls_long_extreme": 2.5}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {"oi_4h_alt_pct": 1, "funding_hot_pct": 0.03, "ls_long_extreme": 2.5}})
    long_a = next(item for item in results if item.scenario.startswith("LONG A"))
    assert long_a.complete
    assert long_a.status == "WAITING_FOR_1H_BREAKOUT_RETEST"
    assert "not confirmed" in long_a.reason


def test_long_c_preparation_reports_two_of_two_stage_checks():
    row = MarketRow(
        symbol="ALTUSDT_PERP", name="ALT", price=10, chg24_pct=0,
        market_cap_usd=1_000_000_000, volume24_usd=100_000_000, oi_usd=100_000_000,
        oi_chg24_pct=0, oi_chg24_usd=0, oi_chg4h_pct=0, oi_chg4h_usd=0,
        oi_vol_ratio=1, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.1,
        fr_avg_pct=-0.01, fr_oiw_pct=-0.01, pfr_oiw_pct=-0.01,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=0.8, futures_imbalance_4h=1, spot_imbalance_4h=1,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    result = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    long_c = next(item for item in result if item.scenario.startswith("LONG C"))
    assert long_c.status == "PREPARATION"
    assert long_c.passed_count == 2
    assert long_c.total_count == 2
    assert long_c.technical["trigger_checks"]


def test_possible_long_liquidation_flush_blocks_short_even_when_flows_are_positive():
    row = MarketRow(
        symbol="ALTUSDT_PERP", name="ALT", price=10, chg24_pct=-6,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-8, oi_chg24_usd=-8_000_000, oi_chg4h_pct=-2, oi_chg4h_usd=-2_000_000,
        oi_vol_ratio=0.5, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.2,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=1_000_000, liq_long24_usd=900_000, liq_short24_usd=100_000,
        ls_ratio_1d=1.5, futures_imbalance_4h=5, spot_imbalance_4h=7,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    short_a = next(item for item in results if item.scenario.startswith("SHORT A"))
    no_cascade = next(check for check in short_a.checks if check.name == "No cascade")
    assert no_cascade.passed is False


def test_short_e_uses_live_flow_turn_down_without_waiting_for_another_4h_snapshot():
    row = MarketRow(
        symbol="ALTUSDT_PERP", name="ALT", price=10, chg24_pct=-2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-1, oi_chg24_usd=-1_000_000, oi_chg4h_pct=-0.2, oi_chg4h_usd=-200_000,
        oi_vol_ratio=0.5, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.2,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=1_000_000, liq_long24_usd=400_000, liq_short24_usd=600_000,
        ls_ratio_1d=1.5, futures_imbalance_4h=-4, spot_imbalance_4h=-5,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    from datetime import datetime, timedelta, timezone
    base_time = datetime.now(timezone.utc) - timedelta(hours=8)
    values = [
        (2, 3, 300_000, 700_000, -3),
        (1, 2, 500_000, 500_000, -1),
        (1, 2, 400_000, 600_000, -1),
    ]
    history = [
        {"snapshot_at": (base_time + timedelta(hours=4 * index)).isoformat(),
         "futures_imbalance_4h": value[0], "spot_imbalance_4h": value[1],
         "liq_long24_usd": value[2], "liq_short24_usd": value[3], "oi_chg24_pct": value[4]}
        for index, value in enumerate(values)
    ]
    technical = {"failed_breakout_rejection": {"confirmed": True}}
    result = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history, technical=technical,
        as_of=base_time + timedelta(hours=8),
    )
    short_e = next(item for item in result if item.scenario.startswith("SHORT E"))
    assert short_e.status == "CONFIRMED_SIGNAL"
    assert short_e.complete

    result_without_3 = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history[-2:], technical=technical,
        as_of=base_time + timedelta(hours=8),
    )
    short_e_without_3 = next(item for item in result_without_3 if item.scenario.startswith("SHORT E"))
    assert short_e_without_3.status == "NOT_MET"
    assert short_e_without_3.checks[0].passed is False


def test_pons_is_five_of_six_not_long_a_when_ranked_twelfth_in_both_flows():
    from screener.config import DEFAULT_P1_CONFIG

    rows = []
    for index in range(75):
        symbol = "PONS" if index == 11 else f"ALT{index:02d}"
        spot = None if index >= 72 else float(100 - index)
        rows.append(MarketRow(
            symbol=symbol, name=symbol, price=1.0, chg24_pct=5.07,
            market_cap_usd=1_000_000_000, volume24_usd=500_000_000, oi_usd=50_000_000,
            oi_chg24_pct=1.0, oi_chg24_usd=1_000_000,
            oi_chg4h_pct=2.62, oi_chg4h_usd=3_000_000,
            oi_vol_ratio=0.5, oi_mktcap_ratio=0.05, vol_mktcap_ratio=0.5,
            fr_avg_pct=0.01, fr_oiw_pct=0.0090, pfr_oiw_pct=0.0085,
            liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
            ls_ratio_1d=1.74,
            futures_imbalance_4h=float(100 - index),
            spot_imbalance_4h=spot,
        ))

    enriched, stats = analyze_universe(rows, DEFAULT_P1_CONFIG)
    pons = next(item for item in enriched if item.symbol == "PONS")
    result = evaluate_scenarios(pons, stats, DEFAULT_P1_CONFIG)
    long_a = next(item for item in result if item.scenario.startswith("LONG A"))

    assert pons.f_rank_desc == 12
    assert pons.s_rank_desc == 12
    assert pons.f_strong_buy is False
    assert pons.s_strong_buy is False
    assert long_a.passed_count == 5
    assert long_a.total_count == 6
    assert long_a.status == "NOT_MET"


def test_grass_keeps_six_of_six_but_surfaces_funding_sign_reversal_warning():
    from copy import deepcopy
    from screener.config import DEFAULT_P1_CONFIG

    config = deepcopy(DEFAULT_P1_CONFIG)
    config["thresholds"]["strong_flow_min_rows"] = 1  # Isolate scenario behavior from the universe-size guard.
    row = MarketRow(
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
    enriched, stats = analyze_universe([row], config)
    result = evaluate_scenarios(
        enriched[0], stats, config,
        technical={"trend_retest_long": {"confirmed": True, "level": 0.95, "retest_open_time_ms": 1_790_000_000_000}},
    )
    long_a = next(item for item in result if item.scenario.startswith("LONG A"))

    assert long_a.passed_count == 6
    assert long_a.total_count == 6
    assert long_a.status == "CONFIRMED_SIGNAL_WITH_WARNING"
    assert "qualitative warning only" in long_a.reason
    assert any("changes sign versus current FR" in note for note in enriched[0].diagnostic_notes)
    assert "OI 4H change is small in USD" in enriched[0].diagnostic_notes


def test_oi_growth_threshold_uses_exact_major_asset_codes_only():
    from screener.scenarios import _oi_growth_min, _symbol_base

    thresholds = {"oi_4h_major_pct": 0.3, "oi_4h_alt_pct": 1.0}
    assert _symbol_base("BTCUSDT_PERP.A") == "BTC"
    assert _symbol_base("ETHUSDT_PERP.A") == "ETH"
    assert _symbol_base("ETHFIUSDT_PERP.A") == "ETHFI"
    assert _symbol_base("ETHWUSDT_PERP.A") == "ETHW"
    assert _symbol_base("BTCDOMUSDT_PERP.A") == "BTCDOM"
    assert _oi_growth_min("BTCUSDT_PERP.A", thresholds) == 0.3
    assert _oi_growth_min("ETHUSDT_PERP.A", thresholds) == 0.3
    for symbol in ("ETHFIUSDT_PERP.A", "ETHWUSDT_PERP.A", "BTCDOMUSDT_PERP.A", "STRKUSDT_PERP.A"):
        assert _oi_growth_min(symbol, thresholds) == 1.0


def test_short_e_requires_non_overlapping_4h_snapshots_and_fresh_history():
    from datetime import datetime, timedelta, timezone

    market = MarketRow(
        symbol="ALTUSDT_PERP", name="ALT", price=10, chg24_pct=-2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-1, oi_chg24_usd=-1_000_000, oi_chg4h_pct=-0.2, oi_chg4h_usd=-200_000,
        oi_vol_ratio=0.5, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.2,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=1_000_000, liq_long24_usd=400_000, liq_short24_usd=600_000,
        ls_ratio_1d=1.5, futures_imbalance_4h=-4, spot_imbalance_4h=-5,
    )
    enriched, stats = analyze_universe([market], {"thresholds": {}})
    base = datetime.now(timezone.utc) - timedelta(hours=7, minutes=48)
    values = [(2, 3, 300_000, 700_000, -3), (1, 2, 500_000, 500_000, -1), (-4, -5, 400_000, 600_000, -1)]

    def history_at(hours):
        return [
            {"snapshot_at": (base + timedelta(hours=gap)).isoformat(),
             "futures_imbalance_4h": value[0], "spot_imbalance_4h": value[1],
             "liq_long24_usd": value[2], "liq_short24_usd": value[3], "oi_chg24_pct": value[4]}
            for gap, value in zip(hours, values)
        ]

    tech = {"failed_breakout_rejection": {"confirmed": True}}
    latest_at = base + timedelta(hours=7, minutes=48)
    early_tolerance = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history_at((0, 3.9, 7.8)),
        technical=tech, as_of=latest_at,
    )
    assert next(item for item in early_tolerance if item.scenario.startswith("SHORT E")).status == "NOT_MET"

    independent_snapshots = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history_at((0, 4.0, 8.0)),
        technical=tech, as_of=base + timedelta(hours=8),
    )
    assert next(item for item in independent_snapshots if item.scenario.startswith("SHORT E")).status == "CONFIRMED_SIGNAL"

    too_early = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history_at((0, 3.8, 7.6)),
        technical=tech, as_of=base + timedelta(hours=7, minutes=36),
    )
    assert next(item for item in too_early if item.scenario.startswith("SHORT E")).status == "NOT_MET"

    late_gap = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}}, history=history_at((0, 4.2, 8.4)),
        technical=tech, as_of=base + timedelta(hours=8, minutes=24),
    )
    assert next(item for item in late_gap if item.scenario.startswith("SHORT E")).status == "NOT_MET"

    stale = evaluate_scenarios(
        enriched[0], stats, {"thresholds": {}},
        history=history_at((0, 4, 8)), technical=tech, as_of=latest_at + timedelta(days=2),
    )
    assert next(item for item in stale if item.scenario.startswith("SHORT E")).status == "NOT_MET"


def test_long_c_preparation_is_blocked_by_market_quality_flags():
    row = MarketRow(
        symbol="BTWUSDT_PERP", name="BTW", price=1.0, chg24_pct=0,
        market_cap_usd=1_000_000_000, volume24_usd=100_000_000, oi_usd=23_700_000,
        oi_chg24_pct=0, oi_chg24_usd=0, oi_chg4h_pct=0, oi_chg4h_usd=0,
        oi_vol_ratio=1.0, oi_mktcap_ratio=0.0237, vol_mktcap_ratio=0.1,
        fr_avg_pct=-0.01, fr_oiw_pct=-0.01, pfr_oiw_pct=-0.01,
        liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
        ls_ratio_1d=0.8, futures_imbalance_4h=1.0, spot_imbalance_4h=None,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    result = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    long_c = next(item for item in result if item.scenario.startswith("LONG C"))

    assert long_c.status == "BLOCKED_QUALITY"
    assert long_c.status != "PREPARATION"
    assert "SPOT_IMBALANCE_NA" not in long_c.quality_flags
    assert any("SPOT_IMBALANCE_NA" in note for note in enriched[0].diagnostic_notes)
    assert any(check.name == "Quality flags" and check.passed is False for check in long_c.checks)


def test_opposed_oi_and_positive_flows_remain_a_warning_not_a_scenario_veto():
    from datetime import datetime, timedelta, timezone

    rows = []
    for index in range(75):
        is_target = index == 0
        rows.append(MarketRow(
            symbol="TARGETUSDT_PERP" if is_target else f"ALT{index:02d}USDT_PERP",
            name="TARGET" if is_target else f"ALT{index:02d}",
            price=10, chg24_pct=1.0, market_cap_usd=1_000_000_000,
            volume24_usd=100_000_000, oi_usd=100_000_000,
            oi_chg24_pct=0, oi_chg24_usd=0,
            oi_chg4h_pct=-1.0 if is_target else 0.0, oi_chg4h_usd=-1_000_000 if is_target else 0,
            oi_vol_ratio=1.0, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.1,
            fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
            liq_total24_usd=100_000, liq_long24_usd=50_000, liq_short24_usd=50_000,
            ls_ratio_1d=1.2, futures_imbalance_4h=1.0 if is_target else -1.0,
            spot_imbalance_4h=20.0 if is_target else -1.0,
        ))
    enriched, stats = analyze_universe(rows, {"thresholds": {}})
    target = next(item for item in enriched if item.symbol == "TARGETUSDT_PERP")
    results = evaluate_scenarios(
        target, stats, {"thresholds": {}},
        technical={"spot_sequence_ok": True, "higher_low": {"confirmed": True}},
        history=[],
    )
    long_b = next(item for item in results if item.scenario.startswith("LONG B"))

    assert any("observation only" in note for note in target.diagnostic_notes)
    assert long_b.status == "CONFIRMED_SIGNAL"
    assert "observation only" not in long_b.reason.lower()


def test_native_scenario_flow_oi_signatures_keep_confirmed_status():
    from screener.models import Check
    from screener.scenarios import _result

    cases = (
        ("LONG B", -0.3, 0.2, 15.0),
        ("LONG D", -2.0, 2.0, 3.0),
        ("SHORT A", 0.4, -2.0, -20.0),
        ("SHORT B", 1.0, -1.5, -15.0),
        ("SHORT C", 0.3, -3.0, -2.0),
        ("SHORT E", 0.2, -1.0, -2.0),
    )
    for scenario, oi4h, futures, spot in cases:
        market = MarketRow(
            symbol="TESTUSDT_PERP", name="TEST", oi_chg4h_pct=oi4h,
            futures_imbalance_4h=futures, spot_imbalance_4h=spot,
            fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        )
        enriched, _ = analyze_universe([market], {"thresholds": {}})
        result = _result(
            enriched[0], f"{scenario} — native-signature test", "LONG" if scenario.startswith("LONG") else "SHORT",
            [Check("all P1 predicates", True, "passed", "scenario-local rules")],
            status="CONFIRMED_SIGNAL",
        )
        assert result.status == "CONFIRMED_SIGNAL", scenario
        assert "observation only" not in result.reason.lower(), scenario


def test_unlisted_opposed_flow_oi_signature_remains_observation_only():
    from screener.models import Check
    from screener.scenarios import _result

    market = MarketRow(
        symbol="TESTUSDT_PERP", name="TEST", oi_chg4h_pct=-0.2,
        futures_imbalance_4h=2.0, spot_imbalance_4h=3.0,
        fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = _result(
        enriched[0], "LONG C — Short Squeeze", "LONG",
        [Check("stage predicates", True, "passed", "scenario-local rules")],
        status="TRIGGER",
    )

    assert result.status == "OBSERVATION"
    assert "observation only" in result.reason.lower()


def test_long_c_trigger_is_always_labeled_observation_without_entry_gate():
    from screener.models import Check
    from screener.scenarios import _result

    market = MarketRow(
        symbol="TESTUSDT_PERP", name="TEST", oi_chg4h_pct=0.2,
        futures_imbalance_4h=2.0, spot_imbalance_4h=3.0,
    )
    enriched, _ = analyze_universe([market], {"thresholds": {}})
    result = _result(
        enriched[0], "LONG C — Short Squeeze", "LONG",
        [Check("all trigger predicates", True, "passed", "configured thresholds")],
        status="TRIGGER",
    )

    assert result.status == "OBSERVATION"
    assert "no defined structural entry anchor" in result.reason


def test_known_non_cascade_oi_direction_overrides_missing_liquidation_side():
    from screener.scenarios import _is_cascade

    row = MarketRow(
        symbol="VIRTUALUSDT_PERP", name="VIRTUAL", price=1.0,
        oi_chg24_pct=2.0, oi_chg4h_pct=1.35,
        liq_long24_usd=100_000.0, liq_short24_usd=None,
        futures_imbalance_4h=-2.0, spot_imbalance_4h=-5.0,
    )
    enriched, _ = analyze_universe([row], {"thresholds": {}})
    assert enriched[0].long_liq_share is None
    assert _is_cascade(enriched[0]) is False


def test_blocked_scenarios_are_not_near_candidates():
    from screener.models import Check, ScenarioResult

    result = ScenarioResult(
        symbol="ALTUSDT", name="ALT", scenario="LONG C", direction="LONG",
        checks=[Check("prep 1", True, "yes", "yes"), Check("prep 2", True, "yes", "yes"),
                Check("quality", False, "bad", "none")],
        quality_flags=["bad"], status="BLOCKED_QUALITY",
    )
    assert result.near_candidate is False


def test_zero_known_liquidations_mean_no_cascade_not_unknown():
    row = MarketRow(
        symbol="ZEROUSDT_PERP", name="ZERO", price=10, chg24_pct=-2,
        market_cap_usd=1_000_000_000, volume24_usd=100_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-2, oi_chg24_usd=-2_000_000, oi_chg4h_pct=-0.5, oi_chg4h_usd=-500_000,
        oi_vol_ratio=1.0, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.1,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=0, liq_long24_usd=0, liq_short24_usd=0,
        ls_ratio_1d=1.4, futures_imbalance_4h=-5.0, spot_imbalance_4h=-3.0,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    short_a = next(item for item in results if item.scenario.startswith("SHORT A"))
    no_cascade = next(check for check in short_a.checks if check.name == "No cascade")

    assert enriched[0].long_liq_share is None
    assert no_cascade.passed is True



def test_short_d_requires_material_oi_decline_before_liquidation_coverage():
    row = MarketRow(
        symbol="SMALLUSDT_PERP", name="SMALL", price=1.0, chg24_pct=-1.0,
        volume24_usd=100_000_000, oi_usd=50_000_000,
        oi_vol_ratio=0.5, vol_mktcap_ratio=0.5,
        oi_chg24_pct=-0.21, oi_chg24_usd=-109_000,
        oi_chg4h_pct=-0.2, oi_chg4h_usd=-100_000,
        futures_imbalance_4h=-5.0, spot_imbalance_4h=-6.0,
        liq_total24_usd=1_000_000, liq_long24_usd=900_000, liq_short24_usd=100_000,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}}, technical={"weak_bounce_rejection": {"confirmed": True}})
    short_d = next(item for item in results if item.scenario.startswith("SHORT D"))

    assert enriched[0].liquidation_coverage_of_oi_drop_pct is None
    assert short_d.status == "BLOCKED_CAUSALITY"
    assert next(check for check in short_d.checks if check.name == "OI 24H decline material").passed is False


def test_empty_scenario_result_is_not_a_near_candidate():
    from screener.models import ScenarioResult

    result = ScenarioResult(
        symbol="X", name="X", scenario="LONG A", direction="LONG",
        checks=[], quality_flags=[], status="NOT_MET",
    )
    assert result.near_candidate is False
