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
    assert long_a.status == "OBSERVATION"


def test_missing_value_is_unknown_not_zero():
    row = MarketRow(symbol="A", name="A", futures_imbalance_4h=None, spot_imbalance_4h=2)
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    results = evaluate_scenarios(enriched[0], stats, {"thresholds": {}})
    long_a = next(item for item in results if item.scenario.startswith("LONG A"))
    assert long_a.checks[0].passed is None


def test_long_a_observation_does_not_require_chart_retest():
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
    assert long_a.status == "OBSERVATION"
    assert "before any entry" in long_a.reason


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


def test_short_e_requires_three_snapshots_and_actual_flow_turn_down():
    row = MarketRow(
        symbol="ALTUSDT_PERP", name="ALT", price=10, chg24_pct=-2,
        market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-1, oi_chg24_usd=-1_000_000, oi_chg4h_pct=0.2, oi_chg4h_usd=200_000,
        oi_vol_ratio=0.5, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.2,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=1_000_000, liq_long24_usd=400_000, liq_short24_usd=600_000,
        ls_ratio_1d=1.5, futures_imbalance_4h=-4, spot_imbalance_4h=-5,
    )
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    history = [
        {"futures_imbalance_4h": 2, "spot_imbalance_4h": 3, "liq_long24_usd": 300_000,
         "liq_short24_usd": 700_000, "oi_chg24_pct": -3},
        {"futures_imbalance_4h": 1, "spot_imbalance_4h": 2, "liq_long24_usd": 500_000,
         "liq_short24_usd": 500_000, "oi_chg24_pct": -1},
        {"futures_imbalance_4h": -4, "spot_imbalance_4h": -5, "liq_long24_usd": 400_000,
         "liq_short24_usd": 600_000, "oi_chg24_pct": -1},
    ]
    technical = {"failed_breakout_rejection": {"confirmed": True}}
    result = evaluate_scenarios(enriched[0], stats, {"thresholds": {}}, history=history, technical=technical)
    short_e = next(item for item in result if item.scenario.startswith("SHORT E"))
    assert short_e.status == "OBSERVATION"
    assert short_e.complete

    result_without_3 = evaluate_scenarios(enriched[0], stats, {"thresholds": {}}, history=history[-2:], technical=technical)
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
    from screener.config import DEFAULT_P1_CONFIG

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
    enriched, stats = analyze_universe([row], DEFAULT_P1_CONFIG)
    result = evaluate_scenarios(enriched[0], stats, DEFAULT_P1_CONFIG)
    long_a = next(item for item in result if item.scenario.startswith("LONG A"))

    assert long_a.passed_count == 6
    assert long_a.total_count == 6
    assert long_a.status == "OBSERVATION_WITH_WARNING"
    assert "qualitative warning only" in long_a.reason
    assert any("changes sign versus current FR" in note for note in enriched[0].diagnostic_notes)
    assert "OI 4H change is small in USD" in enriched[0].diagnostic_notes
