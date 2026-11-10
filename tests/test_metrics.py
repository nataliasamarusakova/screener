from screener.metrics import analyze_universe
from screener.models import MarketRow


def mk(symbol: str, f: float, spot: float | None, oi24: float, oi4: float, chg: float) -> MarketRow:
    return MarketRow(
        symbol=symbol, name=symbol, price=10, chg24_pct=chg, market_cap_usd=1_000_000_000,
        volume24_usd=100_000_000, oi_usd=50_000_000, oi_chg24_pct=oi24,
        oi_chg24_usd=-1_000_000 if oi24 < 0 else 1_000_000, oi_chg4h_pct=oi4,
        oi_chg4h_usd=3_000_000, fr_avg_pct=0.002, fr_oiw_pct=0.003,
        pfr_oiw_pct=0.004, liq_long24_usd=800_000, liq_short24_usd=200_000,
        liq_total24_usd=1_000_000, ls_ratio_1d=1.2, futures_imbalance_4h=f,
        spot_imbalance_4h=spot,
    )


def test_metrics_keep_n_a_and_calculate_derived_fields():
    rows = [mk(f"X{i}", float(i - 5), None if i == 0 else float(5 - i), -1.0 if i < 5 else 1.0, 1.2, 0.4) for i in range(10)]
    rows[0].symbol = "BTCUSDT_PERP.A"
    rows[0].chg24_pct = -1.0
    enriched, stats = analyze_universe(rows, {"thresholds": {"strong_flow_quantile": 0.15}})
    assert stats.row_count == 10
    assert stats.s_count == 9
    assert stats.oi24_count == 10
    assert stats.price_count == 10
    assert stats.median_f is not None
    assert enriched[0].futures_buy_share_pct == (100 + rows[0].futures_imbalance_4h) / 2
    assert enriched[0].spot_buy_share_pct is None
    assert "SPOT_IMBALANCE_NA" in enriched[0].quality_flags
    assert enriched[0].long_liq_share == 0.8
    assert enriched[0].liq_imbalance == -0.6
    assert stats.warnings


def test_unavailable_price_is_a_quality_flag_not_a_dropped_row():
    row = mk("ZERO", 1.0, 2.0, -1.0, 1.5, 1.0)
    row.price = None
    enriched, stats = analyze_universe([row], {"thresholds": {}})
    assert len(enriched) == 1
    assert "PRICE_UNAVAILABLE_OR_ROUNDED_ZERO" in enriched[0].quality_flags


def test_equal_flow_values_receive_shared_average_rank():
    rows = [mk(f"X{i}", 10.0 if i in {0, 1, 2} else -1.0, 5.0 if i in {0, 1, 2} else -2.0, -1.0, 1.2, 1.0) for i in range(20)]
    enriched, _ = analyze_universe(rows, {"thresholds": {"strong_flow_quantile": 0.15}})
    assert enriched[0].f_rank_desc == enriched[1].f_rank_desc == enriched[2].f_rank_desc == 2.0
    assert all(item.f_strong_buy for item in enriched[:3])
    assert all(item.s_strong_buy for item in enriched[:3])
    assert not any(item.f_strong_buy for item in enriched[3:])


def test_top_15_percent_count_matches_p1_reference_universes():
    # The same rounding rule must yield 11 rows for both 75 total Futures
    # values and 72 non-null Spot values (75*0.15=11.25; 72*0.15=10.8).
    from screener.metrics import _rank_tail_count

    assert _rank_tail_count(75, 0.15) == 11
    assert _rank_tail_count(72, 0.15) == 11


def test_rank_twelve_is_not_strong_for_75_row_p1_universe():
    rows = []
    for i in range(75):
        symbol = "PONS" if i == 11 else f"ALT{i:02d}"
        spot = None if i in {72, 73, 74} else float(100 - i)
        rows.append(mk(symbol, float(100 - i), spot, 1.0, 1.2, 2.0))
    enriched, _ = analyze_universe(rows, {"thresholds": {"strong_flow_quantile": 0.15}})
    pons = next(item for item in enriched if item.symbol == "PONS")
    assert pons.f_rank_desc == 12
    assert pons.s_rank_desc == 12
    assert pons.f_strong_buy is False
    assert pons.s_strong_buy is False


def test_p1_divergences_and_funding_sign_reversal_are_diagnostic_notes():
    row = mk("GRASS", 7.6293, -4.2696, -1.0, 1.5, 4.92)
    row.fr_oiw_pct = 0.0096
    row.pfr_oiw_pct = -0.0049
    row.btc_corr7d = 0.88
    enriched, _ = analyze_universe([row], {"thresholds": {"strong_flow_quantile": 0.15}})
    notes = " | ".join(enriched[0].diagnostic_notes)
    assert "opposite signs" in notes
    assert "price/Spot divergence" in notes
    assert "Price 24H up while OI 24H down" in notes
    assert "changes sign versus current FR" in notes
    assert "BTC Correlation 7D > 0.7" in notes
    assert "SPOT_IMBALANCE_NA" not in enriched[0].quality_flags


def test_bitcoin_self_correlation_is_not_reported_as_altcoin_exposure():
    row = mk("BTCUSDT_PERP.A", 1.0, 1.0, 1.0, 1.0, 1.0)
    row.btc_corr7d = 1.0
    enriched, _ = analyze_universe([row], {"thresholds": {}})
    assert not any("BTC Correlation 7D > 0.7" in note for note in enriched[0].diagnostic_notes)


def test_strk_reference_report_derived_metrics_match_p1_formulas():
    btc = mk("BTCUSDT_PERP.A", 0.0, 0.0, 0.08, 0.33, 0.32)
    alt = mk("STRKUSDT_PERP.A", 21.2674, 3.4397, -0.07, 1.22, 2.91)
    btc.oi_chg24_usd = 21_600_000
    btc.oi_chg4h_usd = 86_800_000
    btc.fr_oiw_pct = 0.0084
    btc.pfr_oiw_pct = 0.0084
    alt.oi_chg24_usd = -40_600
    alt.oi_chg4h_usd = 671_700
    alt.fr_avg_pct = 0.0043
    alt.fr_oiw_pct = 0.0093
    alt.pfr_oiw_pct = 0.0098
    alt.liq_long24_usd = 310_000
    alt.liq_short24_usd = 690_000
    alt.liq_total24_usd = 1_000_000
    enriched, _ = analyze_universe([btc, alt], {"thresholds": {}})
    strk = next(item for item in enriched if item.symbol.startswith("STRK"))

    assert abs(strk.futures_buy_share_pct - 60.6337) < 0.0001
    assert abs(strk.spot_buy_share_pct - 51.71985) < 0.0001
    assert abs(strk.spread - (-17.8277)) < 0.0001
    assert abs(strk.delta_funding_oiw_pp - 0.0005) < 1e-9
    assert abs(strk.fr_venue_dispersion_pp - (-0.0050)) < 1e-9
    assert abs(strk.rel_strength_vs_btc_pp - 2.59) < 1e-9
    assert abs(strk.long_liq_share - 0.31) < 1e-9
    assert abs(strk.liq_imbalance - 0.38) < 1e-9
