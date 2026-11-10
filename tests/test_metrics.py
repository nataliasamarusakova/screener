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


def test_strong_inverse_btc_correlation_is_not_treated_as_independent():
    row = mk("ALTUSDT_PERP", 5.0, 4.0, 1.0, 1.2, 1.0)
    row.btc_corr7d = -0.82
    enriched, _ = analyze_universe([row], {"thresholds": {}})
    assert any("strong (inverse)" in note for note in enriched[0].diagnostic_notes)
