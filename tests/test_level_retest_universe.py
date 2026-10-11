from engine.level_retest_universe import select_usdt_m_universe


def _ticker(volume: str, price: str = "1.0") -> dict[str, str]:
    return {"quoteVolume": volume, "lastPrice": price}


def test_default_universe_has_no_volume_floor_or_symbol_cap():
    tickers = {
        "BTCUSDT": _ticker("1000000000", "60000"),
        "SMALLUSDT": _ticker("1200", "0.01"),
        "QUIETUSDT": _ticker("0", "2"),
        "BTCUSDC": _ticker("100000000", "60000"),
        "BROKENUSDT": _ticker("NaN", "2"),
        "BADPRICEUSDT": _ticker("500", "0"),
    }

    universe = select_usdt_m_universe(tickers)

    assert [item[0] for item in universe] == ["BTCUSDT", "SMALLUSDT", "QUIETUSDT"]
    assert len(universe) == 3


def test_optional_top_n_and_volume_floor_are_applied_only_when_configured():
    tickers = {
        "AUSDT": _ticker("500"),
        "BUSDT": _ticker("2000"),
        "CUSDT": _ticker("1500"),
    }

    universe = select_usdt_m_universe(
        tickers,
        min_24h_volume_usdt=1000,
        top_n_symbols=1,
    )

    assert [item[0] for item in universe] == ["BUSDT"]
