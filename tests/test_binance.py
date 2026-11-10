from screener.binance import BinanceClient


def test_orderbook_diagnostics_calculate_spread_depth_and_estimated_slippage(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: {
        "lastUpdateId": 1,
        "bids": [["99", "10"], ["98", "10"]],
        "asks": [["101", "10"], ["102", "10"]],
    })

    result = client.fetch_orderbook_diagnostics("BTCUSDT_PERP.A", limit=20, notional_usd=1000)

    assert result["best_bid"] == 99
    assert result["best_ask"] == 101
    assert round(result["spread_bps"], 2) == 200.0
    assert result["bid_depth_1pct_usd"] == 990.0
    assert result["ask_depth_1pct_usd"] == 1010.0
    assert result["market_buy_estimate"]["unfilled_usd"] == 0
    assert result["market_sell_estimate"]["slippage_bps"] >= 0
    assert result["estimated_only"] is True


def test_orderbook_rejects_crossed_market(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: {
        "bids": [["102", "10"]],
        "asks": [["101", "10"]],
    })

    try:
        client.fetch_orderbook_diagnostics("BTCUSDT", limit=5)
    except Exception as exc:
        assert "Crossed/invalid" in str(exc)
    else:
        raise AssertionError("Expected crossed order book to be rejected")


def test_binance_proxy_is_explicit_and_does_not_inherit_runner_proxy(monkeypatch):
    client = BinanceClient(
        "https://fapi.binance.com",
        min_interval_sec=0,
        http_proxy="http://127.0.0.1:18080",
        vpn_enabled=True,
    )
    assert client.session.trust_env is False
    assert client.session.proxies["http"] == "http://127.0.0.1:18080"
    assert client.session.proxies["https"] == "http://127.0.0.1:18080"


def test_binance_vpn_mode_fails_closed_without_proxy():
    from screener.binance import BinanceDataError

    try:
        BinanceClient("https://fapi.binance.com", vpn_enabled=True)
    except BinanceDataError as exc:
        assert "BINANCE_HTTP_PROXY" in str(exc)
    else:
        raise AssertionError("Expected missing explicit proxy to fail")


def _valid_klines(count=21, interval="1h"):
    from datetime import datetime, timezone

    duration = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    boundary = now_ms // duration * duration
    first = boundary - count * duration
    rows = []
    for index in range(count):
        open_time = first + index * duration
        rows.append([
            open_time, "100", "102", "99", "101", "10", open_time + duration - 1,
            "1010", 25, "0", "0", "0",
        ])
    return rows, duration


def test_binance_candles_reject_non_consecutive_hourly_series(monkeypatch):
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    rows, _ = _valid_klines()
    rows.pop(7)
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: rows)
    try:
        client.fetch_candles("BTCUSDT", "1h")
    except BinanceDataError as exc:
        assert "Non-consecutive 1h candles" in str(exc)
    else:
        raise AssertionError("Discontinuous candles must not feed ATR/pivot calculations")


def test_binance_candles_ignore_non_finite_or_impossible_ohlc(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    rows, _ = _valid_klines(count=22)
    bad = list(rows[5])
    bad[2] = "NaN"
    rows[5] = bad
    # The invalid row is discarded, so the resulting gap is rejected rather than
    # letting a structurally discontinuous candle series pass as consecutive.
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: rows)
    from screener.binance import BinanceDataError
    try:
        client.fetch_candles("BTCUSDT", "1h")
    except BinanceDataError as exc:
        assert "Non-consecutive 1h candles" in str(exc)
    else:
        raise AssertionError("A malformed candle must not be silently compressed out of the series")


def test_orderbook_ignores_non_finite_levels_and_rejects_non_finite_notional(monkeypatch):
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: {
        "bids": [["99", "10"], ["inf", "100"], ["98", "nan"]],
        "asks": [["101", "10"], ["102", "inf"]],
    })
    result = client.fetch_orderbook_diagnostics("BTCUSDT", limit=5, notional_usd=100)
    assert result["best_bid"] == 99
    assert result["best_ask"] == 101

    try:
        client.fetch_orderbook_diagnostics("BTCUSDT", limit=5, notional_usd=float("inf"))
    except ValueError as exc:
        assert "finite non-negative" in str(exc)
    else:
        raise AssertionError("Infinite orderbook notional must be rejected")
