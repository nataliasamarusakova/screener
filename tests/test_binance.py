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


def test_orderbook_maps_unsupported_depth_limit_and_uses_correct_sell_slippage(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    captured = {}
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")

    def fake_get_json(_path, params):
        captured.update(params)
        return {
            "bids": [["100", "1"], ["90", "10"]],
            "asks": [["101", "10"]],
        }

    monkeypatch.setattr(client, "_get_json", fake_get_json)
    result = client.fetch_orderbook_diagnostics("BTCUSDT", limit=99, notional_usd=500)

    expected_average = 500 / (1 + 400 / 90)
    expected_slippage = (100 - expected_average) / 100 * 10_000
    assert captured["limit"] == 100
    assert abs(result["market_sell_estimate"]["slippage_bps"] - expected_slippage) < 1e-9

    # Mapping rounds upward, so the API response cannot have fewer levels
    # than the caller requested (e.g. 75 -> 100, not 50).
    client.fetch_orderbook_diagnostics("BTCUSDT", limit=75, notional_usd=0)
    assert captured["limit"] == 100


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


def _valid_closed_klines(interval="1h", count=21, *, latest_age_intervals=0):
    from datetime import datetime, timezone

    durations = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}
    duration = durations[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_boundary = now_ms // duration * duration
    last_open = current_boundary - (latest_age_intervals + 1) * duration
    first_open = last_open - (count - 1) * duration
    return [
        [
            first_open + index * duration, "100", "102", "99", "101", "10",
            first_open + index * duration + duration - 1, "1010", 25, "0", "0", "0",
        ]
        for index in range(count)
    ], duration


def test_binance_candles_reject_non_consecutive_hourly_series(monkeypatch):
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    rows, _ = _valid_closed_klines("1h")
    rows.pop(7)
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: rows)
    try:
        client.fetch_candles("BTCUSDT", "1h")
    except BinanceDataError as exc:
        assert "consecutive valid closed 1h candles" in str(exc)
    else:
        raise AssertionError("Discontinuous candles must not feed ATR/pivot calculations")


def test_binance_candles_keep_recent_contiguous_tail_after_an_old_gap(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    rows, _ = _valid_closed_klines("1h", count=100)
    rows.pop(5)
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: rows)

    candles = client.fetch_candles("BTCUSDT", "1h")

    assert len(candles) == 94
    assert all(b.open_time_ms - a.open_time_ms == 60 * 60_000 for a, b in zip(candles, candles[1:]))


def test_binance_resolves_multiplier_contract_alias_only_when_unambiguous(monkeypatch):
    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "_load_symbols", lambda: {"1000PEPEUSDT", "BTCUSDT"})
    assert client.resolve_symbol("PEPEUSDT_PERP.A") == "1000PEPEUSDT"
    assert client.resolve_symbol("BTCUSD_PERPETUAL") == "BTCUSDT"

    monkeypatch.setattr(client, "_load_symbols", lambda: {"1000PEPEUSDT", "10000PEPEUSDT"})
    assert client.resolve_symbol("PEPEUSDT_PERP.A") is None


def test_binance_candles_reject_malformed_ohlc_and_stale_series(monkeypatch):
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    rows, _ = _valid_closed_klines("1h", count=22)
    bad = list(rows[5])
    bad[2] = "NaN"
    rows[5] = bad
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: rows)
    try:
        client.fetch_candles("BTCUSDT", "1h")
    except BinanceDataError as exc:
        assert "consecutive valid closed 1h candles" in str(exc)
    else:
        raise AssertionError("A malformed candle must not be silently compressed out of the series")

    stale_rows, _ = _valid_closed_klines("1h", latest_age_intervals=2)
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: stale_rows)
    try:
        client.fetch_candles("BTCUSDT", "1h")
    except BinanceDataError as exc:
        assert "Stale 1h candles" in str(exc)
    else:
        raise AssertionError("Stale chart data must fail closed")


def test_binance_orderbook_ignores_non_finite_levels(monkeypatch):
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    monkeypatch.setattr(client, "resolve_symbol", lambda _: "BTCUSDT")
    monkeypatch.setattr(client, "_get_json", lambda *_args, **_kwargs: {
        "bids": [["99", "10"], ["inf", "100"], ["98", "nan"]],
        "asks": [["101", "10"], ["102", "inf"]],
    })
    result = client.fetch_orderbook_diagnostics("BTCUSDT", limit=5, notional_usd=100)
    assert result["best_bid"] == 99.0
    assert result["best_ask"] == 101.0
    assert result["estimated_only"] is True

    try:
        client.fetch_orderbook_diagnostics("BTCUSDT", limit=5, notional_usd=float("inf"))
    except BinanceDataError as exc:
        assert "finite non-negative USD" in str(exc)
    else:
        raise AssertionError("Infinite execution notional must be rejected")


def test_binance_http_418_stops_without_retry(monkeypatch):
    import pytest
    from screener.binance import BinanceDataError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    calls = []

    class Response:
        status_code = 418
        headers = {}

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    with pytest.raises(BinanceDataError, match="HTTP 418"):
        client._get_json("/fapi/v1/klines")
    assert calls == [1]


def test_binance_retry_after_is_honoured_without_thirty_second_cap(monkeypatch):
    import screener.binance as binance_module

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    responses = []

    class Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self.payload = payload or {"ok": True}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise binance_module.requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            return self.payload

    responses.extend([
        Response(429, {"code": -1003}, {"Retry-After": "120"}),
        Response(200, {"serverTime": 1}),
    ])
    calls = []
    sleeps = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(binance_module.time, "sleep", lambda delay: sleeps.append(delay))

    assert client._get_json("/fapi/v1/time") == {"serverTime": 1}
    assert len(calls) == 2
    assert sleeps == [120.0]



def test_binance_retry_after_http_date_is_parsed(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    import screener.binance as binance_module

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=90)
    header = format_datetime(retry_at, usegmt=True)

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {"Retry-After": header} if status_code == 429 else {}

        def json(self):
            return {"serverTime": 1}

        def raise_for_status(self):
            return None

    responses = [Response(429), Response(200)]
    sleeps = []
    monkeypatch.setattr(client.session, "get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(binance_module.time, "sleep", lambda delay: sleeps.append(delay))

    assert client._get_json("/fapi/v1/time") == {"serverTime": 1}
    assert len(sleeps) == 1
    assert 88 <= sleeps[0] <= 90

def test_binance_http_418_permanently_blocks_client_and_preserves_retry_after(monkeypatch):
    import pytest
    import screener.binance as binance_module
    from screener.binance import BinanceIPBanError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    calls = []

    class Response:
        status_code = 418
        headers = {"Retry-After": "300"}

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    with pytest.raises(BinanceIPBanError, match="Retry-After=300s"):
        client._get_json("/fapi/v1/klines")
    with pytest.raises(BinanceIPBanError, match="refusing all further requests"):
        client._get_json("/fapi/v1/exchangeInfo")
    assert calls == [1]


def test_binance_ip_ban_propagates_out_of_technical_symbol_pipeline():
    import pytest
    from types import SimpleNamespace
    from screener.binance import BinanceIPBanError
    from screener.service import _technical_for_symbol

    class BannedClient:
        def __init__(self):
            self.intervals = []

        def resolve_symbol(self, _symbol):
            return "BTCUSDT"

        def fetch_candles(self, _symbol, interval, _limit):
            self.intervals.append(interval)
            raise BinanceIPBanError("HTTP 418")

    client = BannedClient()
    settings = SimpleNamespace(
        p1={"structure": {}, "scanner": {"fetch_15m_context": False}},
        binance_kline_limit=300,
        binance_fetch_orderbook=False,
        trading_fee_bps_per_side=None,
    )

    with pytest.raises(BinanceIPBanError, match="HTTP 418"):
        _technical_for_symbol(client, "BTCUSDT_PERP", settings)
    assert client.intervals == ["4h"]


def test_binance_repeated_http_429_stops_future_requests_after_honoured_backoff(monkeypatch):
    import pytest
    import screener.binance as binance_module
    from screener.binance import BinanceRateLimitError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    calls = []
    sleeps = []

    class Response:
        status_code = 429
        headers = {"Retry-After": "75"}

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(binance_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(BinanceRateLimitError, match="HTTP 429 on all three attempts"):
        client._get_json("/fapi/v1/klines")
    with pytest.raises(BinanceRateLimitError, match="refusing further requests"):
        client._get_json("/fapi/v1/exchangeInfo")

    assert calls == [1, 1, 1]
    assert sleeps == [75.0, 75.0]


def test_binance_http_403_stops_the_current_scan_without_more_requests(monkeypatch):
    import pytest
    from screener.binance import BinanceWAFBlockError

    client = BinanceClient("https://fapi.binance.com", min_interval_sec=0)
    calls = []

    class Response:
        status_code = 403
        headers = {}

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    with pytest.raises(BinanceWAFBlockError, match="HTTP 403"):
        client._get_json("/fapi/v1/klines")
    with pytest.raises(BinanceWAFBlockError, match="refusing all further requests"):
        client._get_json("/fapi/v1/exchangeInfo")
    assert calls == [1]


def test_all_binance_global_blocks_propagate_out_of_symbol_pipeline():
    import pytest
    from types import SimpleNamespace
    from screener.binance import BinanceRateLimitError, BinanceWAFBlockError
    from screener.service import _technical_for_symbol

    settings = SimpleNamespace(
        p1={"structure": {}, "scanner": {"fetch_15m_context": False}},
        binance_kline_limit=300,
        binance_fetch_orderbook=False,
        trading_fee_bps_per_side=None,
    )
    for failure in (BinanceWAFBlockError("HTTP 403"), BinanceRateLimitError("HTTP 429")):
        class BlockedClient:
            intervals = []

            def resolve_symbol(self, _symbol):
                return "BTCUSDT"

            def fetch_candles(self, _symbol, interval, _limit):
                self.intervals.append(interval)
                raise failure

        client = BlockedClient()
        with pytest.raises(type(failure)):
            _technical_for_symbol(client, "BTCUSDT_PERP", settings)
        assert client.intervals == ["4h"]


def test_run_scan_aborts_before_later_symbol_after_binance_ip_ban(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import pytest
    import screener.service as service_module
    from screener.binance import BinanceIPBanError
    from screener.coinalyze import ScrapeResult
    from screener.config import DEFAULT_P1_CONFIG
    from screener.models import MarketRow

    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        dry_run=False,
        p1={key: value.copy() for key, value in DEFAULT_P1_CONFIG.items()},
        coinalyze_url="https://coinalyze.net/",
        coinalyze_p_sid="",
        coinalyze_chat_sid="",
        coinalyze_max_pages=20,
        coinalyze_headless=True,
        coinalyze_timeout_ms=30_000,
        binance_base_url="https://fapi.binance.com",
        binance_request_interval_sec=0.0,
        binance_http_proxy="",
        binance_vpn_enabled=False,
        binance_kline_limit=300,
        binance_fetch_orderbook=False,
        telegram_bot_token="token",
        telegram_chat_ids=["chat-1"],
    )
    scraped = ScrapeResult(
        rows=[MarketRow(symbol="AAAUSDT_PERP", name="AAA")],
        fetched_at=datetime.now(timezone.utc),
        contract_type="PERPETUAL",
        pages_loaded=1,
        headers_found=[],
        missing_columns=[],
        source_url=settings.coinalyze_url,
    )
    monkeypatch.setattr(service_module, "fetch_table", lambda **_kwargs: scraped)
    monkeypatch.setattr(
        service_module,
        "_preliminary_candidate_symbols",
        lambda *_args, **_kwargs: {"AAAUSDT_PERP", "ZZZUSDT_PERP"},
    )
    calls = []

    def abort_on_first_symbol(_client, symbol, _settings):
        calls.append(symbol)
        raise BinanceIPBanError("HTTP 418")

    monkeypatch.setattr(service_module, "_technical_for_symbol", abort_on_first_symbol)

    with pytest.raises(BinanceIPBanError, match="HTTP 418"):
        service_module.run_scan_once(settings, dry_run=False)

    assert calls == ["AAAUSDT_PERP"]
