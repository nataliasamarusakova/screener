from screener.coinalyze import (
    CoinalyzeError,
    RetryableCoinalyzeError,
    _page_content_with_retry,
    build_session_cookies,
    detect_contract_type,
    fetch_table,
    _pagination_urls,
    _pagination_next_url,
    parse_number,
    parse_table,
)


FIXTURE_HTML = """
<html><body><h1>Perpetual contracts</h1><table><thead><tr>
<th></th><th>Coin</th><th>Price</th><th>Chg 24H</th><th>Mkt Cap</th><th>FDV</th><th>Vol 24H</th><th>Open Interest</th>
<th>OI Chg 24H</th><th>OI Chg 24H</th><th>OI Chg 4H</th><th>OI Chg 4H</th><th>OI Share</th>
<th>OI / VOL24H</th><th>OI / MktCap</th><th>VOL24H / MktCap</th><th>FR AVG</th><th>PFR AVG</th><th>FR AVG OI_W</th><th>PFR AVG OI_W</th>
<th>Liqs. 24H</th><th>Short Liqs. 24H</th><th>Long Liqs. 24H</th><th>Liqs. 24H / Vol 24H</th><th>Liqs. 24H / OI</th>
<th>L/S Ratio (1H)</th><th>Long (1H)</th><th>Short (1H)</th><th>L/S Ratio (1D)</th><th>Long (1D)</th><th>Short (1D)</th>
<th>BTC Corr. 7D</th><th>FUTURES IMBALANCE 4H</th><th>SPOT IMBALANCE 4H</th>
</tr></thead><tbody>
<tr data-coin="BTCUSDT_PERP.A">
<td></td><td><span>Bitcoin</span></td><td>$74,790.08</td><td>-1.47%</td><td>$1.5t</td><td>$1.5t</td><td>$50.2b</td><td>$26.5b</td><td>-0.56%</td><td>$-147.9m</td><td>+1.58%</td><td>$411.1m</td><td>47.57%</td>
<td>0.527</td><td>0.018</td><td>0.033</td><td>+0.0080%</td><td>+0.0072%</td><td>+0.0086%</td><td>+0.0090%</td>
<td>$208.3m</td><td>$19.9m</td><td>$188.4m</td><td>0.0041</td><td>0.0079</td>
<td>1.64</td><td>62.06%</td><td>37.90%</td><td>1.61</td><td>61.70%</td><td>38.27%</td><td>+1.00</td><td>4.9580</td><td>9.5607</td>
</tr>
<tr data-coin="ETHUSDT_PERP.A">
<td></td><td><span>Ethereum</span></td><td>$2,124.33</td><td>-4.97%</td><td>$260.0b</td><td>$260.0b</td><td>$34.4b</td><td>$13.7b</td><td>-8.50%</td><td>$-1.3b</td><td>+1.03%</td><td>$138.7m</td><td>24.55%</td>
<td>0.397</td><td>0.053</td><td>0.132</td><td>+0.0000%</td><td>+0.0007%</td><td>-0.0021%</td><td>+0.0031%</td>
<td>$318.7m</td><td>$24.6m</td><td>$294.1m</td><td>0.0093</td><td>0.0233</td>
<td>1.95</td><td>66.09%</td><td>33.85%</td><td>1.90</td><td>65.44%</td><td>34.50%</td><td>0.82</td><td>4.6794</td><td>8.5289</td>
</tr>
</tbody></table></body></html>
"""
TEST_TABLE_URL = "https://coinalyze.net/?columns=test-columns&filter=test-filter&contract_type=perpetual"


def test_parse_number_keeps_na_missing_and_compact_usd():
    assert parse_number("n/a") is None
    assert parse_number("—") is None
    assert parse_number("$-147.9m") == -147_900_000
    assert parse_number("+0.0080%") == 0.008
    assert parse_number("1.5t") == 1_500_000_000_000


def test_parse_table_maps_duplicate_oi_change_columns_and_custom_imbalance():
    rows, headers, missing, missing_required = parse_table(FIXTURE_HTML)
    assert len(rows) == 2
    assert not missing_required
    assert missing == []
    btc = rows[0]
    assert btc.symbol == "BTCUSDT_PERP.A"
    assert btc.name == "Bitcoin"
    assert btc.oi_chg24_pct == -0.56
    assert btc.oi_chg24_usd == -147_900_000
    assert btc.oi_chg4h_pct == 1.58
    assert btc.oi_chg4h_usd == 411_100_000
    assert btc.futures_imbalance_4h == 4.9580
    assert btc.spot_imbalance_4h == 9.5607
    assert btc.btc_corr7d == 1.0
    assert btc.ls_ratio_1h == 1.64
    assert btc.long_accounts_1h_pct == 62.06
    assert btc.short_accounts_1h_pct == 37.90
    assert btc.long_accounts_1d_pct == 61.70
    assert btc.short_accounts_1d_pct == 38.27
    assert btc.liq_vol_ratio == 0.0041
    assert btc.liq_oi_ratio == 0.0079
    assert "FUTURES IMBALANCE 4H" in headers


def test_parse_table_keeps_leading_row_header_in_column_alignment():
    html = FIXTURE_HTML.replace('<td></td><td><span>Bitcoin</span>', '<th scope="row"></th><td><span>Bitcoin</span>', 1)

    rows, _, _, missing_required = parse_table(html)

    assert not missing_required
    assert rows[0].name == "Bitcoin"
    assert rows[0].price == 74_790.08
    assert rows[0].chg24_pct == -1.47
    assert rows[0].oi_usd == 26_500_000_000
    assert rows[0].futures_imbalance_4h == 4.958


def test_zero_rounded_price_does_not_drop_the_market_row():
    html = FIXTURE_HTML.replace("$74,790.08", "$0.0000")
    rows, _, _, _ = parse_table(html)
    assert len(rows) == 2
    assert rows[0].price is None
    assert rows[0].raw["price_rounded_or_invalid_zero"] is True


def test_negative_volume_capitalization_and_ls_ratios_are_rejected():
    html = FIXTURE_HTML.replace("$1.5t</td><td>$1.5t", "$-1.5t</td><td>$-1.5t", 1)
    html = html.replace("$50.2b</td>", "$-50.2b</td>", 1)
    html = html.replace("<td>1.64</td>", "<td>-1.64</td>", 1)
    html = html.replace("<td>1.61</td>", "<td>-1.61</td>", 1)

    rows, _, _, _ = parse_table(html)
    row = rows[0]

    assert row.market_cap_usd is None
    assert row.fdv_usd is None
    assert row.volume24_usd is None
    assert row.ls_ratio_1h is None
    assert row.ls_ratio_1d is None
    assert {"market_cap_usd", "fdv_usd", "volume24_usd", "ls_ratio_1h", "ls_ratio_1d"}.issubset(
        set(row.raw["invalid_market_fields"])
    )


def test_parser_accepts_data_symbol_attribute_fallback():
    html = FIXTURE_HTML.replace("data-coin=", "data-symbol=")
    rows, _, _, _ = parse_table(html)
    assert {row.symbol for row in rows} == {"BTCUSDT_PERP.A", "ETHUSDT_PERP.A"}


def test_explicit_saved_url_contract_type_wins_over_generic_navigation_text():
    html = "<html><body><nav>All Contracts | Perpetual contracts</nav><table></table></body></html>"
    url = (
        "https://coinalyze.net/?columns=x&filter=y&contract_type=perpetual"
    )
    assert detect_contract_type(html, url) == "PERPETUAL"


def test_explicit_all_contracts_url_is_not_misclassified_as_perpetual():
    html = "<html><body><h1>All Contracts</h1></body></html>"
    assert detect_contract_type(html, "https://coinalyze.net/?contract_type=all") == "ALL_CONTRACTS"


def test_p1_required_market_quality_columns_are_reported_when_missing():
    html = FIXTURE_HTML.replace("<th>VOL24H / MktCap</th>", "<th>Turnover</th>")
    _, _, missing, missing_required = parse_table(html)
    assert "vol_mktcap_ratio" in missing
    assert "vol_mktcap_ratio" in missing_required


def test_session_cookies_match_existing_divergence_main_names_and_scope():
    cookies = build_session_cookies("p-cookie-value", "chat-cookie-value")
    assert [item["name"] for item in cookies] == ["p_sid", "chat_sid", "cookies_accepted"]
    assert cookies[0]["value"] == "p-cookie-value"
    assert cookies[1]["value"] == "chat-cookie-value"
    assert all(item["domain"] == "coinalyze.net" and item["path"] == "/" for item in cookies)


def test_session_cookies_fail_closed_when_either_secret_is_missing():
    try:
        build_session_cookies("only-one", "")
    except CoinalyzeError as exc:
        assert "COINALYZE_CHAT_SID" in str(exc)
    else:
        raise AssertionError("Expected missing-cookie configuration error")


def test_pagination_keeps_user_selected_columns_filter_and_contract_type():
    html = (
        '<html><body><div class="pagination"><a href="?page=2">2</a>'
        '<a href="#page=3">3</a></div></body></html>'
    )
    base = (
        "https://coinalyze.net/?columns=custom-columns&filter=my-filter&contract_type=perpetual"
    )
    urls = _pagination_urls(html, base)
    assert len(urls) == 2
    assert "columns=custom-columns" in urls[1]
    assert "filter=my-filter" in urls[1]
    assert "contract_type=perpetual" in urls[1]
    assert "page=2" in urls[1]


def test_scanner_requires_saved_perpetual_url_as_runtime_secret():
    from screener.coinalyze import CoinalyzeError

    try:
        fetch_table(url="https://coinalyze.net/", p_sid="p", chat_sid="c")
    except CoinalyzeError as exc:
        assert "contract_type=perpetual" in str(exc)
    else:
        raise AssertionError("Expected an explicit Perpetual URL requirement")


def test_fetch_table_uses_session_cookies_and_reads_the_saved_page(monkeypatch):
    import sys
    import types

    html = FIXTURE_HTML
    captured = {}

    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector
            self.first = self

        def count(self):
            return 2 if self.selector == "tbody tr" else 0

    class FakePage:
        def goto(self, target_url, *_args, **_kwargs):
            captured["url"] = target_url
            captured["wait_until"] = _kwargs.get("wait_until")

        def wait_for_timeout(self, *_args):
            pass

        def wait_for_selector(self, *_args, **_kwargs):
            pass

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, *_args):
            pass

        def content(self):
            return html

    class FakeContext:
        def add_cookies(self, cookies):
            captured["cookies"] = cookies

        def new_page(self):
            return FakePage()

        def close(self):
            pass

    class FakeBrowser:
        def new_context(self, **_kwargs):
            return FakeContext()

        def close(self):
            pass

    class FakePlaywright:
        chromium = types.SimpleNamespace(launch=lambda **_kwargs: FakeBrowser())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: FakePlaywright()
    playwright_pkg = types.ModuleType("playwright")
    playwright_pkg.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", playwright_pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr("screener.coinalyze._apply_stealth", lambda _page: None)

    result = fetch_table(
        url=TEST_TABLE_URL,
        p_sid="session-p",
        chat_sid="session-chat",
        max_pages=5,
    )

    assert result.contract_type == "PERPETUAL"
    assert result.missing_required_columns == []
    assert result.missing_columns == []
    assert captured["url"] == TEST_TABLE_URL
    assert captured["wait_until"] == "commit"
    assert {row.symbol for row in result.rows} == {"BTCUSDT_PERP.A", "ETHUSDT_PERP.A"}
    assert [cookie["name"] for cookie in captured["cookies"]] == [
        "p_sid", "chat_sid", "cookies_accepted"
    ]


def test_fetch_table_loads_all_contiguous_pagination_urls_from_first_page(monkeypatch):
    import sys
    import types

    page_urls = [
        TEST_TABLE_URL,
        TEST_TABLE_URL + "&page=2",
        TEST_TABLE_URL + "&page=3",
    ]
    page_rows = [
        ("BTCUSDT_PERP.A", "ETHUSDT_PERP.A"),
        ("BNBUSDT_PERP.A", "SOLUSDT_PERP.A"),
        ("XRPUSDT_PERP.A", "DOGEUSDT_PERP.A"),
    ]
    pagination = (
        '<ul class="pagination">'
        '<li><a href="?page=1">1</a></li>'
        '<li><a href="?page=2">2</a></li>'
        '<li><a href="?page=3">3</a></li>'
        '</ul>'
    )
    html_by_url = {}
    for url, symbols in zip(page_urls, page_rows):
        html = FIXTURE_HTML
        replacements = {
            "BTCUSDT_PERP.A": symbols[0],
            "ETHUSDT_PERP.A": symbols[1],
            "Bitcoin": symbols[0].replace("USDT_PERP.A", ""),
            "Ethereum": symbols[1].replace("USDT_PERP.A", ""),
        }
        for old, new in replacements.items():
            html = html.replace(old, new)
        html = html.replace("</body>", pagination + "</body>")
        html_by_url[url] = html

    captured = {"visited": []}

    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector
            self.first = self

        def count(self):
            if self.selector == "tbody tr":
                return 2
            if self.selector == ".pagination":
                return 1
            return 0

        def locator(self, selector):
            return FakeLocator(selector)

    class FakePage:
        def __init__(self):
            self.url = ""

        def goto(self, target_url, *_args, **_kwargs):
            self.url = target_url
            captured["visited"].append(target_url)

        def wait_for_timeout(self, *_args):
            pass

        def wait_for_selector(self, *_args, **_kwargs):
            pass

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, *_args):
            pass

        def content(self):
            from urllib.parse import parse_qs, urlparse
            page_number = parse_qs(urlparse(self.url).query).get("page", ["1"])[0]
            canonical_test_url = TEST_TABLE_URL if page_number == "1" else TEST_TABLE_URL + f"&page={page_number}"
            return html_by_url[canonical_test_url]

    class FakeContext:
        def add_cookies(self, _cookies):
            pass

        def new_page(self):
            return FakePage()

        def close(self):
            pass

    class FakeBrowser:
        def new_context(self, **_kwargs):
            return FakeContext()

        def close(self):
            pass

    class FakePlaywright:
        chromium = types.SimpleNamespace(launch=lambda **_kwargs: FakeBrowser())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: FakePlaywright()
    playwright_pkg = types.ModuleType("playwright")
    playwright_pkg.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", playwright_pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr("screener.coinalyze._apply_stealth", lambda _page: None)

    result = fetch_table(
        url=TEST_TABLE_URL,
        p_sid="session-p",
        chat_sid="session-chat",
        max_pages=5,
    )

    from urllib.parse import parse_qs, urlparse
    visited_pages = [parse_qs(urlparse(item).query).get("page", ["1"])[0] for item in captured["visited"]]
    assert visited_pages == ["1", "2", "3"]
    assert all("columns=test-columns" in item and "filter=test-filter" in item for item in captured["visited"])
    assert result.pages_loaded == 3
    assert len(result.rows) == 6
    assert {row.symbol for row in result.rows} == {
        "BTCUSDT_PERP.A", "ETHUSDT_PERP.A",
        "BNBUSDT_PERP.A", "SOLUSDT_PERP.A",
        "XRPUSDT_PERP.A", "DOGEUSDT_PERP.A",
    }

    # A known page count beyond the configured cap must fail closed before
    # returning a partial market universe.
    import pytest
    from screener.coinalyze import IncompleteUniverseError
    captured["visited"].clear()
    with pytest.raises(IncompleteUniverseError, match="превышает лимит"):
        fetch_table(
            url=TEST_TABLE_URL,
            p_sid="session-p",
            chat_sid="session-chat",
            max_pages=2,
        )
    assert len(captured["visited"]) == 1


def test_coinalyze_navigation_retries_transient_timeouts_with_exponential_backoff(monkeypatch):
    import screener.coinalyze as coinalyze_module

    calls = []
    delays = []

    def flaky_page_content(_page, _url, _timeout_ms):
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("navigation timed out")
        return FIXTURE_HTML

    monkeypatch.setattr(coinalyze_module, "_page_content", flaky_page_content)
    monkeypatch.setattr(coinalyze_module.time, "sleep", delays.append)

    html = _page_content_with_retry(
        object(), TEST_TABLE_URL, timeout_ms=45_000, attempts=3, retry_backoff_sec=2.0
    )

    assert html == FIXTURE_HTML
    assert len(calls) == 3
    assert delays == [2.0, 4.0]


def test_coinalyze_navigation_does_not_retry_unexpected_programming_errors(monkeypatch):
    import screener.coinalyze as coinalyze_module

    calls = []

    def broken_page(_page, _url, _timeout_ms):
        calls.append(1)
        raise ValueError("unexpected parser bug")

    monkeypatch.setattr(coinalyze_module, "_page_content", broken_page)
    monkeypatch.setattr(coinalyze_module.time, "sleep", lambda _delay: None)

    try:
        _page_content_with_retry(object(), TEST_TABLE_URL, timeout_ms=45_000, attempts=4)
    except ValueError as exc:
        assert "parser bug" in str(exc)
    else:
        raise AssertionError("Unexpected programming errors must not be retried")

    assert len(calls) == 1


def test_coinalyze_navigation_does_not_retry_permanent_authentication_error(monkeypatch):
    import screener.coinalyze as coinalyze_module

    calls = []

    def unauthenticated_page(_page, _url, _timeout_ms):
        calls.append(1)
        raise CoinalyzeError("Coinalyze session is not authenticated")

    monkeypatch.setattr(coinalyze_module, "_page_content", unauthenticated_page)
    monkeypatch.setattr(coinalyze_module.time, "sleep", lambda _delay: None)

    try:
        _page_content_with_retry(object(), TEST_TABLE_URL, timeout_ms=45_000, attempts=4)
    except CoinalyzeError as exc:
        assert "not authenticated" in str(exc)
    else:
        raise AssertionError("Expected a permanent authentication error")

    assert len(calls) == 1


def test_coinalyze_navigation_fails_closed_after_all_retry_attempts(monkeypatch):
    import screener.coinalyze as coinalyze_module

    calls = []

    def always_timeout(_page, _url, _timeout_ms):
        calls.append(1)
        raise TimeoutError("navigation timed out")

    monkeypatch.setattr(coinalyze_module, "_page_content", always_timeout)
    monkeypatch.setattr(coinalyze_module.time, "sleep", lambda _delay: None)

    try:
        _page_content_with_retry(
            object(), TEST_TABLE_URL, timeout_ms=12_345, attempts=3, retry_backoff_sec=0
        )
    except CoinalyzeError as exc:
        assert "после 3 попыток" in str(exc)
        assert "timeout=12345ms" in str(exc)
    else:
        raise AssertionError("A failed scrape must not return partial or stale market data")

    assert len(calls) == 3


def test_scan_marks_failed_and_never_analyzes_old_data_when_coinalyze_is_unavailable(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pytest
    import screener.service as service_module

    data_dir = tmp_path / "data"
    settings = SimpleNamespace(
        data_dir=data_dir,
        dry_run=False,
        coinalyze_url=TEST_TABLE_URL,
        coinalyze_p_sid="p-session",
        coinalyze_chat_sid="chat-session",
        coinalyze_max_pages=20,
        coinalyze_headless=True,
        coinalyze_timeout_ms=45_000,
    )

    def unavailable_source(**_kwargs):
        raise CoinalyzeError("Coinalyze unavailable after retries")

    monkeypatch.setattr(service_module, "fetch_table", unavailable_source)

    with pytest.raises(CoinalyzeError, match="unavailable after retries"):
        service_module.run_scan_once(settings, dry_run=False)

    run_records = [json.loads(line) for line in (data_dir / "runs.jsonl").read_text().splitlines()]
    assert [record["record_type"] for record in run_records] == ["run_started", "run_finished"]
    assert run_records[-1]["status"] == "FAILED"
    assert run_records[-1]["row_count"] == 0
    assert (data_dir / "notifications.jsonl").read_text() == ""


def test_coinalyze_table_readiness_failure_is_retryable_but_login_is_not():
    class FakeLocator:
        def __init__(self, text):
            self.text = text

        def inner_text(self, **_kwargs):
            return self.text

    class FakePage:
        def __init__(self, body_text):
            self.body_text = body_text

        def goto(self, *_args, **kwargs):
            assert kwargs["wait_until"] == "commit"

        def content(self):
            return "<html></html>"

        def wait_for_selector(self, *_args, **_kwargs):
            raise TimeoutError("table not ready")

        def locator(self, _selector):
            return FakeLocator(self.body_text)

    from screener.coinalyze import _page_content

    try:
        _page_content(FakePage("temporary empty response"), TEST_TABLE_URL, 1000)
    except RetryableCoinalyzeError:
        pass
    else:
        raise AssertionError("A missing table after timeout should be retryable")

    try:
        _page_content(FakePage("Please sign in to continue"), TEST_TABLE_URL, 1000)
    except CoinalyzeError as exc:
        assert not isinstance(exc, RetryableCoinalyzeError)
        assert "cookies" in str(exc)
    else:
        raise AssertionError("An expired session should be a permanent error")


def test_volume_24h_market_capitalization_header_alias():
    html = FIXTURE_HTML.replace("<th>VOL24H / MktCap</th>", "<th>Volume 24H / Market Capitalization</th>")
    rows, _, missing, missing_required = parse_table(html)
    assert "vol_mktcap_ratio" not in missing
    assert "vol_mktcap_ratio" not in missing_required
    assert rows[0].vol_mktcap_ratio == 0.033


def test_coin_name_uses_header_index_not_assumed_second_cell():
    html = FIXTURE_HTML.replace("<th></th><th>Coin</th>", "<th>Coin</th>")
    html = html.replace("<td></td><td><span>Bitcoin</span></td>", "<td><span>Bitcoin</span></td>")
    html = html.replace("<td></td><td><span>Ethereum</span></td>", "<td><span>Ethereum</span></td>")
    rows, _, _, _ = parse_table(html)
    assert rows[0].name == "Bitcoin"
    assert rows[1].name == "Ethereum"


def test_parser_rejects_negative_liquidation_amounts_without_dropping_market_row():
    html = (
        FIXTURE_HTML
        .replace("$208.3m", "$-208.3m", 1)
        .replace("$19.9m", "$-19.9m", 1)
        .replace("$188.4m", "$-188.4m", 1)
    )

    rows, _, _, _ = parse_table(html)

    assert len(rows) == 2
    btc = rows[0]
    assert btc.liq_total24_usd is None
    assert btc.liq_short24_usd is None
    assert btc.liq_long24_usd is None
    assert set(btc.raw["invalid_market_fields"]) == {
        "liq_total24_usd", "liq_short24_usd", "liq_long24_usd"
    }


def test_parser_rejects_imbalance_outside_physical_range():
    html = (
        FIXTURE_HTML
        .replace("4.9580", "104.9580", 1)
        .replace("9.5607", "-109.5607", 1)
    )

    rows, _, _, _ = parse_table(html)

    assert len(rows) == 2
    btc = rows[0]
    assert btc.futures_imbalance_4h is None
    assert btc.spot_imbalance_4h is None
    assert set(btc.raw["invalid_market_fields"]) == {
        "futures_imbalance_4h", "spot_imbalance_4h"
    }


def test_parse_table_ignores_unrelated_tables_and_grouped_headers():
    unrelated = "<table><thead><tr><th>Layout</th></tr></thead><tbody><tr><td>menu</td></tr></tbody></table>"
    grouped_header = "<tr><th colspan='34'>Grouped market columns</th></tr>"
    html = FIXTURE_HTML.replace("<thead><tr>", f"<thead>{grouped_header}<tr>", 1)
    rows, headers, _, missing_required = parse_table(unrelated + html)

    assert [row.symbol for row in rows] == ["BTCUSDT_PERP.A", "ETHUSDT_PERP.A"]
    assert "Coin" in headers
    assert missing_required == []


def test_parse_table_rejects_duplicate_symbol_within_one_page():
    import re
    import pytest

    row = re.search(r"<tr data-coin=\"BTCUSDT_PERP\.A\">.*?</tr>", FIXTURE_HTML, re.S).group(0)
    html = FIXTURE_HTML.replace("</tbody>", row + "</tbody>", 1)

    with pytest.raises(CoinalyzeError, match="Duplicate Coinalyze symbol BTCUSDT_PERP.A"):
        parse_table(html)


def test_parse_table_rejects_out_of_range_btc_correlation():
    html = FIXTURE_HTML.replace("+1.00</td>", "1.20</td>", 1)
    rows, _, _, _ = parse_table(html)

    assert rows[0].btc_corr7d is None
    assert "btc_corr7d" in rows[0].raw["invalid_market_fields"]


def test_pagination_next_disabled_checks_bootstrap_parent_state():
    from screener.coinalyze import _pagination_next_is_disabled

    class FakeButton:
        def __init__(self, *, visible=True, disabled=False):
            self.visible = visible
            self.disabled = disabled

        def is_visible(self):
            return self.visible

        def evaluate(self, _script):
            return self.disabled

        def get_attribute(self, _name):
            return None

    assert _pagination_next_is_disabled(FakeButton(disabled=True)) is True
    assert _pagination_next_is_disabled(FakeButton(visible=False)) is True
    assert _pagination_next_is_disabled(FakeButton()) is False


def test_parse_number_handles_mixed_and_decimal_comma_formats():
    assert parse_number("1.234,56") == 1234.56
    assert parse_number("4,9580") == 4.958
    assert parse_number("1,234.56") == 1234.56
    assert parse_number("1,234") == 1234.0
    assert parse_number("–") is None


def test_duplicate_oi_column_without_unit_is_not_guessed_as_usd():
    from bs4 import BeautifulSoup
    from screener.coinalyze import _disambiguate_oi

    cells = BeautifulSoup("<table><tr><td>1.58</td><td>$250k</td></tr></table>", "lxml").select("td")
    percent, usd = _disambiguate_oi(cells, [0, 1])
    assert percent is None
    assert usd == 250_000


def test_pagination_accepts_www_variant_of_same_coinalyze_host():
    html = '<div class="pagination"><a href="https://www.coinalyze.net/?page=2">2</a></div>'
    urls = _pagination_urls(html, "https://coinalyze.net/?columns=x&contract_type=perpetual")
    assert len(urls) == 2
    assert "www.coinalyze.net" in urls[1]
    assert "columns=x" in urls[1]


def test_one_hour_long_short_ratio_is_optional_for_p1_schema():
    from screener.coinalyze import REQUIRED_FOR_P1

    # P1 scenarios use the 1D account ratio; the 1H ratio is parsed for
    # diagnostics only and must not disable every alert when absent.
    assert "ls_ratio_1h" not in REQUIRED_FOR_P1
    assert "ls_ratio_1d" in REQUIRED_FOR_P1



def test_parse_number_extracts_compact_suffix_before_thousands_separator():
    assert parse_number("$1,234M") == 1_234_000_000
    assert parse_number("$1,234K") == 1_234_000
    assert parse_number("$-1,234M") == -1_234_000_000
    assert parse_number("1,234.56M") == 1_234_560_000
    assert parse_number("1.234,56M") == 1_234_560_000
    assert parse_number("1.2M USD") == 1_200_000
    assert parse_number("1,234 USDT") == 1_234
    assert parse_number("-1.17%") == -1.17


def test_sequential_pagination_url_discovery_reuses_all_contiguous_links():
    from screener.coinalyze import _sequential_pagination_urls

    html = (
        '<div class="pagination">'
        '<a href="?page=1">1</a>'
        '<a href="?page=2">2</a>'
        '<a href="?page=3">3</a>'
        '</div>'
    )
    urls = _sequential_pagination_urls(html, TEST_TABLE_URL)

    assert len(urls) == 2
    assert "page=2" in urls[0]
    assert "page=3" in urls[1]
    assert all("columns=test-columns" in url for url in urls)
    assert all("filter=test-filter" in url for url in urls)
    assert all("contract_type=perpetual" in url for url in urls)


def test_sequential_pagination_url_discovery_rejects_hidden_page_gaps():
    from screener.coinalyze import _sequential_pagination_urls

    html = (
        '<div class="pagination">'
        '<a href="?page=2">2</a>'
        '<a href="?page=3">3</a>'
        '<span class="ellipsis">...</span>'
        '<a href="?page=8">8</a>'
        '</div>'
    )

    assert _sequential_pagination_urls(html, TEST_TABLE_URL) == []


def test_javascript_pagination_finds_only_the_immediate_numeric_page_button():
    from screener.coinalyze import _find_js_pagination_next_button

    class FakeCandidate:
        def __init__(self, label):
            self.label = label

        def inner_text(self):
            return self.label

        def is_visible(self):
            return True

        def evaluate(self, _script):
            return False

        def get_attribute(self, _name):
            return None

    class FakeLocator:
        def __init__(self, count=0):
            self.first = self
            self._count = count

        def count(self):
            return self._count

        def locator(self, _selector):
            return FakeLocator()

    class FakePage:
        def __init__(self, candidates):
            self.candidates = candidates

        def locator(self, _selector):
            return FakeLocator(count=1)

        def query_selector_all(self, _selector):
            return self.candidates

    html = (
        '<ul class="pagination">'
        '<li class="active">1</li>'
        '<li><button>2</button></li>'
        '<li><span class="ellipsis">...</span></li>'
        '<li><a href="?page=8">8</a></li>'
        '</ul>'
    )
    from screener.coinalyze import _pagination_next_url
    assert _pagination_next_url(html, TEST_TABLE_URL) is None

    immediate = FakeCandidate("2")
    distant = FakeCandidate("8")

    selected = _find_js_pagination_next_button(
        FakePage([distant, immediate]), html, TEST_TABLE_URL
    )
    assert selected is immediate

    assert _find_js_pagination_next_button(
        FakePage([distant]), html, TEST_TABLE_URL
    ) is None

    disabled_next_html = html.replace("<button>2</button>", '<button disabled="disabled">2</button>')
    import pytest
    from screener.coinalyze import IncompleteUniverseError
    with pytest.raises(IncompleteUniverseError, match="refusing to skip hidden pages"):
        _pagination_next_url(disabled_next_html, TEST_TABLE_URL)


def test_wait_for_javascript_pagination_transition_polls_until_first_row_changes():
    from screener.coinalyze import _wait_for_table_transition

    class FakeRows:
        def __init__(self, page):
            self.page = page
            self.first = self

        def get_attribute(self, name):
            self.page.reads += 1
            symbol = "OLD" if self.page.reads < 2 else "NEW"
            return symbol if name in {"data-coin", "data-symbol"} else None

    class FakePage:
        def __init__(self):
            self.reads = 0
            self.waits = []

        def wait_for_timeout(self, milliseconds):
            self.waits.append(milliseconds)

        def locator(self, _selector):
            return FakeRows(self)

    page = FakePage()
    assert _wait_for_table_transition(page, "OLD", attempts=5, interval_ms=250) is True
    assert len(page.waits) == 2
    assert page.waits == [250, 250]


def test_pagination_ignores_page_one_alias_of_base_url():
    html = '<div class="pagination"><a href="?page=1">1</a><a href="?page=2">2</a></div>'
    base = "https://coinalyze.net/?columns=x&contract_type=perpetual"
    urls = _pagination_urls(html, base)
    assert len(urls) == 2
    assert "page=2" in urls[1]


def test_pagination_walks_next_page_and_fails_closed_at_bootstrap_ellipsis():
    html = (
        '<ul class="pagination">'
        '<li><a href="?page=1">1</a></li>'
        '<li><a href="?page=2">2</a></li>'
        '<li><a href="?page=3">3</a></li>'
        '<li class="disabled"><span>...</span></li>'
        '<li><a href="?page=8">8</a></li>'
        '</ul>'
    )
    base = "https://coinalyze.net/?columns=x&contract_type=perpetual&page=1"
    from urllib.parse import parse_qs, urlparse
    assert parse_qs(urlparse(_pagination_next_url(html, base)).query)["page"] == ["2"]
    from screener.coinalyze import IncompleteUniverseError
    with __import__("pytest").raises(IncompleteUniverseError, match="skip hidden pages"):
        _pagination_next_url(html, base.replace("page=1", "page=3"))


def test_pagination_uses_explicit_next_link_across_hidden_page_range():
    html = (
        '<ul class="pagination">'
        '<li><a href="?page=2">2</a></li>'
        '<li class="next"><a rel="next" href="?page=4">Next</a></li>'
        '</ul>'
    )
    base = "https://coinalyze.net/?columns=x&contract_type=perpetual&page=3"
    from urllib.parse import parse_qs, urlparse
    assert parse_qs(urlparse(_pagination_next_url(html, base)).query)["page"] == ["4"]


def test_single_generic_oi_change_column_is_classified_from_cell_unit_not_as_usd():
    html = """
    <table><thead><tr><th>Coin</th><th>Open Interest Change 24H</th><th>OI Chg 4H</th></tr></thead>
    <tbody><tr data-coin="ALTUSDT_PERP"><td>ALT</td><td>-1.17%</td><td>$1.2M</td></tr></tbody></table>
    """
    rows, _, _, missing_required = parse_table(html)
    row = rows[0]

    assert row.oi_chg24_pct == -1.17
    assert row.oi_chg24_usd is None
    assert row.oi_chg4h_pct is None
    assert row.oi_chg4h_usd == 1_200_000
    assert "oi_chg24_usd" in missing_required
    assert "oi_chg24_pct" not in missing_required
    assert "oi_chg4h_pct" in missing_required


def test_single_bare_generic_oi_change_column_remains_ambiguous():
    html = """
    <table><thead><tr><th>Coin</th><th>OI Chg 24H</th></tr></thead>
    <tbody><tr data-coin="ALTUSDT_PERP"><td>ALT</td><td>-1.17</td></tr></tbody></table>
    """
    rows, _, _, missing_required = parse_table(html)
    assert rows[0].oi_chg24_pct is None
    assert rows[0].oi_chg24_usd is None
    assert "oi_chg24_pct" in missing_required
    assert "oi_chg24_usd" in missing_required


def test_javascript_next_control_is_deferred_to_browser_click_instead_of_skipping_pages():
    html = (
        '<ul class="pagination">'
        '<li><a href="?page=1">1</a></li>'
        '<li><a href="?page=2">2</a></li>'
        '<li><a href="?page=3">3</a></li>'
        '<li><span class="ellipsis">...</span></li>'
        '<li><a href="?page=8">8</a></li>'
        '<li class="next"><a href="#" aria-label="Next">Next</a></li>'
        '</ul>'
    )
    base = "https://coinalyze.net/?columns=x&contract_type=perpetual&page=3"
    assert _pagination_next_url(html, base) is None


def test_parse_table_rejects_symbol_delimiter_injection():
    html = """
    <table><thead><tr><th>Coin</th></tr></thead>
    <tbody><tr data-coin="BTC|SHORT|INJECT"><td>Bitcoin</td></tr></tbody></table>
    """
    from screener.coinalyze import CoinalyzeError
    import pytest
    with pytest.raises(CoinalyzeError, match="unsafe event identity"):
        parse_table(html)
