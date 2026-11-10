from screener.coinalyze import (
    CoinalyzeError,
    build_session_cookies,
    detect_contract_type,
    fetch_table,
    _pagination_urls,
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


def test_zero_rounded_price_does_not_drop_the_market_row():
    html = FIXTURE_HTML.replace("$74,790.08", "$0.0000")
    rows, _, _, _ = parse_table(html)
    assert len(rows) == 2
    assert rows[0].price is None
    assert rows[0].raw["price_rounded_or_invalid_zero"] is True


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
    assert {row.symbol for row in result.rows} == {"BTCUSDT_PERP.A", "ETHUSDT_PERP.A"}
    assert [cookie["name"] for cookie in captured["cookies"]] == [
        "p_sid", "chat_sid", "cookies_accepted"
    ]


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
