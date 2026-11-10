from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from .models import MarketRow

log = logging.getLogger(__name__)


class CoinalyzeError(RuntimeError):
    pass


class IncompleteUniverseError(CoinalyzeError):
    """A page/pagination failure means the universe is incomplete; do not analyze it."""


@dataclass(slots=True)
class ScrapeResult:
    rows: list[MarketRow]
    fetched_at: datetime
    contract_type: str
    pages_loaded: int
    headers_found: list[str]
    missing_columns: list[str]
    source_url: str
    missing_required_columns: list[str] = field(default_factory=list)


def parse_number(value: Any) -> float | None:
    """Parse Coinalyze compact values without turning n/a into zero."""
    if value is None:
        return None
    text = str(value).strip().replace("\u2212", "-").replace(",", "")
    if not text or text.lower() in {"n/a", "na", "nan", "none", "null", "-", "—"}:
        return None
    text = text.replace("$", "").replace("%", "").replace(" ", "")
    multiplier = 1.0
    if text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def normalize_header(value: str) -> str:
    value = str(value or "").strip().lower().replace("\xa0", " ").replace("&", "and")
    value = value.replace("\u2212", "-")
    value = re.sub(r"[^a-z0-9%/]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _header_text(th: Any) -> str:
    for selector in ("span[title]", "[title]"):
        elem = th.select_one(selector)
        if elem and elem.get("title"):
            return str(elem.get("title")).strip()
    for attr in ("aria-label", "data-title", "data-label"):
        if th.get(attr):
            return str(th.get(attr)).strip()
    return th.get_text(" ", strip=True)


ALIASES: dict[str, set[str]] = {
    "coin": {"coin", "asset"},
    "price": {"price"},
    "chg24_pct": {"chg 24h", "price change % 24h", "price change 24h", "change 24h"},
    "market_cap_usd": {"mkt cap", "market cap", "market capitalization", "market capitalisation"},
    "fdv_usd": {"fdv", "fully diluted valuation"},
    "volume24_usd": {"vol 24h", "volume 24h", "24h volume"},
    "oi_usd": {"open interest", "oi"},
    "oi_chg24_pct": {"oi chg 24h %", "oi change 24h %", "open interest change % 24h", "open interest change 24h %"},
    "oi_chg24_usd": {"oi chg 24h usd", "oi change 24h usd", "open interest change 24h usd", "open interest change 24h"},
    "oi_chg4h_pct": {"oi chg 4h %", "oi change 4h %", "open interest change % 4h", "open interest change 4h %"},
    "oi_chg4h_usd": {"oi chg 4h usd", "oi change 4h usd", "open interest change 4h usd", "open interest change 4h"},
    "oi_share_pct": {"oi share", "open interest share"},
    "oi_vol_ratio": {"oi / vol24h", "oi / volume 24h", "open interest / volume 24h", "open interest / volume24h"},
    "oi_mktcap_ratio": {"oi / mktcap", "oi / market cap", "open interest / market capitalization", "open interest / market capitalisation"},
    "vol_mktcap_ratio": {
        "vol24h / mktcap", "volume 24h / market cap", "volume 24h / market capitalization",
        "volume 24h / market capitalisation", "vol 24h / mktcap",
    },
    "fr_avg_pct": {"fr avg", "funding rate average"},
    "pfr_avg_pct": {"pfr avg", "predicted funding rate average"},
    "fr_oiw_pct": {"fr avg oi w", "fr avg oi weighted", "funding rate average oi weighted", "funding rate average oi w"},
    "pfr_oiw_pct": {"pfr avg oi w", "pfr avg oi weighted", "predicted funding rate average oi weighted", "predicted funding rate average oi w"},
    "liq_total24_usd": {"liqs 24h", "liquidations 24h", "liq 24h"},
    "liq_short24_usd": {"short liqs 24h", "short liquidations 24h", "short liqs"},
    "liq_long24_usd": {"long liqs 24h", "long liquidations 24h", "long liqs"},
    "liq_vol_ratio": {"liqs 24h / vol 24h", "liquidations 24h / volume 24h", "liqs / volume 24h"},
    "liq_oi_ratio": {"liqs 24h / oi", "liquidations 24h / open interest", "liqs / oi"},
    "ls_ratio_1h": {"l/s ratio 1h", "long short ratio 1h", "long/short accounts ratio 1h"},
    "long_accounts_1h_pct": {"long 1h", "long accounts 1h", "long accounts % 1h"},
    "short_accounts_1h_pct": {"short 1h", "short accounts 1h", "short accounts % 1h"},
    "ls_ratio_1d": {"l/s ratio 1d", "long short ratio 1d", "long/short accounts ratio 1d"},
    "long_accounts_1d_pct": {"long 1d", "long accounts 1d", "long accounts % 1d"},
    "short_accounts_1d_pct": {"short 1d", "short accounts 1d", "short accounts % 1d"},
    "futures_imbalance_4h": {"futures imbalance 4h", "future imbalance 4h", "futures imbalance"},
    "spot_imbalance_4h": {"spot imbalance 4h", "spot imbalance"},
    "btc_corr7d": {"btc correlation 7 days", "btc corr 7d", "btc correlation 7d"},
}

REQUIRED_FOR_P1 = {
    "price", "chg24_pct", "market_cap_usd", "volume24_usd", "oi_usd",
    "oi_chg24_pct", "oi_chg24_usd", "oi_chg4h_pct", "oi_chg4h_usd",
    "oi_vol_ratio", "oi_mktcap_ratio", "vol_mktcap_ratio",
    "fr_avg_pct", "fr_oiw_pct", "pfr_oiw_pct", "liq_total24_usd",
    "liq_long24_usd", "liq_short24_usd", "ls_ratio_1h", "ls_ratio_1d",
    "futures_imbalance_4h", "spot_imbalance_4h",
}


def _build_header_map(soup: BeautifulSoup) -> tuple[dict[str, int], list[str], list[int], list[int]]:
    ths = soup.select("thead tr th")
    if not ths:
        raise CoinalyzeError("Не найдена строка заголовков thead/th.")
    headers = [_header_text(th) for th in ths]
    normalized = [normalize_header(x) for x in headers]
    index: dict[str, int] = {}

    for field, aliases in ALIASES.items():
        norm_aliases = {normalize_header(a) for a in aliases}
        hits = [i for i, header in enumerate(normalized) if header in norm_aliases]
        if len(hits) == 1:
            index[field] = hits[0]
        elif len(hits) > 1:
            log.warning("Duplicate Coinalyze column for %s at %s; row-level disambiguation used", field, hits)

    # Coinalyze sometimes renders the same visible label twice (OI Chg 24H),
    # with the unit only visible in the cell. Keep both positions and inspect each row.
    oi24_ambiguous = [i for i, h in enumerate(normalized) if h in {"oi chg 24h", "open interest change 24h"}]
    oi4h_ambiguous = [i for i, h in enumerate(normalized) if h in {"oi chg 4h", "open interest change 4h"}]
    if len(oi24_ambiguous) >= 2:
        index.pop("oi_chg24_pct", None)
        index.pop("oi_chg24_usd", None)
    if len(oi4h_ambiguous) >= 2:
        index.pop("oi_chg4h_pct", None)
        index.pop("oi_chg4h_usd", None)

    available = set(index)
    if len(oi24_ambiguous) >= 2:
        available.update({"oi_chg24_pct", "oi_chg24_usd"})
    if len(oi4h_ambiguous) >= 2:
        available.update({"oi_chg4h_pct", "oi_chg4h_usd"})
    return index, headers, oi24_ambiguous, oi4h_ambiguous


def _cell_text(cells: list[Any], idx: int | None) -> str | None:
    if idx is None or idx < 0 or idx >= len(cells):
        return None
    return cells[idx].get_text(" ", strip=True)


def _value(cells: list[Any], index: dict[str, int], field: str) -> float | None:
    return parse_number(_cell_text(cells, index.get(field)))


def _disambiguate_oi(cells: list[Any], indices: list[int]) -> tuple[float | None, float | None]:
    pct = usd = None
    for idx in indices:
        raw = _cell_text(cells, idx) or ""
        parsed = parse_number(raw)
        if parsed is None:
            continue
        if "%" in raw:
            pct = parsed
        else:
            usd = parsed
    return pct, usd


def parse_table(html: str) -> tuple[list[MarketRow], list[str], list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    trs = soup.select("tbody tr")
    if not trs:
        raise CoinalyzeError("Таблица Coinalyze не содержит tbody/tr.")
    index, headers, oi24_ambiguous, oi4h_ambiguous = _build_header_map(soup)
    available = set(index)
    # Coinalyze renders percentage and USD OI changes as duplicate visible labels.
    if len(oi24_ambiguous) >= 2:
        available.update({"oi_chg24_pct", "oi_chg24_usd"})
    if len(oi4h_ambiguous) >= 2:
        available.update({"oi_chg4h_pct", "oi_chg4h_usd"})
    missing_columns = sorted(set(ALIASES) - available)
    missing_required = sorted(REQUIRED_FOR_P1 - available)

    out: list[MarketRow] = []
    for row_idx, tr in enumerate(trs, start=1):
        symbol = str(tr.get("data-coin") or tr.get("data-symbol") or "").strip().upper()
        cells = tr.find_all("td")
        if not symbol or not cells:
            raise CoinalyzeError(f"Invalid Coinalyze row {row_idx}: missing symbol or cells; refusing partial universe")
        name = symbol
        coin_cell_index = index.get("coin")
        if coin_cell_index is not None and coin_cell_index < len(cells):
            coin_cell = cells[coin_cell_index]
            spans = coin_cell.find_all("span")
            name = spans[0].get_text(strip=True) if spans else (coin_cell.get_text(" ", strip=True) or symbol)
        oi24_pct, oi24_usd = _disambiguate_oi(cells, oi24_ambiguous) if len(oi24_ambiguous) >= 2 else (None, None)
        oi4h_pct, oi4h_usd = _disambiguate_oi(cells, oi4h_ambiguous) if len(oi4h_ambiguous) >= 2 else (None, None)
        if oi24_pct is None:
            oi24_pct = _value(cells, index, "oi_chg24_pct")
        if oi24_usd is None:
            oi24_usd = _value(cells, index, "oi_chg24_usd")
        if oi4h_pct is None:
            oi4h_pct = _value(cells, index, "oi_chg4h_pct")
        if oi4h_usd is None:
            oi4h_usd = _value(cells, index, "oi_chg4h_usd")

        price_raw = _cell_text(cells, index.get("price"))
        parsed_price = _value(cells, index, "price")
        price_unavailable = parsed_price is not None and parsed_price <= 0
        row = MarketRow(
            symbol=symbol,
            source_symbol=symbol,
            name=name,
            price=None if price_unavailable else parsed_price,
            chg24_pct=_value(cells, index, "chg24_pct"),
            market_cap_usd=_value(cells, index, "market_cap_usd"),
            fdv_usd=_value(cells, index, "fdv_usd"),
            volume24_usd=_value(cells, index, "volume24_usd"),
            oi_usd=_value(cells, index, "oi_usd"),
            oi_chg24_pct=oi24_pct,
            oi_chg24_usd=oi24_usd,
            oi_chg4h_pct=oi4h_pct,
            oi_chg4h_usd=oi4h_usd,
            oi_share_pct=_value(cells, index, "oi_share_pct"),
            oi_vol_ratio=_value(cells, index, "oi_vol_ratio"),
            oi_mktcap_ratio=_value(cells, index, "oi_mktcap_ratio"),
            vol_mktcap_ratio=_value(cells, index, "vol_mktcap_ratio"),
            fr_avg_pct=_value(cells, index, "fr_avg_pct"),
            pfr_avg_pct=_value(cells, index, "pfr_avg_pct"),
            fr_oiw_pct=_value(cells, index, "fr_oiw_pct"),
            pfr_oiw_pct=_value(cells, index, "pfr_oiw_pct"),
            liq_total24_usd=_value(cells, index, "liq_total24_usd"),
            liq_short24_usd=_value(cells, index, "liq_short24_usd"),
            liq_long24_usd=_value(cells, index, "liq_long24_usd"),
            liq_vol_ratio=_value(cells, index, "liq_vol_ratio"),
            liq_oi_ratio=_value(cells, index, "liq_oi_ratio"),
            ls_ratio_1h=_value(cells, index, "ls_ratio_1h"),
            long_accounts_1h_pct=_value(cells, index, "long_accounts_1h_pct"),
            short_accounts_1h_pct=_value(cells, index, "short_accounts_1h_pct"),
            ls_ratio_1d=_value(cells, index, "ls_ratio_1d"),
            long_accounts_1d_pct=_value(cells, index, "long_accounts_1d_pct"),
            short_accounts_1d_pct=_value(cells, index, "short_accounts_1d_pct"),
            futures_imbalance_4h=_value(cells, index, "futures_imbalance_4h"),
            spot_imbalance_4h=_value(cells, index, "spot_imbalance_4h"),
            btc_corr7d=_value(cells, index, "btc_corr7d"),
            raw={"row_number": row_idx, "price_raw": price_raw, "price_rounded_or_invalid_zero": price_unavailable},
        )
        # Hard data integrity checks: malformed values do not silently enter ranks.
        if price_unavailable:
            log.warning("%s: displayed price is non-positive/rounded; keep row with unknown numeric price", symbol)
        if row.oi_usd is not None and row.oi_usd < 0:
            raise CoinalyzeError(f"Invalid negative OI for {symbol}; refusing partial universe")
        out.append(row)
    if not out:
        raise CoinalyzeError("В Coinalyze не удалось разобрать ни одной валидной строки.")
    return out, headers, missing_columns, missing_required


def detect_contract_type(html: str, url: str) -> str:
    """Identify the selected contract tab, preferring an explicit URL query value."""
    soup = BeautifulSoup(html, "lxml")
    active_text = " ".join(
        elem.get_text(" ", strip=True).lower()
        for elem in soup.select(".active, .selected, [aria-selected='true'], [aria-pressed='true']")
    )
    heading_text = " ".join(
        elem.get_text(" ", strip=True).lower() for elem in soup.select("h1, h2, h3")
    )
    focused_text = f"{active_text} {heading_text}"
    body_text = soup.get_text(" ", strip=True).lower()

    if "all contracts" in focused_text and "perpetual contracts" not in focused_text:
        return "ALL_CONTRACTS"
    if "perpetual contracts" in focused_text and "all contracts" not in focused_text:
        return "PERPETUAL"

    selected_type = parse_qs(urlparse(url).query).get("contract_type", [""])[0].strip().lower()
    if selected_type in {"perpetual", "perpetuals", "perp"}:
        return "PERPETUAL"
    if selected_type in {"all", "all_contracts", "all-contracts"}:
        return "ALL_CONTRACTS"
    if "all contracts" not in body_text and "perpetual contracts" in body_text:
        return "PERPETUAL"
    if "perpetual contracts" not in body_text and "all contracts" in body_text:
        return "ALL_CONTRACTS"
    return "UNKNOWN"


def build_session_cookies(p_sid: str, chat_sid: str) -> list[dict[str, Any]]:
    """Build the exact cookie set used by divergence-main without exposing values."""
    if not p_sid or not chat_sid:
        raise CoinalyzeError(
            "Обе cookies Coinalyze обязательны: задайте GitHub Secrets "
            "COINALYZE_P_SID и COINALYZE_CHAT_SID."
        )
    return [
        {"name": "p_sid", "value": p_sid, "domain": "coinalyze.net", "path": "/", "secure": True},
        {"name": "chat_sid", "value": chat_sid, "domain": "coinalyze.net", "path": "/", "secure": True},
        {"name": "cookies_accepted", "value": "1", "domain": "coinalyze.net", "path": "/", "secure": True},
    ]


def _apply_stealth(page: Any) -> None:
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
    except (ImportError, AttributeError):
        try:
            from playwright_stealth import stealth_sync
            stealth_sync(page)
        except ImportError as exc:
            raise CoinalyzeError(
                "playwright-stealth не установлен. Проверь шаг установки зависимостей в GitHub Actions."
            ) from exc


def _page_content(page: Any, url: str, timeout_ms: int) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(3000)
    if "Attention Required" in page.content():
        log.warning("Coinalyze anti-bot challenge detected; waiting before table readiness check")
        page.wait_for_timeout(10_000)
    try:
        page.wait_for_selector("tbody tr", timeout=timeout_ms)
    except Exception as exc:
        body = page.locator("body").inner_text(timeout=3000).lower()
        if any(marker in body for marker in ("sign in", "log in", "login", "access denied")):
            raise CoinalyzeError(
                "Coinalyze не показал таблицу: сессия не авторизована или cookies истекли. "
                "Обнови COINALYZE_P_SID и COINALYZE_CHAT_SID в GitHub Secrets."
            ) from exc
        raise CoinalyzeError(
            "Coinalyze не показал tbody tr. Проверь сохранённый URL, cookies и текущую HTML-разметку сайта."
        ) from exc
    previous = page.locator("tbody tr").count()
    for attempt in range(15):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(700)
        current = page.locator("tbody tr").count()
        if page.locator(".pagination").count() > 0:
            break
        if current == previous and attempt >= 3:
            break
        previous = current
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)
    return page.content()


def _page_content_with_retry(page: Any, url: str, timeout_ms: int, attempts: int = 2) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _page_content(page, url, timeout_ms)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            log.warning(
                "Coinalyze page load attempt %d/%d failed (%s); retrying",
                attempt, attempts, type(exc).__name__,
            )
            time.sleep(2.0)
    raise CoinalyzeError(f"Не удалось загрузить таблицу Coinalyze после {attempts} попыток: {last_error}") from last_error


def _pagination_urls(html: str, base_url: str) -> list[str]:
    """Discover page links while retaining saved columns/filter/contract query values."""
    soup = BeautifulSoup(html, "lxml")
    pagination = soup.select_one(".pagination")
    if not pagination:
        return [base_url]

    base_parsed = urlparse(base_url)
    base_query = parse_qs(base_parsed.query, keep_blank_values=True)
    urls = [base_url]
    for anchor in pagination.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        target = urlparse(urljoin(base_url, href))
        if target.scheme not in {"http", "https"} or target.netloc != base_parsed.netloc:
            continue
        target_query = parse_qs(target.query, keep_blank_values=True)
        for key, value in base_query.items():
            target_query.setdefault(key, value)
        full = urlunparse(target._replace(query=urlencode(target_query, doseq=True), fragment=""))
        if full != base_url and full not in urls:
            urls.append(full)
    return urls


def fetch_table(
    *,
    url: str,
    p_sid: str = "",
    chat_sid: str = "",
    max_pages: int = 20,
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> ScrapeResult:
    """Fetch the paginated Perpetual table using the saved URL and GitHub Secrets cookies."""
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    if parsed_url.scheme != "https" or parsed_url.netloc.lower() not in {"coinalyze.net", "www.coinalyze.net"}:
        raise CoinalyzeError("COINALYZE_URL must be the saved HTTPS Coinalyze table URL.")
    if query.get("contract_type", [""])[0].lower() not in {"perpetual", "perpetuals", "perp"}:
        raise CoinalyzeError("COINALYZE_URL must explicitly select contract_type=perpetual.")
    session_cookies = build_session_cookies(p_sid, chat_sid)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise CoinalyzeError("Playwright is not installed. Run: playwright install chromium") from exc

    rows: list[MarketRow] = []
    seen: set[str] = set()
    headers: list[str] = []
    missing_columns: list[str] = []
    missing_required: list[str] = []
    pages_loaded = 0
    contract_type = "UNKNOWN"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Europe/Berlin",
            viewport={"width": 1920, "height": 1080},
        )
        context.add_cookies(session_cookies)
        page = context.new_page()
        _apply_stealth(page)
        try:
            first_html = _page_content_with_retry(page, url, timeout_ms)
            # The user-provided saved URL explicitly selects contract_type=perpetual.
            contract_type = detect_contract_type(first_html, url)
            first_urls = _pagination_urls(first_html, url)
            if len(first_urls) > max_pages:
                raise IncompleteUniverseError(f"Пагинация Coinalyze содержит больше {max_pages} страниц; выборка обрезана.")
            html_pages: list[str] = [first_html]
            if len(first_urls) > 1:
                for page_url in first_urls[1:]:
                    try:
                        html_pages.append(_page_content_with_retry(page, page_url, timeout_ms))
                    except Exception as exc:
                        raise IncompleteUniverseError(f"Не удалось загрузить страницу {len(html_pages) + 1}: {exc}") from exc
            else:
                # Fallback for JS pagination without href links. A clicked but unchanged
                # table is treated as an error; it is not silently considered last page.
                for page_number in range(1, max_pages):
                    pagination = page.locator(".pagination")
                    if pagination.count() == 0:
                        break
                    first_row = page.locator("tbody tr").first
                    before = first_row.get_attribute("data-coin") or first_row.get_attribute("data-symbol")
                    next_button = pagination.locator("a[rel='next'], button[aria-label='Next'], a[aria-label='Next'], .next").first
                    if next_button.count() == 0:
                        break
                    if next_button.get_attribute("disabled") is not None or (next_button.get_attribute("aria-disabled") or "").lower() == "true":
                        break
                    next_button.click()
                    page.wait_for_timeout(1200)
                    first_row = page.locator("tbody tr").first
                    after = first_row.get_attribute("data-coin") or first_row.get_attribute("data-symbol")
                    if not after or after == before:
                        raise IncompleteUniverseError(f"Переход на страницу {page_number + 1} не подтвердил смену таблицы.")
                    html_pages.append(page.content())
                if len(html_pages) == max_pages:
                    pagination = page.locator(".pagination")
                    next_button = pagination.locator(
                        "a[rel='next'], button[aria-label='Next'], a[aria-label='Next'], .next"
                    ).first
                    has_enabled_next = next_button.count() > 0
                    if has_enabled_next:
                        has_enabled_next = (
                            next_button.get_attribute("disabled") is None
                            and (next_button.get_attribute("aria-disabled") or "").lower() != "true"
                        )
                    if has_enabled_next:
                        raise IncompleteUniverseError(
                            f"Достигнут лимит страниц Coinalyze ({max_pages}); полнота не подтверждена."
                        )

            for page_no, html in enumerate(html_pages, start=1):
                page_rows, page_headers, page_missing, page_missing_required = parse_table(html)
                if headers and page_headers != headers:
                    raise IncompleteUniverseError(f"Coinalyze table schema changed between pages (page {page_no}); refusing mixed-schema universe")
                if missing_columns and page_missing != missing_columns:
                    raise IncompleteUniverseError(f"Coinalyze columns differ on page {page_no}; refusing mixed-schema universe")
                if missing_required and page_missing_required != missing_required:
                    raise IncompleteUniverseError(f"Coinalyze required columns differ on page {page_no}; refusing mixed-schema universe")
                pages_loaded += 1
                headers = page_headers
                missing_columns = page_missing
                missing_required = page_missing_required
                added = 0
                for row in page_rows:
                    if row.symbol not in seen:
                        rows.append(row)
                        seen.add(row.symbol)
                        added += 1
                if page_no > 1 and added == 0:
                    raise IncompleteUniverseError(
                        f"Страница Coinalyze {page_no} не добавила новых монет; пагинация могла повторить первую страницу."
                    )
                log.info("Coinalyze page %d: parsed=%d added=%d unique_total=%d", page_no, len(page_rows), added, len(rows))
        finally:
            context.close()
            browser.close()

    if contract_type == "ALL_CONTRACTS":
        log.warning("Coinalyze view is All Contracts; P1 expects Perpetual contracts")
    if contract_type == "UNKNOWN":
        log.warning("Could not identify Perpetual vs All Contracts from page text")
    if missing_columns:
        log.warning("Missing recognized Coinalyze columns: %s", ", ".join(missing_columns))
    if missing_required:
        log.error("Missing REQUIRED P1 Coinalyze columns: %s", ", ".join(missing_required))
    if not rows:
        raise CoinalyzeError("Пустая выборка Coinalyze.")
    return ScrapeResult(
        rows=rows,
        fetched_at=datetime.now(timezone.utc),
        contract_type=contract_type,
        pages_loaded=pages_loaded,
        headers_found=headers,
        missing_columns=missing_columns,
        source_url=url,
        missing_required_columns=missing_required,
    )
