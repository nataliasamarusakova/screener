from __future__ import annotations

import html
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

from .metrics import EnrichedRow
from .models import ScenarioResult
from .structure import scenario_invalidation

log = logging.getLogger(__name__)


def _safe_telegram_truncate(message: str, limit: int = 3900) -> str:
    """Compact alerts are not silently truncated; fail closed on oversized content."""
    if len(message) > limit:
        raise ValueError(f"Telegram message exceeds safe limit ({len(message)} > {limit})")
    return message


def _esc(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=False)


def _number(value: float | int | None, precision: int = 2, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "n/a"
        raw = f"{numeric:+.{precision}f}" if signed else f"{numeric:.{precision}f}"
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    return raw.replace(".", ",")


def _percent(value: float | None, precision: int = 2, *, signed: bool = True) -> str:
    rendered = _number(value, precision, signed=signed)
    return f"{rendered}%" if rendered != "n/a" else "n/a"


def _money(value: float | None, *, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "n/a"
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    sign = "-" if numeric < 0 else ("+" if signed else "")
    absolute = abs(numeric)
    if absolute >= 1_000_000_000:
        amount, suffix = absolute / 1_000_000_000, "b"
    elif absolute >= 1_000_000:
        amount, suffix = absolute / 1_000_000, "m"
    elif absolute >= 1_000:
        amount, suffix = absolute / 1_000, "k"
    else:
        amount, suffix = absolute, ""
    return f"{sign}${_number(amount, 1)}{suffix}"


def _base_symbol(symbol: str) -> str:
    value = str(symbol or "").upper().split(".", 1)[0]
    for suffix in ("_PERPETUAL", "-PERPETUAL", "_PERP", "-PERP", "PERPETUAL", "PERP"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "BNB", "EUR"):
        if value.endswith(quote) and len(value) > len(quote):
            value = value[: -len(quote)]
            break
    return value or str(symbol)


def _contract_symbol(symbol: str) -> str:
    """Return the source contract ticker, preserving its quote asset (e.g. INITUSDT)."""
    original = str(symbol or "").strip()
    value = original.upper().split(".", 1)[0]
    for suffix in ("_PERPETUAL", "-PERPETUAL", "_PERP", "-PERP", "PERPETUAL", "PERP", "_SWAP", "-SWAP"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value or original


def _safe_telegram_error(error: Any, token: str, recipient_id: str | None = None) -> str:
    """Return a bounded diagnostic without the Bot API token or recipient ID."""
    message = str(error)
    secrets = {token, quote(token, safe="")} if token else set()
    if recipient_id and len(str(recipient_id)) >= 4:
        secrets.update({str(recipient_id), quote(str(recipient_id), safe="")})
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:500]


def _scenario_code(scenario: str) -> str:
    return str(scenario or "").split(" — ", 1)[0].strip() or "SCENARIO"


def _confirmation_anchor(result: ScenarioResult) -> int | None:
    technical = result.technical or {}
    code = _scenario_code(result.scenario).upper()
    source_field = {
        "LONG A": ("trend_retest_long", "retest_open_time_ms"),
        "LONG B": ("higher_low", "latest_open_time_ms"),
        "LONG D": ("higher_low", "latest_open_time_ms"),
        "SHORT A": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT B": ("trend_retest_short", "retest_open_time_ms"),
        "SHORT C": ("bearish_structure_break", "break_open_time_ms"),
        "SHORT D": ("weak_bounce_rejection", "rejection_open_time_ms"),
        "SHORT E": ("failed_breakout_rejection", "failure_open_time_ms"),
    }.get(code)
    if not source_field:
        return None
    value = (technical.get(source_field[0]) or {}).get(source_field[1])
    return value if isinstance(value, int) and value > 0 else None


def _trigger_description(result: ScenarioResult) -> str:
    return {
        "LONG A": "1H: пробой вверх и успешный ретест подтверждены",
        "LONG B": "1H: Higher Low подтверждён; Spot положителен в трёх разнесённых 4H-снимках",
        "LONG D": "1H: Higher Low подтверждён после long flush",
        "SHORT A": "1H: пробой вниз и успешный ретест подтверждены",
        "SHORT B": "1H: пробой вниз и успешный ретест подтверждены",
        "SHORT C": "1H: медвежий пробой структуры и ретест подтверждены",
        "SHORT D": "1H: слабый отскок и отказ от сопротивления подтверждены",
        "SHORT E": "1H: неудавшийся пробой вверх подтверждён; 4H-последовательность выполнена",
    }.get(_scenario_code(result.scenario).upper(), "Обязательное подтверждение структуры получено")


def _signal_warnings(result: ScenarioResult, row: EnrichedRow, technical: dict[str, Any]) -> list[str]:
    m = row.market
    warnings: list[str] = []
    if m.fr_oiw_pct is not None and m.pfr_oiw_pct is not None and m.fr_oiw_pct * m.pfr_oiw_pct < 0:
        warnings.append("PFR OI_W меняет знак относительно FR OI_W")

    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    direction = result.direction.upper()
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
        if direction == "LONG" and ema20 < ema50:
            warnings.append("EMA20 ниже EMA50 на 4H — старший тренд против LONG")
        elif direction == "SHORT" and ema20 > ema50:
            warnings.append("EMA20 выше EMA50 на 4H — старший тренд против SHORT")

    if m.oi_chg4h_usd is not None and abs(m.oi_chg4h_usd) < 2_000_000:
        warnings.append(f"OI 4H мало в USD: {_money(m.oi_chg4h_usd)}")
    if m.chg24_pct is not None and m.oi_chg24_pct is not None:
        if direction == "SHORT" and m.chg24_pct < 0 and m.oi_chg24_pct < 0:
            warnings.append("Цена и OI 24H падают: часть движения может быть закрытием LONG")
        elif direction == "LONG" and m.chg24_pct > 0 and m.oi_chg24_pct < 0:
            warnings.append("Цена растёт при падающем OI 24H: движение может быть закрытием SHORT")
    if m.btc_corr7d is not None and m.btc_corr7d > 0.7 and _base_symbol(result.symbol) != "BTC":
        warnings.append(f"BTC Corr. 7D {_number(m.btc_corr7d, 2, signed=True)} — высокая связь с BTC")
    return list(dict.fromkeys(warnings))[:2]


def _trigger_timestamp(result: ScenarioResult) -> str:
    value = _confirmation_anchor(result)
    if value is None:
        return "n/a"
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_alert(
    result: ScenarioResult,
    row: EnrichedRow,
    technical: dict[str, Any] | None = None,
    *,
    fetched_at: datetime | None = None,
    snapshot_delta: dict[str, Any] | None = None,
) -> str:
    """Compact confirmed P1 scenario; all numeric values remain copyable via <code>."""
    m = row.market
    technical = technical or result.technical or {}
    direction = result.direction.upper()
    emoji = "🟢" if direction == "LONG" else "🔴"
    pair = str(technical.get("binance_symbol") or _contract_symbol(m.symbol)).upper()
    if not pair.endswith("USDT"):
        raise ValueError(f"No resolved Binance USDT perpetual for {m.symbol}")
    name = str(m.name or _base_symbol(m.symbol)).strip()
    warnings = _signal_warnings(result, row, technical)
    status_parts = ["CONFIRMED", f"P1 {result.passed_count}/{result.total_count}"]
    if result.status == "CONFIRMED_SIGNAL_WITH_WARNING" or warnings:
        status_parts.append("WARNING")
    lines = [
        f"{emoji} <b>{_esc(direction)}</b> · {_esc(name)} (<code>{_esc(pair)}</code>)",
        f"<b>{_esc(_scenario_code(result.scenario))} · {' · '.join(status_parts)}</b>",
        "",
        "<b>MARKET</b>",
        f"Price 24H: <code>{_percent(m.chg24_pct, 2)}</code>",
        f"Futures / Spot: <code>{_number(m.futures_imbalance_4h, 4, signed=True)}</code> / <code>{_number(m.spot_imbalance_4h, 4, signed=True)}</code>",
        f"Buy F/S: <code>{_percent(row.futures_buy_share_pct, 2, signed=False)}</code> / <code>{_percent(row.spot_buy_share_pct, 2, signed=False)}</code>",
        f"Flow Gap (S−F): <code>{_number(row.spread, 4, signed=True)}</code>",
        f"OI 4H: <code>{_percent(m.oi_chg4h_pct, 2)}</code> / <code>{_money(m.oi_chg4h_usd)}</code>",
        f"OI 24H: <code>{_percent(m.oi_chg24_pct, 2)}</code> / <code>{_money(m.oi_chg24_usd)}</code>",
        "",
        "<b>DERIVATIVES</b>",
        f"FR / PFR / Δ: <code>{_percent(m.fr_oiw_pct, 4)}</code> / <code>{_percent(m.pfr_oiw_pct, 4)}</code> / <code>{_number(row.delta_funding_oiw_pp, 4, signed=True)} п.п.</code>",
        f"L/S 1D: <code>{_number(m.ls_ratio_1d, 2)}</code>",
        f"Long Liq: <code>{_percent(row.long_liq_share * 100 if row.long_liq_share is not None else None, 2, signed=False)}</code>",
    ]
    if _scenario_code(result.scenario).upper() == "SHORT D" and row.liquidation_coverage_of_oi_drop_pct is not None:
        lines.append(f"Liqs / OI drop: <code>{_number(row.liquidation_coverage_of_oi_drop_pct, 2)}%</code>")
    lines.extend([
        "",
        "<b>MARKET CONTEXT</b>",
        f"BTC Corr. 7D: <code>{_number(m.btc_corr7d, 2, signed=True)}</code>" if m.btc_corr7d is not None else "BTC Corr. 7D: <code>n/a</code> — не проверена",
        f"RS vs BTC: <code>{_number(row.rel_strength_vs_btc_pp, 2, signed=True)} п.п.</code>" if row.rel_strength_vs_btc_pp is not None else "RS vs BTC: <code>n/a</code>",
        "",
        "<b>CONFIRMATION</b>",
        _esc(_trigger_description(result)),
        f"Trigger candle: <code>{_trigger_timestamp(result)}</code>",
    ])
    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
        relation = "EMA20 выше EMA50" if ema20 > ema50 else "EMA20 ниже EMA50" if ema20 < ema50 else "EMA20 = EMA50"
        opposing = (direction == "LONG" and ema20 < ema50) or (direction == "SHORT" and ema20 > ema50)
        if not opposing:
            lines.append(f"4H context: {_esc(relation)}")
    invalidation = scenario_invalidation(direction, result.scenario, technical)
    level = invalidation.get("level")
    if not isinstance(level, (int, float)) or not math.isfinite(float(level)):
        raise ValueError(f"No valid 1H invalidation level for {result.scenario}")
    comparator = "ниже" if direction == "LONG" else "выше"
    lines.append(f"Invalidation: 1H close {comparator} <code>{_number(level, 8)}</code>")
    if warnings:
        lines.append("")
        lines.extend(f"⚠️ {_esc(warning)}" for warning in warnings)
    return _safe_telegram_truncate("\n".join(lines), limit=3900)


def format_cancellation_message(
    pair: str, direction: str, scenario_code: str, level: float,
) -> str:
    relation = "ниже" if direction.upper() == "LONG" else "выше"
    level_text = f"{float(level):.8f}" if math.isfinite(float(level)) else "n/a"
    return (
        f"⚪ <b>СЦЕНАРИЙ ОТМЕНЁН</b> — <code>{_esc(pair)}</code>\n"
        f"{_esc(scenario_code)}: закрытие 1H {relation} <code>{level_text}</code>."
    )


def format_reversal_message(
    pair: str, old_direction: str, old_scenario: str,
    new_direction: str, new_scenario: str,
) -> str:
    # Scenario codes already contain their direction (SHORT D / LONG A).
    old_label = old_scenario.upper() if old_scenario.upper().startswith(("LONG", "SHORT")) else f"{old_direction.upper()} {old_scenario.upper()}"
    new_label = new_scenario.upper() if new_scenario.upper().startswith(("LONG", "SHORT")) else f"{new_direction.upper()} {new_scenario.upper()}"
    return (
        f"🔄 <b>РАЗВОРОТ</b> — <code>{_esc(pair)}</code>\n"
        f"{_esc(old_label)} → {_esc(new_label)}\n"
        f"Противоположный сценарий подтверждён закрытыми 1H-свечами."
    )


class TelegramClient:
    def __init__(self, token: str, chat_ids: list[str], timeout_sec: float = 15.0) -> None:
        self.token = token
        self.chat_ids = chat_ids
        self.timeout_sec = timeout_sec

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_ids)

    def send(
        self, text: str, chat_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if not self.configured:
            log.warning("Telegram is not configured; notification skipped")
            return []
        recipients = self.chat_ids if chat_ids is None else chat_ids
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        results: list[dict[str, Any]] = []
        payload_text = _safe_telegram_truncate(text)
        for chat_id in recipients:
            attempts = 0
            while attempts < 3:
                attempts += 1
                try:
                    response = requests.post(
                        url,
                        json={
                            "chat_id": chat_id,
                            "text": payload_text,
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True,
                        },
                        timeout=self.timeout_sec,
                    )
                except requests.RequestException as exc:
                    # Telegram may have accepted a POST before the connection
                    # failed. A blind retry can duplicate a cancellation/reversal.
                    # The Bot API token is part of the URL and must never reach logs/JSONL.
                    safe_error = _safe_telegram_error(exc, self.token, chat_id)
                    results.append({"chat_id": chat_id, "sent": False, "delivery_unknown": True, "error": safe_error})
                    log.error("Telegram delivery outcome unknown for a configured recipient: %s", safe_error)
                    break

                if response.status_code == 429:
                    try:
                        retry_after = float((response.json().get("parameters") or {}).get("retry_after", 1))
                    except (ValueError, TypeError, AttributeError):
                        retry_after = 1.0
                    if attempts < 3:
                        time.sleep(min(max(retry_after, 1.0), 30.0))
                        continue
                    results.append({
                        "chat_id": chat_id, "sent": False, "delivery_unknown": False,
                        "error": f"Telegram rate limit; retry_after={retry_after}s",
                    })
                    break

                if response.status_code >= 500:
                    # A server-side error after a POST is treated as ambiguous;
                    # don't re-send an event which may already have been accepted.
                    results.append({
                        "chat_id": chat_id, "sent": False, "delivery_unknown": True,
                        "error": f"Telegram server error {response.status_code}",
                    })
                    break

                try:
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError) as exc:
                    permanent_failure = (
                        400 <= response.status_code < 500
                        and response.status_code not in {408, 425, 429}
                    )
                    results.append({
                        "chat_id": chat_id, "sent": False,
                        "delivery_unknown": response.status_code < 400,
                        "permanent_failure": permanent_failure,
                        "error": _safe_telegram_error(exc, self.token, chat_id),
                    })
                    break
                if not payload.get("ok"):
                    error_code = payload.get("error_code")
                    try:
                        error_status = int(error_code)
                    except (TypeError, ValueError):
                        error_status = None
                    result = {
                        "chat_id": chat_id, "sent": False, "delivery_unknown": False,
                        "error": _safe_telegram_error(payload.get("description", "Telegram API returned ok=false"), self.token, chat_id),
                    }
                    if (
                        error_status is not None
                        and 400 <= error_status < 500
                        and error_status not in {408, 425, 429}
                    ):
                        result["permanent_failure"] = True
                    results.append(result)
                    break
                results.append({
                    "chat_id": chat_id, "sent": True, "delivery_unknown": False,
                    "message_id": (payload.get("result") or {}).get("message_id"),
                })
                break
        return results

