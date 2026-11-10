from __future__ import annotations

import html
import logging
import math
import time
from html.parser import HTMLParser
from datetime import datetime, timezone
from typing import Any

import requests

from .metrics import EnrichedRow
from .models import ScenarioResult
from .structure import scenario_invalidation

log = logging.getLogger(__name__)
DISCLAIMER = (
    "\n\n<i>Сценарий прошёл формальные условия P1 и обязательное подтверждение графика. "
    "Это не гарантия результата, не торговая рекомендация и не команда на вход. "
    "Перед входом проверь стоп, риск, комиссии, funding и проскальзывание. "
    "Параметры P1 не проходили backtest.</i>"
)


class _PlainTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _safe_telegram_truncate(message: str, limit: int = 3900) -> str:
    """Keep the Telegram message within the limit without cutting HTML tags."""
    if len(message) <= limit:
        return message
    parser = _PlainTextExtractor()
    parser.feed(message)
    plain = "".join(parser.parts)
    disclaimer_plain = (
        "Сценарий прошёл формальные условия P1 и обязательное подтверждение графика. "
        "Это не гарантия результата, не торговая рекомендация и не команда на вход. "
        "Перед входом проверь стоп, риск, комиссии, funding и проскальзывание. "
        "Параметры P1 не проходили backtest."
    )
    if plain.endswith(disclaimer_plain):
        body = plain[:-len(disclaimer_plain)].rstrip()
    else:
        body = plain.rstrip()
    suffix = "…\n\n" + DISCLAIMER
    budget = max(0, limit - len(suffix))
    # Escaping can expand text (for example '&' -> '&amp;'), so choose the
    # longest prefix by escaped length rather than slicing raw characters.
    low, high = 0, min(len(body), budget)
    while low < high:
        middle = (low + high + 1) // 2
        if len(html.escape(body[:middle].rstrip(), quote=False)) <= budget:
            low = middle
        else:
            high = middle - 1
    return html.escape(body[:low].rstrip(), quote=False) + suffix


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


def _scenario_code(scenario: str) -> str:
    return str(scenario or "").split(" — ", 1)[0].strip() or "SCENARIO"


def _status_label(status: str) -> str:
    return {
        "CONFIRMED_SIGNAL": "ПОДТВЕРЖДЁННЫЙ СЦЕНАРИЙ",
        "CONFIRMED_SIGNAL_WITH_WARNING": "СЦЕНАРИЙ ПОДТВЕРЖДЁН · ЕСТЬ ПРЕДУПРЕЖДЕНИЕ",
        "WAITING_FOR_HISTORY_OR_HIGHER_LOW": "WAITING_HISTORY / HIGHER LOW",
        "WAITING_FOR_HIGHER_LOW": "WAITING_FOR_HIGHER_LOW",
        "WAITING_FOR_1H_BREAKOUT_RETEST": "WAITING_FOR_1H_BREAKOUT_RETEST",
        "WAITING_FOR_1H_BREAKDOWN_RETEST": "WAITING_FOR_1H_BREAKDOWN_RETEST",
        "WAITING_FOR_1H_STRUCTURE_BREAK": "WAITING_FOR_1H_BREAK",
        "BLOCKED_QUALITY": "BLOCKED_QUALITY",
        "INVALIDATED_1H_STRUCTURE": "INVALIDATED · 1H STRUCTURE",
    }.get(str(status).upper(), str(status).upper())


def _technical_bullets(result: ScenarioResult, technical: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
        relation = "выше" if ema20 > ema50 else "ниже" if ema20 < ema50 else "на уровне"
        bullets.append(f"4H EMA20 {relation} EMA50.")
    elif context:
        bullets.append("4H EMA20/EMA50: данных недостаточно.")
    else:
        bullets.append("4H контекст: нет данных.")

    close = context.get("last_close")
    is_long = result.direction.upper() == "LONG"
    level_key = "last_confirmed_support" if is_long else "last_confirmed_resistance"
    level = context.get(level_key)
    level_phrase = "подтверждённой поддержки" if is_long else "подтверждённого сопротивления"
    if isinstance(close, (int, float)) and isinstance(level, (int, float)):
        relation = "ниже" if close < level else "выше" if close > level else "на уровне"
        bullets.append(f"Последнее закрытие 4H {relation} {level_phrase}.")
    elif context:
        bullets.append(f"Уровень {level_phrase} 4H: нет данных.")

    higher = technical.get("higher_low")
    if isinstance(higher, dict):
        if higher.get("confirmed") is True:
            bullets.append("Higher Low подтверждён.")
        elif higher.get("confirmed") is False:
            reason = str(higher.get("reason") or "").lower()
            if higher.get("invalidated") is True:
                bullets.append("Higher Low отменён: последняя закрытая 1H-свеча закрылась ниже pivot.")
            elif "unavailable" in reason or "insufficient candles" in reason:
                bullets.append("Higher Low: свечных данных недостаточно.")
            elif "need at least two" in reason:
                bullets.append("Higher Low не подтверждён: недостаточно подтверждённых минимумов.")
            else:
                bullets.append("Higher Low отсутствует.")
        else:
            bullets.append("Higher Low: нет данных.")
    else:
        bullets.append("Higher Low: нет данных.")

    code = _scenario_code(result.scenario)
    check_map = {
        "LONG A": ("trend_retest_long", "LONG retest"),
        "SHORT A": ("trend_retest_short", "SHORT retest"),
        "SHORT B": ("trend_retest_short", "SHORT breakdown/retest"),
        "SHORT C": ("bearish_structure_break", "Медвежий пробой 1H"),
        "SHORT D": ("weak_bounce_rejection", "Отказ от сопротивления после 1H-отскока"),
        "SHORT E": ("failed_breakout_rejection", "Ложный пробой вверх на 1H"),
    }
    if code in check_map:
        key, label = check_map[code]
        state = technical.get(key)
        if isinstance(state, dict):
            if state.get("invalidated") is True:
                bullets.append(f"{label} был подтверждён, но затем отменён последующим закрытием 1H.")
            elif state.get("confirmed") is True:
                bullets.append(f"{label} подтверждён.")
            elif state.get("confirmed") is False:
                reason = str(state.get("reason") or "").lower()
                if "unavailable" in reason or "insufficient" in reason:
                    bullets.append(f"{label}: данных недостаточно.")
                else:
                    bullets.append(f"{label} не подтверждён.")
            else:
                bullets.append(f"{label}: нет данных.")
        else:
            bullets.append(f"{label}: нет данных.")

    if code == "LONG B":
        sequence = technical.get("spot_sequence_ok")
        if sequence is True:
            bullets.append("Spot Imbalance положительный в трёх разнесённых примерно на 4H снимках.")
        elif sequence is False:
            bullets.append("Последовательность положительного Spot Imbalance не подтверждена.")
        else:
            bullets.append("Истории 4H Spot Imbalance недостаточно.")

    context15 = technical.get("context_15m") or {}
    if isinstance(context15, dict) and isinstance(context15.get("last_closed_change_pct"), (int, float)):
        direction15 = str(context15.get("direction") or "UNKNOWN").upper()
        direction_label = {"BULLISH": "бычья", "BEARISH": "медвежья", "DOJI": "нейтральная"}.get(direction15, "неизвестная")
        bullets.append(
            f"15m контекст (последняя закрытая свеча): {direction_label}, "
            f"изменение {_number(context15.get('last_closed_change_pct'), 2, signed=True)}%. "
            "Это контекст тайминга, не отдельное условие P1."
        )
    elif isinstance(context15, dict) and context15.get("error"):
        bullets.append("15m контекст недоступен; отдельного формального числового порога для него P1 не задаёт.")

    range24 = technical.get("range_24h_1h") or {}
    distance_low = range24.get("distance_from_low_pct") if isinstance(range24, dict) else None
    low24 = range24.get("low") if isinstance(range24, dict) else None
    if isinstance(low24, (int, float)) and isinstance(distance_low, (int, float)):
        bullets.append(
            f"Диапазон Binance 24H по закрытым 1H свечам: low {_number(low24, 8)}, "
            f"закрытие выше low на {_number(distance_low, 2)}%. "
            "P1 не задаёт числовую границу слова «рядом с минимумом»; это справочная оценка."
        )

    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    if isinstance(invalidation.get("level"), (int, float)):
        if result.direction.upper() == "LONG":
            invalidation_rule = "закрытие 1H ниже уровня отменяет LONG"
        else:
            invalidation_rule = "закрытие 1H выше уровня отменяет SHORT"
        source_labels = {
            "confirmed 1H higher-low pivot": "последний подтверждённый минимум Higher Low 1H",
            "confirmed 1H breakout/retest level": "уровень пробоя и ретеста 1H",
            "confirmed 1H weak-bounce resistance": "сопротивление слабого 1H-отскока",
            "confirmed failed 1H breakout level": "уровень неудавшегося пробоя 1H",
            "confirmed 1H bearish structure level": "уровень медвежьей структуры 1H",
            "confirmed 1H breakdown/retest level": "уровень пробоя и ретеста вниз 1H",
        }
        source_label = source_labels.get(str(invalidation.get("source")), str(invalidation.get("source") or "источник не определён"))
        bullets.append(
            f"Уровень отмены сценария: {_number(invalidation['level'], 8)}; "
            f"{invalidation_rule} ({source_label})."
        )
    else:
        bullets.append("Уровень отмены сценария по закрытым 1H свечам: не определён — не придумывать уровень.")
    return bullets


def _diagnostic_bullets(row: EnrichedRow) -> list[str]:
    m = row.market
    bullets: list[str] = []
    if m.oi_chg4h_usd is not None and abs(m.oi_chg4h_usd) < 2_000_000:
        bullets.append(f"OI 4H изменение мало в USD: { _money(m.oi_chg4h_usd) } (<$2 млн по P1).")
    if row.delta_funding_oiw_pp is not None and m.fr_oiw_pct is not None and m.pfr_oiw_pct is not None:
        if m.fr_oiw_pct * m.pfr_oiw_pct < 0:
            bullets.append("ВНИМАНИЕ: прогнозный PFR OI_W меняет знак относительно текущего FR OI_W; P1 не задаёт числовой порог, это предупреждение, не автоматический запрет.")
    f, s = m.futures_imbalance_4h, m.spot_imbalance_4h
    if f is not None and s is not None and f * s < 0:
        bullets.append("Futures и Spot расходятся по знаку; взаимного подтверждения потоков нет.")
    if m.chg24_pct is not None and s is not None:
        if m.chg24_pct > 0 and s < 0:
            bullets.append("Цена 24H растёт, Spot Imbalance отрицательный: расхождение цены и спота.")
        elif m.chg24_pct < 0 and s > 0:
            bullets.append("Цена 24H падает, Spot Imbalance положительный: расхождение цены и спота.")
    if m.chg24_pct is not None and m.oi_chg24_pct is not None:
        if m.chg24_pct > 0 and m.oi_chg24_pct < 0:
            bullets.append("Цена 24H растёт при падающем OI 24H: возможно закрытие SHORT, не доказательство нового устойчивого LONG.")
        elif m.chg24_pct < 0 and m.oi_chg24_pct < 0:
            bullets.append("Цена 24H падает при падающем OI 24H: возможно закрытие LONG, не доказательство открытия нового SHORT.")
        elif m.chg24_pct < 0 and m.oi_chg24_pct > 0:
            bullets.append("Цена 24H падает при растущем OI 24H: позиции наращиваются во время снижения; направление по OI не определяется.")
    if m.fr_oiw_pct is not None and m.fr_oiw_pct >= 0.03:
        bullets.append("Текущий Funding OI_W достиг порога перегретого LONG funding (≥ +0,03% по P1).")
    elif m.fr_oiw_pct is not None and m.fr_oiw_pct <= -0.005:
        bullets.append("Текущий Funding OI_W отрицательный и достиг порога ≤ -0,005% по P1.")
    if m.ls_ratio_1d is not None and m.ls_ratio_1d >= 2.5:
        bullets.append("L/S 1D ≥ 2,5: выраженный перекос аккаунтов в LONG; это не доля капитала.")
    elif m.ls_ratio_1d is not None and m.ls_ratio_1d <= 0.9:
        bullets.append("L/S 1D ≤ 0,9: выраженный перекос аккаунтов в SHORT; это не доля капитала.")
    if row.oi_drop_mostly_voluntary is True and row.liquidation_coverage_of_oi_drop_pct is not None:
        bullets.append(
            f"Ликвидации покрывают только {_number(row.liquidation_coverage_of_oi_drop_pct, 2)}% снижения OI 24H (<10%): "
            "снижение OI, вероятно, в основном связано с добровольным закрытием, а не с каскадом."
        )
    return bullets


def _conclusion(result: ScenarioResult, row: EnrichedRow, technical: dict[str, Any]) -> str:
    status = result.status.upper()
    code = _scenario_code(result.scenario)
    if status == "INVALIDATED_1H_STRUCTURE":
        return "Сценарий отменён: последующая закрытая 1H-свеча нарушила ранее подтверждённую структуру. Старое подтверждение больше не действует; ждать нового сетапа."

    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    last_1h = technical.get("last_closed_1h") or {}
    close_1h = last_1h.get("close")
    level_1h = invalidation.get("level")
    if isinstance(close_1h, (int, float)) and isinstance(level_1h, (int, float)):
        breached = close_1h < level_1h if result.direction.upper() == "LONG" else close_1h > level_1h
        if breached:
            side_word = "ниже" if result.direction.upper() == "LONG" else "выше"
            return (
                f"Последнее закрытие 1H находится {side_word} подтверждённой точки отмены "
                f"({level_1h:g}). Структура нарушена; не входить и ждать нового подтверждения."
            )

    # Keep a safe explanation for local/debug formatting of non-alert states.
    if status == "WAIT_BOUNCE":
        coverage = row.liquidation_coverage_of_oi_drop_pct
        if coverage is not None and coverage < 10:
            return (
                f"Признаки SHORT D выполнены {result.passed_count}/{result.total_count}, но ликвидации объясняют "
                f"только {_number(coverage, 2)}% снижения OI. Каскадная интерпретация ослаблена; SHORT у минимума "
                "запрещён. Ждать слабый отскок и подтверждённый отказ от сопротивления."
            )
        return "Признаки long-liquidation cascade есть, но SHORT у локального дна запрещён. Ждать подтверждённый слабый отскок и отказ от сопротивления."
    if status == "PREPARATION":
        return "Выполнена только стадия подготовки Short Squeeze. Триггера нет; LONG не открывать."
    if status == "WAITING_FOR_HIGHER_LOW":
        return "Higher Low ещё не подтверждён закрытыми свечами 1H. LONG D не подтверждён; не входить."
    if status == "WAITING_FOR_1H_BREAKOUT_RETEST":
        return "Закрытый 1H пробой и успешный ретест вверх ещё не подтверждены. Не входить."
    if status == "WAITING_FOR_1H_BREAKDOWN_RETEST":
        return "Закрытый 1H пробой вниз и успешный ретест ещё не подтверждены. Не входить."
    if status == "WAITING_FOR_1H_STRUCTURE_BREAK":
        return "Медвежий пробой структуры 1H ещё не подтверждён. Не входить."
    if status.startswith("WAITING_"):
        return "Условия P1 предварительно выполнены, но обязательное подтверждение ещё отсутствует. Не входить."

    # A Telegram message may be created only for these two states by the
    # notification collector. Each branch documents the exact confirmation.
    confirmed = status in {"CONFIRMED_SIGNAL", "CONFIRMED_SIGNAL_WITH_WARNING"}
    if confirmed:
        confirmations = {
            "LONG A": "Табличные условия LONG A и закрытый 1H пробой с успешным ретестом подтверждены.",
            "LONG B": "Табличные условия LONG B, три независимых положительных 4H Spot-снимка и Higher Low 1H подтверждены.",
            "LONG C": "Обе стадии LONG C и растущая последняя закрытая 1H-свеча подтверждены.",
            "LONG D": "Табличные условия LONG D и Higher Low 1H подтверждены.",
            "SHORT A": "Табличные условия SHORT A и закрытый 1H пробой вниз с успешным ретестом подтверждены.",
            "SHORT B": "Условия Spot Distribution и закрытый 1H пробой вниз с успешным ретестом подтверждены.",
            "SHORT C": "Табличные условия SHORT C, медвежий пробой структуры 1H и успешный ретест уровня подтверждены.",
            "SHORT D": "Условия liquidation cascade, слабый отскок и отказ от сопротивления на 1H подтверждены. Не открывать SHORT у минимума.",
            "SHORT E": "Три сопоставимых 4H-снимка, разворот F/S вниз и неудавшийся пробой вверх на 1H подтверждены.",
        }
        conclusion = confirmations.get(code, "Сценарий и его обязательное техническое подтверждение выполнены.")
        if status == "CONFIRMED_SIGNAL_WITH_WARNING":
            conclusion += " Предупреждение: прогнозный PFR OI_W меняет знак относительно текущего FR OI_W; P1 не задаёт порог, поэтому это качественный фактор риска."

        context = technical.get("context_4h") or {}
        ema20, ema50 = context.get("ema20"), context.get("ema50")
        if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
            opposing = (result.direction.upper() == "LONG" and ema20 < ema50) or (result.direction.upper() == "SHORT" and ema20 > ema50)
            if opposing:
                conclusion += " Внимание: EMA20/EMA50 на 4H направлены против сценария — это противоречащий контекст, проверь риск особенно внимательно."
        return conclusion + " Перед входом отдельно проверь уровень отмены, стоп, риск, ликвидность и издержки."

    if status == "OBSERVATION_WITH_WARNING":
        return "Числовые условия P1 выполнены, но прогнозный Funding меняет знак относительно текущего; сценарий не подтверждён для отправки."
    if status == "TRIGGER":
        return "Обе стадии Short Squeeze выполнены. Это триггер для проверки графика, не команда на вход."
    if status == "WAITING_FOR_HISTORY_OR_HIGHER_LOW":
        return "Для LONG B не хватает трёх положительных 4H Spot-снимков и/или Higher Low. Не входить."
    if status == "WAITING_FOR_1H_STRUCTURE_BREAK":
        return "Табличные условия выполнены, но пробой структуры 1H не подтверждён. Не входить."
    return "Обязательные подтверждения отсутствуют. Не входить."


def format_alert(
    result: ScenarioResult,
    row: EnrichedRow,
    technical: dict[str, Any] | None = None,
    *,
    fetched_at: datetime | None = None,
    snapshot_delta: dict[str, Any] | None = None,
) -> str:
    m = row.market
    technical = technical or result.technical or {}
    direction = str(result.direction or "").strip().upper()
    if direction == "LONG":
        direction_emoji = "🟢"
    elif direction == "SHORT":
        direction_emoji = "🔴"
    else:
        direction_emoji = "⚪"
    # The ticker is taken from the parsed Coinalyze market row itself. Keep the
    # quote asset (USDT/USDC/etc.); strip only venue/perpetual suffixes.
    contract_symbol = _contract_symbol(m.symbol)
    lines = [
        f"{direction_emoji} <b>{_esc(direction or 'UNKNOWN')}</b> · <code>{_esc(contract_symbol)}</code>",
        f"<b>{_esc(_scenario_code(result.scenario))} · {_esc(_status_label(result.status))} · {result.passed_count}/{result.total_count}</b>",
    ]
    if fetched_at is not None:
        timestamp = fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(f"Снимок Coinalyze: <code>{timestamp}</code>")
    lines.extend([
        "",
        f"F = <code>{_number(m.futures_imbalance_4h, 4, signed=True)}</code> · S = <code>{_number(m.spot_imbalance_4h, 4, signed=True)}</code>",
        f"Buy F/S: <code>{_percent(row.futures_buy_share_pct, 2, signed=False)}</code> / "
        f"<code>{_percent(row.spot_buy_share_pct, 2, signed=False)}</code>; "
        f"Spread S−F: <code>{_number(row.spread, 4, signed=True)}</code>.",
        "",
        f"- OI 4H: <code>{_percent(m.oi_chg4h_pct, 2)}</code> / <code>{_money(m.oi_chg4h_usd)}</code>" + (" (малое в USD)" if m.oi_chg4h_usd is not None and abs(m.oi_chg4h_usd) < 2_000_000 else ""),
        f"- OI 24H: <code>{_percent(m.oi_chg24_pct, 2)}</code> / <code>{_money(m.oi_chg24_usd)}</code>.",
        f"- Цена 24H: <code>{_percent(m.chg24_pct, 2)}</code>.",
        f"- Funding OI_W: FR <code>{_percent(m.fr_oiw_pct, 4)}</code>; PFR <code>{_percent(m.pfr_oiw_pct, 4)}</code>; Δ <code>{_number(row.delta_funding_oiw_pp, 4, signed=True)} п.п.</code>.",
        f"- FR AVG − FR AVG OI_W: <code>{_number(row.fr_venue_dispersion_pp, 4, signed=True)} п.п.</code>.",
        f"- L/S 1D: <code>{_number(m.ls_ratio_1d, 2)}</code> (по аккаунтам, не по размеру капитала).",
    ])
    if m.btc_corr7d is not None:
        corr_note = " — высокая положительная связь с BTC; не независимая экспозиция" if m.btc_corr7d > 0.7 and _base_symbol(result.symbol) != "BTC" else ""
        lines.append(f"- BTC Corr. 7D: <code>{_number(m.btc_corr7d, 2, signed=True)}</code>{_esc(corr_note)}.")
    else:
        lines.append("- BTC Corr. 7D: n/a (корреляция не проверена).")
    if row.rel_strength_vs_btc_pp is not None:
        lines.append(f"- RelStrength к BTC: <code>{_number(row.rel_strength_vs_btc_pp, 2, signed=True)} п.п.</code>.")
    if row.long_liq_share is not None:
        lines.append(f"- Доля ликвидаций LONG: <code>{_number(row.long_liq_share * 100, 2)}%</code>.")
    if row.liq_imbalance is not None:
        lines.append(f"- LiqImb (Short−Long): <code>{_number(row.liq_imbalance * 100, 2, signed=True)}%</code>.")
    if row.liquidation_coverage_of_oi_drop_pct is not None:
        lines.append(f"- Liqs. 24H / падение OI 24H: <code>{_number(row.liquidation_coverage_of_oi_drop_pct, 2)}%</code>.")

    lines.extend(f"- {_esc(item)}" for item in _technical_bullets(result, technical))
    lines.extend(f"⚠️ {_esc(item)}" for item in _diagnostic_bullets(row))

    if snapshot_delta and snapshot_delta.get("price_x_oi_classification") not in {None, "no_new_table_data"}:
        price_delta = snapshot_delta.get("price_change_pct")
        oi_delta = snapshot_delta.get("oi_change_pct")
        interval = snapshot_delta.get("interval_hours")
        lines.append(
            f"- Между снимками ({_number(interval, 2)} ч): ΔPrice <code>{_number(price_delta, 2, signed=True)}%</code>; "
            f"ΔOI <code>{_number(oi_delta, 2, signed=True)}%</code>. {_esc(snapshot_delta.get('window_note'))}."
        )

    orderbook = technical.get("orderbook") or {}
    if orderbook.get("spread_bps") is not None:
        lines.append(
            f"- Стакан Binance (справочно): spread <code>{_number(orderbook.get('spread_bps'), 2)} bps</code>; "
            f"depth ±1% bid/ask <code>{_money(orderbook.get('bid_depth_1pct_usd'), signed=False)} / {_money(orderbook.get('ask_depth_1pct_usd'), signed=False)}</code>."
        )

    lines.extend(["", f"<b>Вывод:</b> {_esc(_conclusion(result, row, technical))}"])
    text = "\n".join(lines) + DISCLAIMER
    return _safe_telegram_truncate(text)




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
        for chat_id in recipients:
            last_error: str | None = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        url,
                        json={
                            "chat_id": chat_id,
                            "text": _safe_telegram_truncate(text),
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True,
                        },
                        timeout=self.timeout_sec,
                    )
                    if response.status_code == 429:
                        try:
                            retry_after = float((response.json().get("parameters") or {}).get("retry_after", 1))
                        except (ValueError, TypeError, AttributeError):
                            retry_after = 1.0
                        last_error = f"Telegram rate limit (429); retry_after={retry_after}s"
                        if attempt < 2:
                            time.sleep(min(max(retry_after, 1.0), 30.0))
                            continue
                        raise RuntimeError(last_error)
                    if response.status_code >= 500 and attempt < 2:
                        last_error = f"Telegram server error {response.status_code}"
                        time.sleep(2 ** attempt)
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    if not payload.get("ok"):
                        raise RuntimeError(payload.get("description", "Telegram API returned ok=false"))
                    message_id = (payload.get("result") or {}).get("message_id")
                    results.append({"chat_id": chat_id, "sent": True, "message_id": message_id})
                    break
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    last_error = str(exc)
                    if attempt < 2 and isinstance(exc, requests.RequestException):
                        time.sleep(2 ** attempt)
                        continue
                    log.error("Telegram delivery failed for chat_id=%s: %s", chat_id, last_error)
                    results.append({"chat_id": chat_id, "sent": False, "error": last_error})
                    break
        return results
