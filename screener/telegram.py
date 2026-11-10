from __future__ import annotations

import html
import logging
import math
import time
from html.parser import HTMLParser
from datetime import datetime, timezone
from typing import Any

import requests

from .metrics import EnrichedRow, _rank_tail_count
from .models import MarketStats, ScenarioResult
from .structure import scenario_invalidation

log = logging.getLogger(__name__)
DISCLAIMER = (
    "\n\n<i>Наблюдение по P1, не торговый сигнал и не команда на вход. "
    "Вход требует проверки графика и риска. Параметры P1 не проходили backtest.</i>"
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
        "Наблюдение по P1, не торговый сигнал и не команда на вход. "
        "Вход требует проверки графика и риска. Параметры P1 не проходили backtest."
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
        "OBSERVATION_AFTER_BOUNCE": "OBSERVATION_AFTER_BOUNCE",
        "OBSERVATION_WITH_WARNING": "OBSERVATION · WARNING",
        "WAITING_FOR_HISTORY_OR_HIGHER_LOW": "WAITING_HISTORY / HIGHER LOW",
        "WAITING_FOR_1H_STRUCTURE_BREAK": "WAITING_FOR_1H_BREAK",
        "BLOCKED_QUALITY": "BLOCKED_QUALITY",
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
            if "unavailable" in reason or "insufficient candles" in reason:
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
        "SHORT C": ("bearish_structure_break", "Медвежий пробой 1H"),
        "SHORT D": ("weak_bounce_rejection", "Отказ от сопротивления после 1H-отскока"),
        "SHORT E": ("failed_breakout_rejection", "Ложный пробой вверх на 1H"),
    }
    if code in check_map:
        key, label = check_map[code]
        state = technical.get(key)
        if isinstance(state, dict):
            if state.get("confirmed") is True:
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
            "latest confirmed 1H higher-low pivot": "последний подтверждённый минимум Higher Low 1H",
            "1H breakout/retest level": "уровень пробоя и ретеста 1H",
            "1H weak-bounce resistance": "сопротивление слабого 1H-отскока",
            "failed 1H breakout level": "уровень неудавшегося пробоя 1H",
            "1H bearish structure level": "уровень медвежьей структуры 1H",
            "1H swing-low structure level": "уровень минимума структуры 1H",
            "1H breakdown/retest level": "уровень пробоя и ретеста вниз 1H",
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
    if status == "WAIT_BOUNCE":
        coverage = row.liquidation_coverage_of_oi_drop_pct
        if coverage is not None and coverage < 10:
            return (
                f"Признаки SHORT D выполнены {result.passed_count}/{result.total_count}, но ликвидации объясняют "
                f"только {_number(coverage, 2)}% снижения OI. Каскадная интерпретация ослаблена; SHORT у минимума "
                "запрещён. Ждать слабый отскок и подтверждённый отказ от сопротивления."
            )
        return "Признаки long-liquidation cascade выполнены. SHORT у локального дна запрещён: ждать слабый отскок и подтверждённый отказ от сопротивления."
    if status == "OBSERVATION_AFTER_BOUNCE":
        return "Отскок и отказ от сопротивления подтверждены. Это наблюдение, не точка входа; проверить структуру, стоп и риск."
    if status == "OBSERVATION_WITH_WARNING":
        return "Числовые условия P1 выполнены, но прогнозный Funding меняет знак относительно текущего. Это качественное предупреждение без заданного P1 порога: сценарий спорный, не считать его безоговорочным наблюдением и не входить автоматически."
    if status == "PREPARATION":
        return "Выполнена только стадия подготовки Short Squeeze. Триггера нет; LONG не открывать."
    if status == "TRIGGER":
        return "Обе стадии Short Squeeze выполнены. Это триггер для проверки графика, не команда на вход."
    if status == "WAITING_FOR_HISTORY_OR_HIGHER_LOW":
        return "Табличные условия выполнены, но для LONG B не хватает трёх положительных 4H Spot-снимков и/или Higher Low. Не входить."
    if status == "WAITING_FOR_1H_STRUCTURE_BREAK":
        return "Табличные условия выполнены, но пробой структуры 1H не подтверждён. Не входить."

    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    close = context.get("last_close")
    if result.direction.upper() == "LONG":
        support = context.get("last_confirmed_support")
        if isinstance(close, (int, float)) and isinstance(support, (int, float)) and close < support:
            return "Табличные условия выполнены, но последнее закрытие 4H ниже подтверждённой поддержки. Графическая структура против LONG. Не входить."
        if code == "LONG A":
            retest = technical.get("trend_retest_long")
            if not isinstance(retest, dict) or retest.get("confirmed") is not True:
                return "Табличные условия выполнены, но LONG breakout/retest не подтверждён закрытыми свечами. Не входить до подтверждения."
        if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 < ema50:
            return "4H EMA20 ниже EMA50 — медвежий контекст против LONG. Даже при подтверждённом 1H Higher Low/retest это контртрендовая ситуация: только наблюдение, не входить автоматически."
        if result.status == "OBSERVATION":
            return "Условия P1 выполнены. График не отменяет сценарий, но статус остаётся наблюдательным; перед входом отдельно проверить точку отмены и риск."
        return "Подтверждённого LONG-входа нет. Наблюдать, не входить."

    resistance = context.get("last_confirmed_resistance")
    if isinstance(close, (int, float)) and isinstance(resistance, (int, float)) and close > resistance:
        return "Табличные условия выполнены, но последнее закрытие 4H выше подтверждённого сопротивления. Графическая структура против SHORT. Не входить."
    if code == "SHORT A":
        retest = technical.get("trend_retest_short")
        if not isinstance(retest, dict) or retest.get("confirmed") is not True:
            return "Табличные условия выполнены, но SHORT breakdown/retest не подтверждён закрытыми свечами. Не входить до подтверждения."
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 > ema50:
        return "4H EMA20 выше EMA50 — бычий контекст против SHORT. Даже при подтверждённом 1H пробое это противоречащий фактор: только наблюдение, не входить автоматически."
    if result.status == "OBSERVATION":
        return "Условия P1 выполнены. Статус наблюдательный; проверить подтверждение структуры, точку отмены и риск."
    return "Подтверждённого SHORT-входа нет. Наблюдать, не входить."


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


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})" if total else "n/a"


def _market_asset_line(symbol: str, rows: list[EnrichedRow]) -> str | None:
    item = next((row for row in rows if _base_symbol(row.symbol) == symbol), None)
    if item is None:
        return None
    m = item.market
    return (
        f"{symbol}: Цена 24H {_percent(m.chg24_pct, 2)}; "
        f"OI 24H {_percent(m.oi_chg24_pct, 2)}/{_money(m.oi_chg24_usd)}; "
        f"OI 4H {_percent(m.oi_chg4h_pct, 2)}/{_money(m.oi_chg4h_usd)}; "
        f"F/S {_number(m.futures_imbalance_4h, 4, signed=True)}/{_number(m.spot_imbalance_4h, 4, signed=True)}; "
        f"FR OI_W {_percent(m.fr_oiw_pct, 4)}; L/S 1D {_number(m.ls_ratio_1d, 2)}; "
        f"BTC Corr. 7D {_number(m.btc_corr7d, 2, signed=True)}"
    )


def format_market_summary(
    rows: list[EnrichedRow],
    stats: MarketStats,
    fetched_at: datetime,
    contract_type: str,
    missing_columns: list[str] | None = None,
) -> str:
    timestamp = fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    markets = [item.market for item in rows]
    price_median_values = sorted(m.chg24_pct for m in markets if m.chg24_pct is not None)
    price_median = None
    if price_median_values:
        middle = len(price_median_values) // 2
        price_median = (
            price_median_values[middle]
            if len(price_median_values) % 2
            else (price_median_values[middle - 1] + price_median_values[middle]) / 2
        )

    lines = [
        "<b>СВОДКА РЫНКА P1</b>",
        f"Снимок: <code>{timestamp}</code> · Контракты: <code>{_esc(contract_type)}</code> · Строк: <code>{len(rows)}</code>",
        "",
        f"Медианы: F <code>{_number(stats.median_f, 4, signed=True)}</code>; S <code>{_number(stats.median_s, 4, signed=True)}</code>; "
        f"OI 24H <code>{_percent(stats.median_oi24_pct, 2)}</code> / <code>{_money(stats.median_oi24_usd)}</code>; "
        f"OI 4H <code>{_percent(stats.median_oi4h_pct, 2)}</code> / <code>{_money(stats.median_oi4h_usd)}</code>; "
        f"Цена 24H <code>{_percent(price_median, 2)}</code>.",
        f"Распределение: F<0 <code>{_share(stats.f_negative_count, stats.f_count)}</code>; "
        f"S<0 <code>{_share(stats.s_negative_count, stats.s_count)}</code>; "
        f"OI 24H<0 <code>{_share(stats.oi24_negative_count, stats.oi24_count)}</code>; "
        f"Цена 24H>0 <code>{_share(stats.price_positive_count, stats.price_count)}</code>.",
    ]
    for symbol in ("BTC", "ETH"):
        asset = _market_asset_line(symbol, rows)
        if asset:
            lines.append(asset)

    valid_f = [item for item in rows if item.market.futures_imbalance_4h is not None]
    valid_s = [item for item in rows if item.market.spot_imbalance_4h is not None]
    f_tail = _rank_tail_count(len(valid_f), 0.15)
    s_tail = _rank_tail_count(len(valid_s), 0.15)

    def ranked_line(label: str, values: list[EnrichedRow], metric: str, adjusted: str, reverse: bool) -> None:
        ordered = sorted(values, key=lambda item: getattr(item.market, metric), reverse=reverse)
        selected = ordered[: (f_tail if metric == "futures_imbalance_4h" else s_tail)]
        if not selected:
            return
        chunks = [
            f"{_base_symbol(item.symbol)} {_number(getattr(item.market, metric), 4, signed=True)} "
            f"(adj {_number(getattr(item, adjusted), 4, signed=True)})"
            for item in selected
        ]
        lines.append(f"{label}: " + "; ".join(chunks))

    if valid_f:
        ranked_line("F верхние 15% (n={})".format(f_tail), valid_f, "futures_imbalance_4h", "f_adj", True)
        ranked_line("F нижние 15% (n={})".format(f_tail), valid_f, "futures_imbalance_4h", "f_adj", False)
    if valid_s:
        ranked_line("S верхние 15% (n={})".format(s_tail), valid_s, "spot_imbalance_4h", "s_adj", True)
        ranked_line("S нижние 15% (n={})".format(s_tail), valid_s, "spot_imbalance_4h", "s_adj", False)

    f_s_valid = [item for item in rows if item.market.futures_imbalance_4h is not None and item.market.spot_imbalance_4h is not None]
    price_spot_valid = [item for item in rows if item.market.chg24_pct is not None and item.market.spot_imbalance_4h is not None]
    f_s_div = sum(1 for item in f_s_valid if item.market.futures_imbalance_4h * item.market.spot_imbalance_4h < 0)
    price_up_spot_down = sum(1 for item in price_spot_valid if item.market.chg24_pct > 0 and item.market.spot_imbalance_4h < 0)
    price_down_spot_up = sum(1 for item in price_spot_valid if item.market.chg24_pct < 0 and item.market.spot_imbalance_4h > 0)
    price_oi_valid = [item for item in rows if item.market.chg24_pct is not None and item.market.oi_chg24_pct is not None]
    price_up_oi_down = sum(1 for item in price_oi_valid if item.market.chg24_pct > 0 and item.market.oi_chg24_pct < 0)
    price_down_oi_down = sum(1 for item in price_oi_valid if item.market.chg24_pct < 0 and item.market.oi_chg24_pct < 0)
    price_down_oi_up = sum(1 for item in price_oi_valid if item.market.chg24_pct < 0 and item.market.oi_chg24_pct > 0)

    rs_rows = [item for item in rows if item.rel_strength_vs_btc_pp is not None]
    rs_positive = [item for item in rs_rows if item.rel_strength_vs_btc_pp > 0]
    rs_negative = [item for item in rs_rows if item.rel_strength_vs_btc_pp < 0]
    rs_top = sorted(rs_positive, key=lambda item: item.rel_strength_vs_btc_pp, reverse=True)[:5]
    rs_bottom = sorted(rs_negative, key=lambda item: item.rel_strength_vs_btc_pp)[:5]
    if rs_rows:
        lines.append(
            f"RelStrength к BTC: положительная <code>{len(rs_positive)}/{len(rs_rows)}</code>, "
            f"отрицательная <code>{len(rs_negative)}/{len(rs_rows)}</code>. "
            "Лидеры: "
            + (", ".join(f"{_base_symbol(x.symbol)} {_number(x.rel_strength_vs_btc_pp, 2, signed=True)} п.п." for x in rs_top) or "нет")
        )
        lines.append(
            "Отстают от BTC: "
            + (", ".join(f"{_base_symbol(x.symbol)} {_number(x.rel_strength_vs_btc_pp, 2, signed=True)} п.п." for x in rs_bottom) or "нет")
        )

    corr_valid = [item for item in rows if item.market.btc_corr7d is not None]
    corr_high = [item for item in corr_valid if item.market.btc_corr7d > 0.7]
    # Include BTC itself in the market statistic (as the other report does),
    # but exclude it from the list of altcoin exposures correlated with BTC.
    corr_linked = [item for item in corr_high if _base_symbol(item.symbol) != "BTC"]
    low_oi = [item for item in rows if item.market.oi_usd is not None and item.market.oi_usd < 40_000_000]
    low_oi4h = [item for item in rows if item.market.oi_chg4h_usd is not None and abs(item.market.oi_chg4h_usd) < 2_000_000]
    high_vol_mc = [item for item in rows if item.market.vol_mktcap_ratio is not None and item.market.vol_mktcap_ratio > 2.0]
    low_oi_vol = [item for item in rows if item.market.oi_vol_ratio is not None and item.market.oi_vol_ratio < 0.1]
    missing_spot = [item for item in rows if item.market.spot_imbalance_4h is None]
    funding_flip = [
        item for item in rows
        if item.market.fr_oiw_pct is not None and item.market.pfr_oiw_pct is not None
        and item.market.fr_oiw_pct * item.market.pfr_oiw_pct < 0
    ]
    voluntary = [
        item for item in rows
        if item.oi_drop_mostly_voluntary is True and item.liquidation_coverage_of_oi_drop_pct is not None
    ]

    lines.append(
        f"Расхождения: F/S разного знака <code>{f_s_div}/{len(f_s_valid)}</code>; "
        f"Цена↑/Spot↓ <code>{price_up_spot_down}/{len(price_spot_valid)}</code>; "
        f"Цена↓/Spot↑ <code>{price_down_spot_up}/{len(price_spot_valid)}</code>; "
        f"Цена↑/OI24↓ <code>{price_up_oi_down}/{len(price_oi_valid)}</code>; "
        f"Цена↓/OI24↓ <code>{price_down_oi_down}/{len(price_oi_valid)}</code>; "
        f"Цена↓/OI24↑ <code>{price_down_oi_up}/{len(price_oi_valid)}</code>."
    )
    lines.append(
        f"Качество/экспозиция: OI<$40 млн <code>{len(low_oi)}</code>; OI 4H <$2 млн <code>{len(low_oi4h)}</code>; "
        f"VOL24H/MktCap>2 <code>{len(high_vol_mc)}</code>; OI/VOL24H<0,1 <code>{len(low_oi_vol)}</code>; "
        f"BTC Corr. 7D>0,7 <code>{len(corr_high)}/{len(corr_valid)}</code>; связанных альтов <code>{len(corr_linked)}</code>; "
        f"Spot n/a <code>{len(missing_spot)}</code>."
    )
    if corr_linked:
        lines.append("Связанные с BTC альты (не независимые позиции): " + ", ".join(_base_symbol(item.symbol) for item in corr_linked[:12]) + ("…" if len(corr_linked) > 12 else ""))
    if funding_flip:
        lines.append(
            "PFR OI_W меняет знак относительно текущего FR OI_W (предупреждение, без выдуманного порога): "
            + ", ".join(_base_symbol(item.symbol) for item in funding_flip[:12])
            + ("…" if len(funding_flip) > 12 else "")
        )
    if voluntary:
        lines.append(
            "Покрытие падения OI ликвидациями <10% (вероятно, преобладает добровольное закрытие): "
            + ", ".join(
                f"{_base_symbol(item.symbol)} {_number(item.liquidation_coverage_of_oi_drop_pct, 2)}%"
                for item in sorted(voluntary, key=lambda item: item.liquidation_coverage_of_oi_drop_pct)[:10]
            )
        )
    if missing_spot:
        lines.append("Spot Imbalance отсутствует: " + ", ".join(_base_symbol(item.symbol) for item in missing_spot[:12]) + ("…" if len(missing_spot) > 12 else ""))
    if len(rows) < 20:
        lines.append("⚠️ Менее 20 строк: ранги верхних/нижних 15% ненадёжны.")
    if missing_columns:
        lines.append("Не распознаны колонки Coinalyze (сигналы при обязательных пропусках блокируются): " + _esc(", ".join(missing_columns)))
    lines.extend([
        "",
        "Интерпретация: доли и ранги описывают только эту выборку; F/S — нормированные дисбалансы, L/S — доли аккаунтов, рост OI не указывает сторону позиции.",
        "P1 — стартовые гипотезы без backtest; по одному показателю вход не определяется.",
    ])
    text = "\n".join(lines)
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
