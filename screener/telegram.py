from __future__ import annotations

import html
import logging
import math
import re
from typing import Any

import requests

from .metrics import EnrichedRow
from .models import ScenarioResult

log = logging.getLogger(__name__)

DISCLAIMER = (
    "\n\n<i>Наблюдение по P1, не торговый сигнал и не команда на вход. "
    "Вход требует проверки графика и риска.</i>"
)


def _esc(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=False)


def _decimal(value: float | int | None, precision: int = 2, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        rendered = f"{number:+,.{precision}f}" if signed else f"{number:,.{precision}f}"
    except (TypeError, ValueError):
        return "n/a"
    return rendered.replace(",", "\u0000").replace(".", ",").replace("\u0000", " ")


def _percent(value: float | int | None, precision: int = 2, *, signed: bool = True) -> str:
    rendered = _decimal(value, precision, signed=signed)
    return f"{rendered}%" if rendered != "n/a" else "n/a"


def _display_symbol(symbol: str) -> str:
    base = str(symbol or "").upper().split(".", 1)[0]
    base = re.sub(r"[-_]?PERPETUAL$|[-_]?PERP$", "", base)
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "BNB", "EUR"):
        if base.endswith(quote) and len(base) > len(quote):
            base = base[: -len(quote)]
            break
    return base or str(symbol)


def _scenario_code(scenario: str) -> str:
    return str(scenario or "").split(" — ", 1)[0].strip() or "SCENARIO"


def _confirmed(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    flag = value.get("confirmed")
    return flag if isinstance(flag, bool) else None


def _looks_unavailable(value: dict[str, Any]) -> bool:
    reason = str(value.get("reason") or "").lower()
    return any(token in reason for token in ("unavailable", "insufficient candles", "candle data", "error"))


def _technical_bullets(result: ScenarioResult, technical: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
        relation = "выше" if ema20 > ema50 else "ниже" if ema20 < ema50 else "на уровне"
        bullets.append(f"4H EMA20 {relation} EMA50.")
    else:
        trend = str(context.get("trend") or "")
        if "unavailable" in trend.lower() or "insufficient" in trend.lower():
            bullets.append("4H EMA20/EMA50: свечных данных недостаточно.")

    close = context.get("last_close")
    level_key = "last_confirmed_support" if result.direction.upper() == "LONG" else "last_confirmed_resistance"
    level = context.get(level_key)
    level_name = "подтверждённой поддержки" if result.direction.upper() == "LONG" else "подтверждённого сопротивления"
    if isinstance(close, (int, float)) and isinstance(level, (int, float)):
        if close < level:
            relation = "ниже"
        elif close > level:
            relation = "выше"
        else:
            relation = "на уровне"
        bullets.append(f"Последнее закрытие 4H {relation} {level_name}.")
    elif context:
        bullets.append(f"Уровень {level_name} по 4H: нет данных.")
    else:
        bullets.append("4H контекст: нет данных.")

    higher = technical.get("higher_low") or {}
    higher_state = _confirmed(higher)
    if higher_state is True:
        bullets.append("Higher Low подтверждён.")
    elif higher_state is False:
        if _looks_unavailable(higher):
            bullets.append("Higher Low: свечные данные недоступны.")
        elif "need at least two" in str(higher.get("reason") or "").lower():
            bullets.append("Higher Low не подтверждён: недостаточно подтверждённых минимумов.")
        else:
            bullets.append("Higher Low отсутствует.")
    else:
        bullets.append("Higher Low: нет данных.")

    code = _scenario_code(result.scenario)
    if code == "LONG A":
        retest = technical.get("trend_retest_long") or technical.get("1h_retest") or {}
        state = _confirmed(retest)
        if state is True:
            bullets.append("LONG retest подтверждён.")
        elif state is False:
            bullets.append("LONG retest: свечные данные недоступны." if _looks_unavailable(retest) else "LONG retest не подтверждён.")
        else:
            bullets.append("LONG retest: нет данных.")
    elif code == "SHORT A":
        retest = technical.get("trend_retest_short") or technical.get("1h_retest") or {}
        state = _confirmed(retest)
        if state is True:
            bullets.append("SHORT retest подтверждён.")
        elif state is False:
            bullets.append("SHORT retest: свечные данные недоступны." if _looks_unavailable(retest) else "SHORT retest не подтверждён.")
        else:
            bullets.append("SHORT retest: нет данных.")
    elif code == "SHORT C":
        structure = technical.get("bearish_structure_break") or {}
        state = _confirmed(structure)
        if state is True:
            bullets.append("Медвежий пробой 1H подтверждён.")
        elif state is False:
            bullets.append("Медвежий пробой 1H не подтверждён.")
    elif code == "SHORT D":
        structure = technical.get("weak_bounce_rejection") or {}
        state = _confirmed(structure)
        if state is True:
            bullets.append("Отбой слабого 1H-отскока от сопротивления подтверждён.")
        elif state is False:
            bullets.append("Отказ от сопротивления после 1H-отскока не подтверждён.")
    elif code == "SHORT E":
        structure = technical.get("failed_breakout_rejection") or {}
        state = _confirmed(structure)
        if state is True:
            bullets.append("Ложный пробой вверх на 1H подтверждён.")
        elif state is False:
            bullets.append("Ложный пробой вверх на 1H не подтверждён.")
    return bullets


def _conclusion(result: ScenarioResult, technical: dict[str, Any]) -> str:
    if result.status == "WAIT_BOUNCE":
        return "Есть признаки каскада ликвидаций. SHORT у локального дна запрещён: ждать слабого отскока и подтверждённого отказа от сопротивления."
    if result.status == "OBSERVATION_AFTER_BOUNCE":
        return "Отскок и отказ от сопротивления обнаружены. Это наблюдение, не точка входа; отдельно проверить структуру и риск."
    if result.status == "TRIGGER":
        return "Условия P1 совпали, но это не команда на вход. Перед решением обязательны проверка графика, стопа и размера позиции."

    context = technical.get("context_4h") or {}
    ema20, ema50 = context.get("ema20"), context.get("ema50")
    close, support = context.get("last_close"), context.get("last_confirmed_support")
    close_below_support = (
        isinstance(close, (int, float)) and isinstance(support, (int, float)) and close < support
    )
    code = _scenario_code(result.scenario)

    if result.direction.upper() == "LONG":
        ema_against = isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 < ema50
        retest = technical.get("trend_retest_long") or technical.get("1h_retest") or {}
        retest_state = _confirmed(retest)
        if ema_against or close_below_support:
            return "Табличные условия выполнены, но графическая структура против LONG. Не входить."
        if code == "LONG A" and retest_state is False:
            return "Табличные условия хорошие, но LONG retest не подтверждён. Не входить до пробоя и успешного ретеста."
        if code == "LONG A" and retest_state is None:
            return "Табличные условия есть, но данных свечей недостаточно для проверки LONG retest. Не входить."
        if result.status == "OBSERVATION":
            return "Условия для наблюдения выполнены. Это не точка входа: дождаться подтверждения структуры и проверить риск."
        return "Есть предварительное совпадение условий, но подтверждённого входа нет. Наблюдать, не входить."

    resistance = context.get("last_confirmed_resistance")
    close_above_resistance = (
        isinstance(close, (int, float)) and isinstance(resistance, (int, float)) and close > resistance
    )
    ema_against = isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)) and ema20 > ema50
    retest = technical.get("trend_retest_short") or technical.get("1h_retest") or {}
    retest_state = _confirmed(retest)
    if ema_against or close_above_resistance:
        return "Табличные условия выполнены, но графическая структура против SHORT. Не входить."
    if code == "SHORT A" and retest_state is False:
        return "Табличные условия хорошие, но SHORT retest не подтверждён. Не входить до пробоя и успешного ретеста снизу."
    if code == "SHORT A" and retest_state is None:
        return "Табличные условия есть, но данных свечей недостаточно для проверки SHORT retest. Не входить."
    if result.status == "OBSERVATION":
        return "Условия для наблюдения выполнены. Это не точка входа: дождаться подтверждения структуры и проверить риск."
    return "Есть предварительное совпадение условий, но подтверждённого входа нет. Наблюдать, не входить."


def format_alert(result: ScenarioResult, row: EnrichedRow, technical: dict[str, Any] | None = None) -> str:
    market = row.market
    technical = technical or result.technical or {}
    scenario_code = _scenario_code(result.scenario)
    status = {
        "OBSERVATION_AFTER_BOUNCE": "OBSERVATION",
    }.get(result.status.upper(), result.status.upper())
    title = f"{_display_symbol(result.symbol)} — {scenario_code}"

    lines = [
        f"<b>{_esc(title)}</b>",
        f"<b>{_esc(status)} · {result.passed_count}/{result.total_count}</b>",
        "",
        f"F = <code>{_esc(_decimal(market.futures_imbalance_4h, 4, signed=True))}</code> · "
        f"S = <code>{_esc(_decimal(market.spot_imbalance_4h, 4, signed=True))}</code>",
        "",
        f"- OI 4H: <code>{_esc(_percent(market.oi_chg4h_pct, 2, signed=True))}</code>.",
        f"- Цена 24H: <code>{_esc(_percent(market.chg24_pct, 2, signed=True))}</code>.",
        f"- Funding OI_W: <code>{_esc(_percent(market.fr_oiw_pct, 4, signed=True))}</code>.",
        f"- L/S 1D: <code>{_esc(_decimal(market.ls_ratio_1d, 2))}</code>.",
    ]
    if market.btc_corr7d is not None:
        correlation_note = ""
        if _display_symbol(result.symbol) != "BTC" and market.btc_corr7d >= 0.7:
            correlation_note = " — высокая положительная связь с BTC; не независимая экспозиция"
        elif _display_symbol(result.symbol) != "BTC" and market.btc_corr7d <= -0.7:
            correlation_note = " — сильная обратная связь с BTC; не независимая экспозиция"
        lines.append(
            f"- BTC Corr. 7D: <code>{_esc(_decimal(market.btc_corr7d, 2, signed=True))}</code>{_esc(correlation_note)}."
        )
    else:
        lines.append("- BTC Corr. 7D: n/a.")

    lines.extend(f"- {_esc(item)}" for item in _technical_bullets(result, technical))
    lines.extend(["", f"<b>Вывод:</b> {_esc(_conclusion(result, technical))}"])
    text = "\n".join(lines) + DISCLAIMER
    if len(text) > 3900:
        text = text[: 3900 - len(DISCLAIMER) - 2].rstrip() + "…" + DISCLAIMER
    return text



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
            try:
                response = requests.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                if not payload.get("ok"):
                    raise RuntimeError(payload.get("description", "Telegram API returned ok=false"))
                message_id = (payload.get("result") or {}).get("message_id")
                results.append({"chat_id": chat_id, "sent": True, "message_id": message_id})
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                log.exception("Telegram delivery failed for chat_id=%s", chat_id)
                results.append({"chat_id": chat_id, "sent": False, "error": str(exc)})
        return results
