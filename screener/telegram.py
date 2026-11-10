from __future__ import annotations

import html
import logging
from typing import Any

import requests

from .metrics import EnrichedRow
from .models import ScenarioResult
from .structure import scenario_invalidation

log = logging.getLogger(__name__)


def _esc(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=False)


def format_alert(result: ScenarioResult, row: EnrichedRow, technical: dict[str, Any] | None = None) -> str:
    m = row.market
    technical = technical or result.technical
    if result.status == "WAIT_BOUNCE":
        title = f"⚠️ КАСКАД / ЖДАТЬ ОТСКОК — {result.name} ({result.symbol})"
    elif result.status == "OBSERVATION_AFTER_BOUNCE":
        title = f"🔴 SHORT ПОСЛЕ ОТСКОКА — НАБЛЮДЕНИЕ — {result.name} ({result.symbol})"
    elif result.status == "PREPARATION":
        title = f"🟡 ПОДГОТОВКА LONG — {result.name} ({result.symbol})"
    elif result.direction == "LONG":
        title = f"🟢 КАНДИДАТ LONG — {result.name} ({result.symbol})"
    else:
        title = f"🔴 КАНДИДАТ SHORT — {result.name} ({result.symbol})"

    lines = [
        f"<b>{_esc(title)}</b>",
        f"Сценарий: <b>{_esc(result.scenario)}</b>",
        f"Условия: <b>{result.passed_count}/{result.total_count}</b>",
        f"Цена: <code>{_esc(m.price)}</code> | 24H: <code>{_esc(m.chg24_pct)}%</code>",
        f"F: <code>{_esc(m.futures_imbalance_4h)}</code> | S: <code>{_esc(m.spot_imbalance_4h)}</code>",
        f"OI 4H: <code>{_esc(m.oi_chg4h_pct)}%</code> ({_esc(_money(m.oi_chg4h_usd))})",
        f"OI 24H: <code>{_esc(m.oi_chg24_pct)}%</code> ({_esc(_money(m.oi_chg24_usd))})",
        f"Funding OI_W: <code>{_esc(m.fr_oiw_pct)}%</code> | L/S 1D: <code>{_esc(m.ls_ratio_1d)}</code>",
    ]
    if m.liq_long24_usd is not None or m.liq_short24_usd is not None:
        lines.append(f"Liqs 24H long/short: <code>{_money(m.liq_long24_usd)} / {_money(m.liq_short24_usd)}</code>")
    for check in result.checks:
        mark = "✔" if check.passed is True else ("✖" if check.passed is False else "?")
        lines.append(f"{mark} {_esc(check.name)}: {_esc(check.actual)} (правило: {_esc(check.rule)})")
    for check in technical.get("trigger_checks", []):
        mark = "✔" if check.get("passed") is True else ("✖" if check.get("passed") is False else "?")
        lines.append(f"Триггер {mark} {_esc(check.get('name'))}: {_esc(check.get('actual'))} (правило: {_esc(check.get('rule'))})")
    if row.long_liq_share is not None:
        lines.append(f"Доля ликвидаций лонгов: <code>{row.long_liq_share:.1%}</code>")
    if row.rel_strength_vs_btc_pp is not None:
        lines.append(f"RelStrength к BTC: <code>{row.rel_strength_vs_btc_pp:+.2f} п.п.</code>")
    if m.btc_corr7d is not None:
        if m.btc_corr7d > 0.7:
            lines.append(f"BTC Correlation 7D: <code>{m.btc_corr7d:.3f}</code> — считать BTC-коррелированной экспозицией, не отдельной независимой ставкой.")
        else:
            lines.append(f"BTC Correlation 7D: <code>{m.btc_corr7d:.3f}</code>")
    else:
        lines.append("BTC Correlation 7D: n/a — корреляция не проверена")
    if result.status == "WAITING_FOR_HISTORY_OR_HIGHER_LOW":
        lines.append(f"Spot sequence 4H: <code>{technical.get('spot_sequence_ok')}</code>; 1H higher low: <code>{(technical.get('higher_low') or {}).get('confirmed')}</code>")
    orderbook = technical.get("orderbook") or {}
    if orderbook.get("spread_bps") is not None:
        lines.append(
            f"Стакан Binance: spread <code>{orderbook['spread_bps']:.2f} bps</code>; "
            f"depth ±1% bid/ask <code>{_money(orderbook.get('bid_depth_1pct_usd'))} / {_money(orderbook.get('ask_depth_1pct_usd'))}</code>"
        )
        lines.append(f"Глубина imbalance ±1%: <code>{orderbook.get('depth_imbalance_1pct')}</code>")
        if orderbook.get("notional_usd"):
            buy = orderbook.get("market_buy_estimate", {})
            sell = orderbook.get("market_sell_estimate", {})
            lines.append(
                f"Оценка проскальзывания на ${orderbook['notional_usd']:,.0f}: "
                f"buy <code>{buy.get('slippage_bps') if buy.get('slippage_bps') is not None else 'n/a'} bps</code>; "
                f"sell <code>{sell.get('slippage_bps') if sell.get('slippage_bps') is not None else 'n/a'} bps</code> (оценка по снимку стакана)"
            )
    fee = technical.get("execution_cost_assumptions", {}).get("taker_fee_bps_per_side")
    lines.append(f"Комиссия taker/сторону: <code>{fee} bps</code>" if fee is not None else "Комиссия аккаунта: не задана")
    if result.quality_flags:
        lines.append(f"Качество рынка: <code>{_esc('; '.join(result.quality_flags))}</code>")
    if result.reason:
        lines.extend(["", _esc(result.reason)])
    chart_context = technical.get("context_4h")
    if chart_context:
        lines.append(
            f"4H контекст: <code>{_esc(chart_context.get('trend'))}</code>; "
            f"support=<code>{_esc(chart_context.get('last_confirmed_support'))}</code>; "
            f"resistance=<code>{_esc(chart_context.get('last_confirmed_resistance'))}</code>"
        )
    context15m = technical.get("context_15m") or {}
    if context15m:
        lines.append(f"15m последняя закрытая свеча: <code>{_esc(context15m.get('last_closed_change_pct'))}%</code>")
    invalidation = scenario_invalidation(result.direction, result.scenario, technical)
    if invalidation["level"] is not None:
        lines.append(f"Уровень отмены: <code>{invalidation['level']}</code> — {_esc(invalidation['rule'])}")
    else:
        lines.append("Уровень отмены: не определён по доступным закрытым свечам")
    structure = result.technical.get("1h_retest") or result.technical.get("higher_low") or result.technical.get("bearish_structure_break") or result.technical.get("weak_bounce_rejection")
    if isinstance(structure, dict) and structure.get("reason"):
        lines.append(f"1H: {_esc(structure['reason'])}")
    disclaimer = "\n\n<i>Наблюдение по P1, не торговый сигнал и не команда на вход. Вход требует проверки графика и риска.</i>"
    text = "\n".join(lines) + disclaimer
    if len(text) > 3900:
        # Preserve the safety disclaimer even when the detailed checks are truncated.
        text = text[: 3900 - len(disclaimer) - 2].rstrip() + "…" + disclaimer
    return text


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}" if abs(value) >= 1_000_000 else f"${value:,.2f}"


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
