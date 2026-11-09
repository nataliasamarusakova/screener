from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import EnrichedRow
from .models import MarketStats, ScenarioResult
from .structure import scenario_invalidation


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}"


def build_markdown(
    fetched_at: datetime,
    contract_type: str,
    row_count: int,
    missing_columns: list[str],
    stats: MarketStats,
    results: list[ScenarioResult],
    rows_by_symbol: dict[str, EnrichedRow],
    notifications: list[dict[str, Any]] | None = None,
    deltas_by_symbol: dict[str, dict[str, Any]] | None = None,
    missing_required_columns: list[str] | None = None,
    headers_found: list[str] | None = None,
) -> str:
    def share(n: int, denominator: int) -> str:
        return f"{n}/{denominator} ({n / denominator:.1%})" if denominator else "n/a"

    lines = [
        "# Coinalyze Screener — отчёт",
        "",
        f"- Снимок получен: **{fetched_at.isoformat()}** (UTC время сбора, не серверный timestamp Coinalyze).",
        f"- Контракты: **{contract_type}**.",
        f"- Строк: **{row_count}**.",
        f"- Версия правил: **P1** (не проходила backtest).",
        "",
        "## Рынок целиком",
        "",
        "| Метрика | Значение |",
        "|---|---:|",
        f"| Медиана F | {stats.median_f if stats.median_f is not None else 'n/a'} |",
        f"| Медиана S | {stats.median_s if stats.median_s is not None else 'n/a'} |",
        f"| Медиана OI Chg 24H | {format_pct(stats.median_oi24_pct)} |",
        f"| Медиана OI Chg 4H | {format_pct(stats.median_oi4h_pct)} |",
        f"| F < 0 | {share(stats.f_negative_count, stats.f_count)} |",
        f"| S < 0 | {share(stats.s_negative_count, stats.s_count)} |",
        f"| OI 24H < 0 | {share(stats.oi24_negative_count, stats.oi24_count)} |",
        f"| Цена 24H > 0 | {share(stats.price_positive_count, stats.price_count)} |",
        f"| Цена 24H < 0 | {share(stats.price_negative_count, stats.price_count)} |",
        "",
    ]
    btc = next((x for x in rows_by_symbol.values() if x.market.symbol.upper().split(".", 1)[0].replace("_PERP", "") in {"BTC", "BTCUSDT"}), None)
    eth = next((x for x in rows_by_symbol.values() if x.market.symbol.upper().split(".", 1)[0].replace("_PERP", "") in {"ETH", "ETHUSDT"}), None)
    if btc or eth:
        lines.extend(["### BTC и ETH", "", "| Монета | Цена 24H | OI 24H | OI 24H USD | OI 4H | OI 4H USD | F | S | Funding OI_W | L/S 1H | L/S 1D | RelStrength |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for label, item in (("BTC", btc), ("ETH", eth)):
            if item:
                m = item.market
                lines.append(f"| {label} | {format_pct(m.chg24_pct)} | {format_pct(m.oi_chg24_pct)} | {_money(m.oi_chg24_usd)} | {format_pct(m.oi_chg4h_pct)} | {_money(m.oi_chg4h_usd)} | {m.futures_imbalance_4h} | {m.spot_imbalance_4h} | {m.fr_oiw_pct} | {m.ls_ratio_1h} | {m.ls_ratio_1d} | {item.rel_strength_vs_btc_pp} п.п. |")
        lines.append("")
    price_down_share = stats.price_negative_count / stats.price_count if stats.price_count else 0
    oi_down_share = stats.oi24_negative_count / stats.oi24_count if stats.oi24_count else 0
    if stats.median_oi24_pct is not None and stats.median_oi24_pct < 0 and oi_down_share > 0.5 and stats.median_f is not None and stats.median_s is not None:
        mode = "Сокращение OI за сутки при одновременных медианных потоках F/S " + ("в плюсе" if stats.median_f > 0 and stats.median_s > 0 else "не в обоих положительных значениях")
        lines.extend(["### Предварительный режим", "", mode + f"; отрицательный OI 24H у {oi_down_share:.1%} выборки, цена 24H отрицательна примерно у {price_down_share:.1%}. Это не односторонний directional signal.", ""])

    if headers_found:
        lines.extend(["**Колонки, найденные в таблице:** " + ", ".join(headers_found), ""])
    if missing_columns:
        lines.extend(["**Отсутствующие распознанные колонки:** " + ", ".join(missing_columns), ""])
    if missing_required_columns:
        lines.extend(["**Отсутствующие обязательные колонки P1:** " + ", ".join(missing_required_columns), ""])
    if stats.warnings:
        lines.extend(["**Предупреждения:** " + "; ".join(stats.warnings), ""])

    lines.extend(["## Ранги потоков по всей таблице", ""])
    for label, attr, adj_attr, rank_attr, condition in (
        ("Futures — верхние 15% положительного потока", "futures_imbalance_4h", "f_adj", "f_rank_desc", "f_strong_buy"),
        ("Futures — нижние 15% отрицательного потока", "futures_imbalance_4h", "f_adj", "f_rank_desc", "f_strong_sell"),
        ("Spot — верхние 15% положительного потока", "spot_imbalance_4h", "s_adj", "s_rank_desc", "s_strong_buy"),
        ("Spot — нижние 15% отрицательного потока", "spot_imbalance_4h", "s_adj", "s_rank_desc", "s_strong_sell"),
    ):
        selected = [item for item in rows_by_symbol.values() if getattr(item, condition)]
        selected.sort(key=lambda item: getattr(item, rank_attr) or 0, reverse=condition.endswith("sell"))
        lines.extend([f"### {label}", "", "| Ранг | Монета | Значение | Adjusted |", "|---:|---|---:|---:|"])
        if not selected:
            lines.append("| — | нет значений | — | — |")
        for item in selected:
            lines.append(f"| {getattr(item, rank_attr)} | {item.symbol} | {getattr(item.market, attr):+.4f} | {getattr(item, adj_attr):+.4f} |")
        lines.append("")

    # P1 asks for derived values for every row, not just scenario candidates.
    # Keep two narrower tables so the complete universe remains readable.
    def nfmt(value: float | None, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:+.{digits}f}"

    lines.extend([
        "## Полные расчёты по каждой монете",
        "",
        "Все строки универсума, включая монеты без сценариев. Формулы: Buy share=(100+Imbalance)/2; Spread=S−F; ΔFunding=PFR AVG OI_W−FR AVG OI_W; FR dispersion=FR AVG−FR AVG OI_W; Long Liq share=Long Liqs/(Long+Short Liqs); LiqImb=(Short−Long)/(Short+Long); RelStrength=Chg24 coin−Chg24 BTC.",
        "",
        "### Потоки, funding, ликвидации и относительная сила",
        "",
        "| Монета | F | S | Buy F | Buy S | Spread S−F | ΔFunding, п.п. | FR−FR OI_W, п.п. | Long Liq share | LiqImb | RelStrength, п.п. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in rows_by_symbol.values():
        lines.append(
            f"| {item.symbol} | {nfmt(item.market.futures_imbalance_4h)} | {nfmt(item.market.spot_imbalance_4h)} "
            f"| {nfmt(item.futures_buy_share_pct, 2)}% | {nfmt(item.spot_buy_share_pct, 2)}% | {nfmt(item.spread)} "
            f"| {nfmt(item.delta_funding_oiw_pp)} | {nfmt(item.fr_venue_dispersion_pp)} "
            f"| {nfmt(item.long_liq_share * 100 if item.long_liq_share is not None else None, 2)}% "
            f"| {nfmt(item.liq_imbalance)} | {nfmt(item.rel_strength_vs_btc_pp, 2)} |"
        )
    lines.extend([
        "",
        "### OI, ликвидации относительно падения OI, ранги и качество",
        "",
        "| Монета | OI 4H % / USD | OI 24H % / USD | Liqs / падение OI | F_adj / ранг | S_adj / ранг | OI/MCap | OI/Vol | Vol/MCap | Флаги качества | Доп. замечания |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for item in rows_by_symbol.values():
        m = item.market
        oi4h = f"{nfmt(m.oi_chg4h_pct, 2)}% / {_money(m.oi_chg4h_usd)}"
        oi24 = f"{nfmt(m.oi_chg24_pct, 2)}% / {_money(m.oi_chg24_usd)}"
        coverage = nfmt(item.liquidation_coverage_of_oi_drop_pct, 2) + "%"
        f_rank = f"{nfmt(item.f_adj)} / {item.f_rank_desc if item.f_rank_desc is not None else 'n/a'}"
        s_rank = f"{nfmt(item.s_adj)} / {item.s_rank_desc if item.s_rank_desc is not None else 'n/a'}"
        flags = "; ".join(item.quality_flags) or "нет"
        notes = "; ".join(item.diagnostic_notes) or "нет"
        lines.append(
            f"| {item.symbol} | {oi4h} | {oi24} | {coverage} | {f_rank} | {s_rank} "
            f"| {nfmt(m.oi_mktcap_ratio, 3)} | {nfmt(m.oi_vol_ratio, 3)} | {nfmt(m.vol_mktcap_ratio, 3)} | {flags} | {notes} |"
        )
    lines.append("")

    lines.extend(["## Дивергенции", ""])
    divergence_lines: list[str] = []
    for item in rows_by_symbol.values():
        m = item.market
        checks: list[str] = []
        if m.chg24_pct is not None and m.spot_imbalance_4h is not None and m.chg24_pct * m.spot_imbalance_4h < 0:
            checks.append("цена 24H и Spot Imbalance расходятся")
        if m.futures_imbalance_4h is not None and m.spot_imbalance_4h is not None and m.futures_imbalance_4h * m.spot_imbalance_4h < 0:
            checks.append("F и S имеют разные знаки")
        if m.chg24_pct is not None and m.oi_chg24_pct is not None:
            if m.chg24_pct > 0 and m.oi_chg24_pct < 0:
                checks.append("цена растёт при падающем OI 24H — возможно закрытие шортов")
            elif m.chg24_pct < 0 and m.oi_chg24_pct < 0:
                checks.append("цена падает при падающем OI 24H — возможно закрытие лонгов")
            elif m.chg24_pct * m.oi_chg24_pct < 0:
                checks.append("знак цены и OI 24H расходится")
        if checks:
            divergence_lines.append(f"- **{m.symbol}:** " + "; ".join(dict.fromkeys(checks)))
    lines.extend(divergence_lines if divergence_lines else ["Нет классифицированных дивергенций по доступным данным."])
    lines.append("")

    flagged = [item for item in rows_by_symbol.values() if item.quality_flags]
    lines.extend(["## Аномалии и флаги качества", ""])
    if flagged:
        for item in flagged:
            lines.append(f"- **{item.market.symbol}:** " + "; ".join(item.quality_flags))
    else:
        lines.append("Флаги качества не обнаружены среди строк с доступными необходимыми данными.")
    funding_diffs = sorted(
        (item for item in rows_by_symbol.values() if item.delta_funding_oiw_pp is not None),
        key=lambda item: abs(item.delta_funding_oiw_pp or 0.0), reverse=True,
    )
    if funding_diffs:
        lines.append("")
        lines.append("**Крупнейшие расхождения прогнозного и текущего funding OI_W (без порога P1):**")
        for item in funding_diffs:
            lines.append(f"- {item.market.symbol}: FR OI_W={item.market.fr_oiw_pct}%; PFR OI_W={item.market.pfr_oiw_pct}%; Δ={item.delta_funding_oiw_pp:+.4f} п.п.")
    corr_available = [item for item in rows_by_symbol.values() if item.market.btc_corr7d is not None]
    if corr_available:
        corr_high = [item.market.symbol for item in corr_available if item.market.btc_corr7d > 0.7]
        lines.append("")
        lines.append("BTC Correlation 7D доступна: " + (", ".join(corr_high) + " имеют >0.7; считать BTC-коррелированной экспозицией." if corr_high else "значений >0.7 нет."))
    else:
        lines.append("BTC Correlation 7D отсутствует: корреляция не проверена.")
    lines.append("")

    if deltas_by_symbol:
        lines.extend(["## Изменения между снимками", ""])
        for symbol, delta in deltas_by_symbol.items():
            lines.append(
                f"- **{symbol}:** интервал {delta.get('interval_hours')} ч ({delta.get('window_note')}); "
                f"ΔPrice={delta.get('price_change_pct')}%; ΔOI={delta.get('oi_change_usd')} USD / {delta.get('oi_change_pct')}%; "
                f"ΔF={delta.get('futures_imbalance_delta')}; ΔS={delta.get('spot_imbalance_delta')}; "
                f"ΔFunding={delta.get('funding_oiw_delta_pp')} п.п.; "
                f"цена×OI={delta.get('price_x_oi_classification')}"
            )
        lines.append("")

    lines.extend(["## Сценарии P1 (только N−1 и лучше)", ""])
    if not results:
        lines.append("**Нет сигнала.** Ни один сценарий не выполнил N−1 условий.")
    else:
        lines.extend([
            "| Монета | Сценарий | Условия | Статус | Красные флаги |",
            "|---|---|---:|---|---|",
        ])
        for item in results:
            flags = "; ".join(item.quality_flags) or "нет"
            lines.append(f"| {item.symbol} | {item.scenario} | {item.passed_count}/{item.total_count} | {item.status} | {flags} |")
        lines.append("")
        for item in results:
            enriched = rows_by_symbol[item.symbol]
            m = enriched.market
            lines.extend([
                f"### {item.symbol} — {item.scenario}",
                "",
                f"**Итог:** {item.status}. {item.reason}",
                "",
                "| Условие | Проверка | Факт | Правило |",
                "|---|:---:|---|---|",
            ])
            for check in item.checks:
                mark = "✔" if check.passed is True else ("✖" if check.passed is False else "UNKNOWN")
                lines.append(f"| {check.name} | {mark} | {check.actual} | {check.rule} |")
            if item.scenario.startswith("LONG C"):
                trigger_checks = item.technical.get("trigger_checks", [])
                trigger_passed = sum(check.get("passed") is True for check in trigger_checks)
                lines.extend([
                    "",
                    f"**LONG C — trigger stage: {trigger_passed}/{len(trigger_checks)}; подготовка: {item.passed_count}/{item.total_count}.**",
                    "",
                    "| Условие триггера | Проверка | Факт | Правило |",
                    "|---|:---:|---|---|",
                ])
                for check in trigger_checks:
                    passed = check.get("passed")
                    mark = "✔" if passed is True else ("✖" if passed is False else "UNKNOWN")
                    lines.append(f"| {check.get('name')} | {mark} | {check.get('actual')} | {check.get('rule')} |")
            lines.extend([
                "",
                "**Расчёты:**",
                f"- Futures Buy = (100 + F) / 2 = {enriched.futures_buy_share_pct:.2f}%" if enriched.futures_buy_share_pct is not None else "- Futures Buy = n/a",
                f"- Spot Buy = (100 + S) / 2 = {enriched.spot_buy_share_pct:.2f}%" if enriched.spot_buy_share_pct is not None else "- Spot Buy = n/a",
                f"- Spread = S − F = {enriched.spread:+.4f}" if enriched.spread is not None else "- Spread = n/a",
                f"- ΔFunding = PFR AVG OI_W − FR AVG OI_W = {enriched.delta_funding_oiw_pp:+.4f} п.п." if enriched.delta_funding_oiw_pp is not None else "- ΔFunding = n/a",
                f"- FR AVG − FR AVG OI_W = {enriched.fr_venue_dispersion_pp:+.4f} п.п." if enriched.fr_venue_dispersion_pp is not None else "- FR AVG − FR AVG OI_W = n/a",
                f"- L/S Ratio: 1H={m.ls_ratio_1h}; 1D={m.ls_ratio_1d}",
                f"- Доля ликвидаций лонгов = {enriched.long_liq_share:.2%}" if enriched.long_liq_share is not None else "- Доля ликвидаций лонгов = n/a",
                f"- LiqImb = {enriched.liq_imbalance:+.4f}" if enriched.liq_imbalance is not None else "- LiqImb = n/a",
                f"- RelStrength к BTC = {enriched.rel_strength_vs_btc_pp:+.2f} п.п." if enriched.rel_strength_vs_btc_pp is not None else "- RelStrength к BTC = n/a",
                f"- Ликвидации / падение OI 24H = {enriched.liquidation_coverage_of_oi_drop_pct:.2f}%" if enriched.liquidation_coverage_of_oi_drop_pct is not None else "- Ликвидации / падение OI 24H = n/a",
                f"- MktCap={_money(m.market_cap_usd)}; FDV={_money(m.fdv_usd)}; VOL24H={_money(m.volume24_usd)}; OI={_money(m.oi_usd)}",
                f"- OI/VOL24H={m.oi_vol_ratio}; OI/MktCap={m.oi_mktcap_ratio}; VOL24H/MktCap={m.vol_mktcap_ratio}",
                f"- Liqs/VOL24H={m.liq_vol_ratio}; Liqs/OI={m.liq_oi_ratio}; Long Liqs={_money(m.liq_long24_usd)}; Short Liqs={_money(m.liq_short24_usd)}",
                f"- OI4H = {format_pct(m.oi_chg4h_pct)} ({_money(m.oi_chg4h_usd)}); OI24H = {format_pct(m.oi_chg24_pct)} ({_money(m.oi_chg24_usd)})",
                f"- F_adj = F − median(F) = {enriched.f_adj:+.4f}; ранг F: {enriched.f_rank_desc}" if enriched.f_adj is not None else "- F_adj / ранг F = n/a",
                f"- S_adj = S − median(S) = {enriched.s_adj:+.4f}; ранг S: {enriched.s_rank_desc}" if enriched.s_adj is not None else "- S_adj / ранг S = n/a",
                "",
                "**Стакан и исполнение (Binance, диагностические данные; не фильтр P1):**",
            ])
            lines.append("**Свечная структура и отмена сценария (закрытые свечи Binance):**")
            context4h = item.technical.get("context_4h") or {}
            if context4h:
                lines.append(f"- 4H контекст: {context4h.get('trend')}; EMA20={context4h.get('ema20')}; EMA50={context4h.get('ema50')}; last close={context4h.get('last_close')}; подтверждённая поддержка={context4h.get('last_confirmed_support')}; сопротивление={context4h.get('last_confirmed_resistance')}")
            context15m = item.technical.get("context_15m") or {}
            if context15m:
                lines.append(f"- 15m последняя закрытая свеча: open time={context15m.get('last_closed_open_time_ms')}; изменение тела/открытие-закрытие={context15m.get('last_closed_change_pct')}% (контекст, не самостоятельный триггер)")
            hl = item.technical.get("higher_low") or {}
            if hl:
                lines.append(f"- 1H higher low: confirmed={hl.get('confirmed')}; previous={hl.get('previous_low')}; latest={hl.get('latest_low')}; {hl.get('reason', '')}")
            retest = item.technical.get("1h_retest") or item.technical.get("trend_retest_long") or item.technical.get("trend_retest_short") or {}
            if retest:
                lines.append(f"- 1H breakout/retest: confirmed={retest.get('confirmed')}; level={retest.get('level')}; ATR14={retest.get('atr14')}; {retest.get('reason', '')}")
            bearish = item.technical.get("bearish_structure_break") or {}
            if bearish:
                lines.append(f"- 1H bearish structure break: confirmed={bearish.get('confirmed')}; level={bearish.get('level')}; last close={bearish.get('last_close')}; {bearish.get('reason', '')}")
            rejection = item.technical.get("weak_bounce_rejection") or {}
            if rejection:
                resistance = rejection.get("resistance") or rejection.get("candidate_resistance")
                lines.append(f"- 1H слабый отскок/rejection: confirmed={rejection.get('confirmed')}; resistance={resistance}; {rejection.get('reason', '')}")
            invalidation = scenario_invalidation(item.direction, item.scenario, item.technical)
            if invalidation["level"] is not None:
                lines.append(f"- Уровень отмены сценария: {invalidation['level']} — {invalidation['rule']} ({invalidation['source']})")
            else:
                lines.append("- Уровень отмены: не определён по доступным закрытым свечам; уровень не придумывается.")
            lines.append("")
            orderbook = item.technical.get("orderbook") or {}
            if orderbook.get("spread_bps") is not None:
                lines.extend([
                    f"- Bid/Ask: {orderbook.get('best_bid')} / {orderbook.get('best_ask')}; spread={orderbook['spread_bps']:.2f} bps",
                    f"- Глубина ±1%: bid={_money(orderbook.get('bid_depth_1pct_usd'))}; ask={_money(orderbook.get('ask_depth_1pct_usd'))}; imbalance={orderbook.get('depth_imbalance_1pct')}",
                ])
                if orderbook.get("notional_usd"):
                    buy = orderbook.get("market_buy_estimate") or {}
                    sell = orderbook.get("market_sell_estimate") or {}
                    lines.append(f"- Гипотетический объём {_money(orderbook['notional_usd'])}: оценка проскальзывания сверх best ask/bid — buy {buy.get('slippage_bps')} bps; sell {sell.get('slippage_bps')} bps; незаполненный остаток buy/sell={_money(buy.get('unfilled_usd'))}/{_money(sell.get('unfilled_usd'))}")
            else:
                lines.append("- Стакан не получен или недоступен: " + str(orderbook.get("error", "не запрошен")))
            fee = item.technical.get("execution_cost_assumptions", {}).get("taker_fee_bps_per_side")
            lines.append(f"- Комиссия taker за сторону: {fee} bps" if fee is not None else "- Комиссия аккаунта не задана; комиссии и реальные издержки не подтверждены.")
            lines.extend([
                "",
                "**Диагностика:** " + ("; ".join(enriched.diagnostic_notes) if enriched.diagnostic_notes else "нет дополнительных замечаний"),
                "",
            ])

    lines.extend([
        "## Правила безопасности и ограничения",
        "",
        "- Неполная пагинация Coinalyze блокирует анализ, чтобы не ранжировать частичную выборку.",
        "- `n/a` остаётся неизвестным значением и не заменяется нулём.",
        "- Futures/Spot Imbalance — пользовательские нормированные метрики, не денежный поток и не доказательство агрессии крупных участников; на тонком споте Spot Imbalance может быть шумным. Спотового объёма в таблице нет, делить Imbalance на Volume нельзя.",
        "- L/S Ratio — отношение числа аккаунтов, не капитала и не размера позиций. У каждого фьючерсного контракта есть обе стороны; рост OI не означает приток покупателей или продавцов.",
        "- Funding показывается за интервал начисления: не годализируется; прогнозный funding не считается уплаченным.",
        "- Liqs. 24H — агрегат за сутки; точное время событий и покрытие бирж неизвестны.",
        "- Доли OI/MktCap, OI/VOL24H, VOL24H/MktCap, Liqs/VOL24H и Liqs/OI выводятся как доли, не проценты (например, 0.0016 = 0.16%).",
        "- Свечи берутся только закрытые; формирующаяся последняя свеча отбрасывается.",
        "- Оповещения — кандидаты/наблюдение, не команда на открытие позиции.",
        "- Нет численного порога «существенных» short liquidations в LONG C; trigger остаётся UNKNOWN, пока порог не задан в настройках P1.",
        "- «Резкий» разрыв текущего и прогнозного funding не формализован в P1 и не превращается в автоматический красный флаг.",
        "- Корреляция 7D проверяется только при наличии колонки; одинаковые BTC-коррелированные альты не диверсифицируют риск. Без корреляционной колонки корреляция не проверена.",
        "- ОИ/Market Cap показывается в отчёте, но P1 не задаёт численный порог аномалии; автоматический красный флаг по нему не придумывается.",
        "- Не получены free float, дата листинга, календарь макро-событий и unlocks; эти риски перед входом нужно проверить отдельно.",
        "- В таблице нет Chg Price 4H. 4H price×OI классифицируется только по ΔPrice/ΔOI между снимками при интервале ровно 4h; при меньшем интервале окна перекрываются, при большем — не сопоставимы. Не смешивать Chg Price 24H с OI 4H.",
        "- Стакан и spread относятся к Binance USDⓈ-M Futures; это не заменяет глубину/комиссии биржи, на которой планируется сделка.",
        "- Нет подключения к балансу и открытым позициям: ограничения риска 0,25–0,5%, суммарного риска 1–1,5% и максимум двух позиций не контролируются программой.",
        "- Параметры P1 не проходили backtest; преимущество не доказано.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    fetched_at: datetime,
    contract_type: str,
    rows: list[EnrichedRow],
    stats: MarketStats,
    results: list[ScenarioResult],
    missing_columns: list[str],
    notifications: list[dict[str, Any]] | None = None,
    deltas_by_symbol: dict[str, dict[str, Any]] | None = None,
    missing_required_columns: list[str] | None = None,
    headers_found: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_symbol = {row.symbol: row for row in rows}
    report_results = [r for r in results if r.near_candidate or r.status in {"PREPARATION", "WAIT_BOUNCE", "OBSERVATION_AFTER_BOUNCE"}]
    md_path = output_dir / "screener_report.md"
    csv_path = output_dir / "candidates.csv"
    json_path = output_dir / "analysis.json"
    md_path.write_text(
        build_markdown(
            fetched_at, contract_type, len(rows), missing_columns, stats,
            report_results, rows_by_symbol, notifications, deltas_by_symbol,
            missing_required_columns,
            headers_found,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "snapshot_at_utc", "symbol", "name", "direction", "scenario", "status", "checks",
            "F", "S", "OI4H_pct", "OI4H_usd", "OI24H_pct", "OI24H_usd", "funding_OI_W_pct", "LS_1D", "quality_flags",
        ])
        writer.writeheader()
        if not report_results:
            writer.writerow({"snapshot_at_utc": fetched_at.isoformat(), "symbol": "нет сигнала", "scenario": "нет сигнала", "status": "NO_SIGNAL"})
        for item in report_results[:10]:
            row = rows_by_symbol[item.symbol]
            m = row.market
            writer.writerow({
                "snapshot_at_utc": fetched_at.isoformat(),
                "symbol": item.symbol,
                "name": item.name,
                "direction": item.direction,
                "scenario": item.scenario,
                "status": item.status,
                "checks": f"{item.passed_count}/{item.total_count}",
                "F": m.futures_imbalance_4h,
                "S": m.spot_imbalance_4h,
                "OI4H_pct": m.oi_chg4h_pct,
                "OI4H_usd": m.oi_chg4h_usd,
                "OI24H_pct": m.oi_chg24_pct,
                "OI24H_usd": m.oi_chg24_usd,
                "funding_OI_W_pct": m.fr_oiw_pct,
                "LS_1D": m.ls_ratio_1d,
                "quality_flags": "; ".join(item.quality_flags),
            })

    payload = {
        "fetched_at_utc": fetched_at.isoformat(),
        "contract_type": contract_type,
        "row_count": len(rows),
        "missing_columns": missing_columns,
        "missing_required_columns": missing_required_columns or [],
        "headers_found": headers_found or [],
        "stats": stats.to_dict(),
        "universe_calculations": [
            {
                "market": {k: v for k, v in item.market.to_dict().items() if k != "raw"},
                "futures_buy_share_pct": item.futures_buy_share_pct,
                "spot_buy_share_pct": item.spot_buy_share_pct,
                "spread_s_minus_f": item.spread,
                "delta_funding_oiw_pp": item.delta_funding_oiw_pp,
                "fr_avg_minus_fr_oiw_pp": item.fr_venue_dispersion_pp,
                "long_liquidation_share": item.long_liq_share,
                "liq_imbalance": item.liq_imbalance,
                "rel_strength_vs_btc_pp": item.rel_strength_vs_btc_pp,
                "liquidation_coverage_of_oi_drop_pct": item.liquidation_coverage_of_oi_drop_pct,
                "oi_drop_mostly_voluntary": item.oi_drop_mostly_voluntary,
                "f_adj": item.f_adj,
                "s_adj": item.s_adj,
                "f_rank_desc": item.f_rank_desc,
                "s_rank_desc": item.s_rank_desc,
                "quality_flags": item.quality_flags,
                "diagnostic_notes": item.diagnostic_notes,
            }
            for item in rows
        ],
        "results": [r.to_dict() for r in results],
        "csv_rows_limited_to": 10,
        "notifications": notifications or [],
        "snapshot_deltas": deltas_by_symbol or {},
        "p1_backtested": False,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, csv_path, json_path
