"""Production 5-minute cron runner for the Binance Futures quant screener."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from binance_ingestion import BinanceRestrictedLocationError
from engine.circuit_breaker import CircuitBreaker
from engine.screener import QuantScreener, save_latest_scan_json
from engine.telegram import TelegramAlerter, TelegramDispatchError
from engine.risk_guard import RiskGuard
from engine.signal_ledger import append_signal_events
from engine.dispatch_ledger import append_dispatch_events

logger = logging.getLogger("cron_runner")


def finalize_research_rows(
    research_rows: list[dict],
    *,
    dispatch_allowed: bool | None,
    block_reason: str | None,
    dispatched_signal_ids: set[str] | None,
    dispatched_at_ms: int,
    not_attempted: bool = False,
) -> None:
    """Materialize final dispatch outcome into immutable research rows before persistence."""
    sent_ids = set(dispatched_signal_ids or ())
    for row in research_rows:
        final_type = str(row.get("final_signal_type", "NONE"))
        is_strong = final_type in ("STRONG_LONG", "STRONG_SHORT")
        signal_id = f"{row.get('symbol')}:{row.get('timestamp_ms')}:{final_type}"
        dispatched = bool(
            is_strong
            and not not_attempted
            and row.get("symbol") is not None
            and signal_id in sent_ids
        )
        row["dispatch_allowed"] = dispatch_allowed
        row["dispatch_block_reason"] = block_reason
        row["dispatched"] = dispatched
        row["dispatched_at_ms"] = int(dispatched_at_ms) if dispatched else None
        row["dispatch_status_source"] = "FINAL_RISK_GUARD_AND_TELEGRAM"


async def main() -> None:
    top_n = int(os.environ.get("TOP_N_SYMBOLS", "100"))
    concurrency = int(os.environ.get("SCAN_CONCURRENCY", "25"))
    paper_trading = os.environ.get("PAPER_TRADING", "0") == "1"

    # Создаем папку data, если её нет
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # NEW: circuit breaker check
    breaker = CircuitBreaker(
        state_file=data_dir / "circuit_breaker.json",
        max_consecutive_failures=int(os.environ.get("CB_MAX_FAILURES", "5")),
        halt_duration_minutes=float(os.environ.get("CB_HALT_MINUTES", "30")),
    )
    risk_guard = RiskGuard(
        state_file=data_dir / "equity_state.json",
        account_equity=float(os.environ.get("ACCOUNT_EQUITY_USDT", "10000.0")),
        max_drawdown_pct=float(os.environ.get("MAX_DRAWDOWN_PCT", "0.20")),
        daily_loss_limit_pct=float(os.environ.get("DAILY_LOSS_LIMIT_PCT", "0.08")),
        paper_trading=paper_trading,
        kill_switch_file=data_dir / "KILL_SWITCH",
        max_state_age_sec=float(os.environ.get("MAX_EQUITY_STATE_AGE_SEC", "600")),
    )
    if breaker.is_halted():
        print(f"⚠️ [CIRCUIT BREAKER] Halted for {breaker.halt_remaining_minutes():.1f} more minutes. Skipping scan.")
        return

    risk_allowed, risk_reason = risk_guard.evaluate()
    if not risk_allowed:
        print(f"🛑 [RISK GUARD] {risk_reason}. New signals are blocked.")
        return

    screener = QuantScreener(
        state_file=data_dir / "market_state.bin",
        concurrency_limit=concurrency,
        top_n_symbols=top_n,
    )
    alerter = TelegramAlerter()

    try:
        signals, synthetic_liqs, summary, research_rows = await screener.scan()

        signals_path = data_dir / ("paper_signals_latest.json" if paper_trading else "signals_latest.json")
        save_latest_scan_json(signals, synthetic_liqs, summary, target_path=signals_path)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 [SCAN COMPLETE] Scanned {summary.total_scanned} symbols in {summary.duration_sec}s")
        print(f"🪙 BTC Regime: {summary.btc_regime} ({summary.btc_change_5m_pct:+.2f}%)")
        print(f"📈 Successful: {summary.successful_symbols} | Rejected: {summary.rejected_symbols} | Signals Ready: {summary.signal_ready_symbols}")
        if summary.signal_ready_symbols == 0 and summary.signal_readiness_reasons:
            top_reasons = ", ".join(f"{k}={v}" for k, v in list(summary.signal_readiness_reasons.items())[:8])
            print(f"🧭 Signal readiness blockers: {top_reasons}")
        if summary.state_recovery_sources:
            recovery_text = ", ".join(f"{k}={v}" for k, v in summary.state_recovery_sources.items())
            print(f"🗃️ State history sources: {recovery_text}")
        if summary.stale_state_symbols_dropped > 0:
            print(f"🧹 Stale state symbols dropped: {summary.stale_state_symbols_dropped}")
        print(f"⚡ Strong Longs: {summary.strong_longs_count} | Strong Shorts: {summary.strong_shorts_count} | Liqs: {summary.synthetic_liqs_count}")
        if summary.portfolio_limited_symbols > 0:
            print(f"🚫 Portfolio-limited (downgraded): {summary.portfolio_limited_symbols}")
        if paper_trading:
            print("📝 MODE: PAPER TRADING (alerts prefixed, no live orders)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        if paper_trading:
            # Prefix all alerts to clearly indicate paper mode
            for sig in signals:
                if sig.signal_type in ("STRONG_LONG", "STRONG_SHORT"):
                    sig.gate_status = f"[PAPER] {sig.gate_status}"

        # Re-check immediately before external dispatch so a kill switch or fresh
        # equity reconciliation received during the scan can still block new risk.
        risk_allowed, risk_reason = risk_guard.evaluate()
        dispatch_ts_ms = int(time.time() * 1000)

        if not risk_allowed:
            finalize_research_rows(
                research_rows,
                dispatch_allowed=False,
                block_reason=risk_reason,
                dispatched_signal_ids=set(),
                dispatched_at_ms=dispatch_ts_ms,
                not_attempted=True,
            )
            append_dispatch_events(
                data_dir / "dispatch_ledger.jsonl",
                signals,
                allowed=False,
                block_reason=risk_reason,
            )
            screener.research_recorder.append_rows(research_rows)
            breaker.record_success()
            print(f"🛑 [RISK GUARD] {risk_reason} after scan. Suppressing alert dispatch.")
            return

        # Risk is final; record the actionable signal before Telegram so the
        # signal ledger remains the durable decision/audit event, while the
        # dispatch ledger records the external delivery result.
        ledger_count = append_signal_events(data_dir / "signal_ledger.jsonl", signals)
        if ledger_count:
            print(f"[LEDGER] Recorded {ledger_count} risk-approved STRONG signal event(s).")

        try:
            sent_alerts, sent_signal_ids = await alerter.process_and_dispatch_signals(signals, synthetic_liqs)
        except TelegramDispatchError as exc:
            finalize_research_rows(
                research_rows,
                dispatch_allowed=True,
                block_reason=str(exc),
                dispatched_signal_ids=exc.sent_signal_ids,
                dispatched_at_ms=dispatch_ts_ms,
            )
            append_dispatch_events(
                data_dir / "dispatch_ledger.jsonl",
                signals,
                allowed=True,
                block_reason=str(exc),
                dispatched_at_ms=dispatch_ts_ms,
                dispatched_signal_ids=exc.sent_signal_ids,
            )
            screener.research_recorder.append_rows(research_rows)
            raise
        except Exception as exc:
            finalize_research_rows(
                research_rows,
                dispatch_allowed=True,
                block_reason=f"DISPATCH_EXCEPTION:{type(exc).__name__}",
                dispatched_signal_ids=set(),
                dispatched_at_ms=dispatch_ts_ms,
            )
            append_dispatch_events(
                data_dir / "dispatch_ledger.jsonl",
                signals,
                allowed=True,
                block_reason=f"DISPATCH_EXCEPTION:{type(exc).__name__}",
                dispatched_at_ms=dispatch_ts_ms,
            )
            screener.research_recorder.append_rows(research_rows)
            raise

        finalize_research_rows(
            research_rows,
            dispatch_allowed=True,
            block_reason=None,
            dispatched_signal_ids=sent_signal_ids,
            dispatched_at_ms=dispatch_ts_ms,
        )
        append_dispatch_events(
            data_dir / "dispatch_ledger.jsonl",
            signals,
            allowed=True,
            dispatched_at_ms=dispatch_ts_ms,
            dispatched_signal_ids=sent_signal_ids,
        )
        screener.research_recorder.append_rows(research_rows)
        breaker.record_success()
        if sent_alerts > 0:
            print(f"[TELEGRAM] Successfully dispatched {sent_alerts} alert(s).")
    except BinanceRestrictedLocationError as exc:
        logger.critical(
            "Binance Futures is unavailable from this runner egress: %s. "
            "Move the job to a Binance-eligible network/runner; do not bypass the restriction.",
            exc,
        )
        breaker.record_failure()
        raise SystemExit(2) from exc
    except Exception:
        breaker.record_failure()
        raise
    finally:
        await screener.close()


if __name__ == "__main__":
    asyncio.run(main())
