"""Production 5-minute cron runner for the Binance Futures quant screener."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from binance_ingestion import BinanceRestrictedLocationError
from engine.screener import QuantScreener, save_latest_scan_json
from engine.telegram import TelegramAlerter

logger = logging.getLogger("cron_runner")


async def main() -> None:
    top_n = int(os.environ.get("TOP_N_SYMBOLS", "100"))
    concurrency = int(os.environ.get("SCAN_CONCURRENCY", "25"))
    
    # Создаем папку data, если её нет
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    screener = QuantScreener(
        state_file=data_dir / "market_state.bin",
        concurrency_limit=concurrency,
        top_n_symbols=top_n,
    )
    alerter = TelegramAlerter()
    try:
        signals, synthetic_liqs, summary = await screener.scan()
        save_latest_scan_json(
            signals,
            synthetic_liqs,
            summary,
            target_path=data_dir / "signals_latest.json",
        )

        # Информативный вывод статистики в логи GitHub Actions
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 [SCAN COMPLETE] Scanned {summary.total_scanned} symbols in {summary.duration_sec}s")
        print(f"🪙 BTC Regime: {summary.btc_regime} ({summary.btc_change_5m_pct:+.2f}%)")
        print(f"📈 Successful: {summary.successful_symbols} | Rejected: {summary.rejected_symbols} | Signals Ready: {summary.signal_ready_symbols}")
        print(f"⚡ Strong Longs: {summary.strong_longs_count} | Strong Shorts: {summary.strong_shorts_count} | Liqs: {summary.synthetic_liqs_count}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        sent_alerts = await alerter.process_and_dispatch_signals(signals, synthetic_liqs)
        if sent_alerts > 0:
            print(f"[TELEGRAM] Successfully dispatched {sent_alerts} alert(s).")
    except BinanceRestrictedLocationError as exc:
        logger.critical(
            "Binance Futures is unavailable from this runner egress: %s. "
            "Move the job to a Binance-eligible network/runner; do not bypass the restriction.",
            exc,
        )
        raise SystemExit(2) from exc
    finally:
        await screener.close()


if __name__ == "__main__":
    asyncio.run(main())
