"""Production 5-minute cron runner for the Binance Futures quant screener."""
from __future__ import annotations

import asyncio
import logging
import os

from binance_ingestion import BinanceRestrictedLocationError
from engine.screener import QuantScreener, save_latest_scan_json
from engine.telegram import TelegramAlerter

logger = logging.getLogger("cron_runner")


async def main() -> None:
    top_n = int(os.environ.get("TOP_N_SYMBOLS", "100"))
    concurrency = int(os.environ.get("SCAN_CONCURRENCY", "25"))

    screener = QuantScreener(concurrency_limit=concurrency, top_n_symbols=top_n)
    alerter = TelegramAlerter()
    try:
        signals, synthetic_liqs, summary = await screener.scan()
        save_latest_scan_json(signals, synthetic_liqs, summary)

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
