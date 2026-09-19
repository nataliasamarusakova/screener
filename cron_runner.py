"""
Production Pure-Python 5-Minute Cron Runner.
Zero-Docker architecture with bounded allocation and explicit data-validity checks.

Execution via crontab:
    */5 * * * * cd /workspaces/screener && /usr/bin/python3 cron_runner.py >> cron.log 2>&1

Full Closed-Loop Pipeline:
1. Orchestrates asynchronous scan across 100+ Binance Futures contracts (< 15s).
2. Executes Numba JIT Microstructure & Anti-Flaw algorithms (VPIN, OBI, CVD Divergence).
3. Reconstructs throttled Synthetic Liquidations.
4. Generates Composite Score (-100 to +100) and risk invalidation parameters.
5. Emits visual Rich Terminal Cockpit.
6. Dispatches Telegram alerts for |Score| >= 75 and Liquidations.
7. Exports JSON state for external frontends/APIs.
8. Exits cleanly with zero residual RAM/CPU overhead.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

from engine.dashboard import export_json, render_dashboard
from engine.screener import QuantScreener
from engine.telegram import TelegramAlerter


async def main() -> None:
    top_n = int(os.environ.get("TOP_N_SYMBOLS", "100"))
    concurrency = int(os.environ.get("SCAN_CONCURRENCY", "25"))

    screener = QuantScreener(concurrency_limit=concurrency, top_n_symbols=top_n)
    alerter = TelegramAlerter()

    # 1. Execute Screener Pipeline
    signals, synthetic_liqs, summary = await screener.scan()

    # 2. Render Rich Terminal Dashboard
    render_dashboard(signals, synthetic_liqs, summary)

    # 3. Export JSON for external consumers
    export_json(signals, synthetic_liqs, summary)

    # 4. Dispatch Telegram Alerts for Strong Signals (|Score| >= 75) and Liquidations
    sent_alerts = await alerter.process_and_dispatch_signals(signals, synthetic_liqs)
    if sent_alerts > 0:
        print(f"📬 [TELEGRAM] Successfully dispatched {sent_alerts} alert(s).")


if __name__ == "__main__":
    asyncio.run(main())
