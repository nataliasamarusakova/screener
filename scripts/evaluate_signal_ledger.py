#!/usr/bin/env python3
"""Evaluate matured STRONG signals at fixed forward horizons.

This is an event-study tool, not a fill simulator. It uses the signal reference
price stored in signal_ledger.jsonl and closes of completed 5m candles. Results
must be interpreted with latency/fill uncertainty and realistic costs.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from binance_ingestion import BinanceFuturesIngestion

INTERVAL_MS = 5 * 60 * 1000
HORIZONS = (5, 15, 30, 60)


def signal_candle_open_ms(timestamp_ms: int, stored_candle_open_ms: int | None = None) -> int:
    """Resolve the signal candle open from the stored completed-candle timestamp."""
    if stored_candle_open_ms is not None:
        candle_open = int(stored_candle_open_ms)
    else:
        candle_open = (int(timestamp_ms) // INTERVAL_MS) * INTERVAL_MS
    if candle_open <= 0 or candle_open % INTERVAL_MS != 0:
        raise ValueError("invalid signal candle timestamp")
    return candle_open


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        signal_id = str(row.get("signal_id", ""))
        if not signal_id or signal_id in seen:
            continue
        seen.add(signal_id)
        rows.append(row)
    return rows


async def evaluate_row(
    ingestion: BinanceFuturesIngestion,
    row: dict[str, Any],
    semaphore: asyncio.Semaphore,
    now_ms: int,
) -> dict[str, Any] | None:
    try:
        symbol = str(row["symbol"])
        signal_timestamp_ms = int(row["timestamp_ms"])
        entry = float(row["price"])
        side = str(row["signal_type"])
    except (KeyError, TypeError, ValueError):
        return None
    if entry <= 0.0 or side not in {"STRONG_LONG", "STRONG_SHORT"}:
        return None

    max_horizon_bars = max(HORIZONS) // 5
    try:
        signal_candle_open = signal_candle_open_ms(
            signal_timestamp_ms,
            int(row["candle_open_ms"]) if row.get("candle_open_ms") is not None else None,
        )
    except (TypeError, ValueError):
        return None
    final_candle_open = signal_candle_open + max_horizon_bars * INTERVAL_MS
    if final_candle_open + INTERVAL_MS > now_ms:
        return None

    async with semaphore:
        klines = await ingestion.fetch_symbol_closed_5m_klines(
            symbol,
            final_candle_open,
            history_bars=max_horizon_bars + 1,
        )
    if not klines or len(klines) != max_horizon_bars + 1:
        return None

    closes: dict[int, float] = {}
    expected_times = [signal_candle_open + i * INTERVAL_MS for i in range(max_horizon_bars + 1)]
    for row_kline, expected_open in zip(klines, expected_times):
        try:
            open_ms = int(row_kline[0])
            close = float(row_kline[4])
        except (IndexError, TypeError, ValueError):
            return None
        if open_ms != expected_open or not math.isfinite(close) or close <= 0.0:
            return None
        closes[(open_ms - signal_candle_open) // INTERVAL_MS] = close

    result = dict(row)
    result["entry_candle_open_ms"] = signal_candle_open
    for minutes in HORIZONS:
        bars = minutes // 5
        exit_px = closes.get(bars)
        if exit_px is None:
            result[f"return_{minutes}m"] = None
            result[f"net_return_{minutes}m"] = None
            continue
        raw = exit_px / entry - 1.0
        signed = raw if side == "STRONG_LONG" else -raw
        result[f"return_{minutes}m"] = signed
        friction = float(row.get("applied_friction_rt_pct", 0.0))
        result[f"net_return_{minutes}m"] = signed - friction
    return result


async def main_async(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.ledger))
    ingestion = BinanceFuturesIngestion(symbols=[])
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    now_ms = int(time.time() * 1000)
    try:
        tasks = [evaluate_row(ingestion, row, semaphore, now_ms) for row in rows]
        evaluated = [row for row in await asyncio.gather(*tasks) if row is not None]
    finally:
        await ingestion.stop()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in evaluated:
                fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, output)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    print(f"signals={len(rows)} matured_evaluated={len(evaluated)} output={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="data/signal_ledger.jsonl")
    parser.add_argument("--output", default="data/signal_outcomes.jsonl")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
