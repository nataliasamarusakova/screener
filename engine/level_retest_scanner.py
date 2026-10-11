"""Standalone 12h resistance-to-support flip scan and alert route."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from binance_ingestion import BinanceFuturesIngestion
from engine.level_retest import (
    LevelRetestConfig,
    LevelRetestEvent,
    OHLCVCandle,
    detect_bullish_level_retest,
)
from engine.level_retest_universe import select_usdt_m_universe
from engine.telegram import TelegramAlerter

logger = logging.getLogger("level_retest_12h")


def _candle_from_binance(row: list) -> OHLCVCandle:
    return OHLCVCandle(
        open_time_ms=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_seen(path: Path) -> list[str]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        values = content.get("event_keys", []) if isinstance(content, dict) else []
        # Preserve insertion order so the bounded cache retains recent alerts.
        return list(dict.fromkeys(value for value in values if isinstance(value, str)))
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("level_retest_seen_cache_load_failed path=%s error=%s", path, exc)
        return []


def _alert_text(event: LevelRetestEvent) -> str:
    return (
        "🟢 <b>12H SUPPORT RETEST — RESISTANCE FLIP</b>\n"
        f"<b>{html.escape(event.symbol)}</b> | score: <b>{event.score:.1f}/100</b>\n\n"
        f"Former swing-high support: <b>{event.support_level:.8g}</b>\n"
        f"Support zone: {event.zone_low:.8g} – {event.zone_high:.8g}\n"
        f"Current price: <b>{event.price:.8g}</b> ({event.distance_to_level_pct:+.2f}% from level)\n"
        f"Confirmed former peaks: {event.touch_count}\n"
        f"Breakout close above level: +{event.breakout_pct:.2f}%\n"
        f"Observed bounce from retest low: +{event.bounce_pct:.2f}%\n"
        f"12h ATR: {event.atr_pct:.2f}% | estimated invalidation: {event.invalidation_price:.8g}\n\n"
        "Setup candidate only; not a trade order. Confirm the level and costs before acting."
    )


async def run_level_retest_mode(
    *,
    data_dir: Path = Path("data"),
    top_n_symbols: Optional[int] = None,
    concurrency_limit: Optional[int] = None,
) -> dict:
    """Run the independent 12h support-retest scan; does not create orders."""
    # This mode has its own universe controls and defaults to every available
    # USDT-margined Futures ticker. A positive top-N is an optional explicit cap;
    # 0 means no cap. Keep the legacy 5m TOP_N_SYMBOLS/volume filters isolated.
    top_n = int(
        top_n_symbols if top_n_symbols is not None
        else os.getenv("LEVEL_RETEST_TOP_N_SYMBOLS", "0")
    )
    concurrency = int(concurrency_limit or os.getenv("SCAN_CONCURRENCY", "25"))
    min_volume = float(os.getenv("LEVEL_RETEST_MIN_24H_VOLUME_USDT", "0"))
    history_bars = int(os.getenv("LEVEL_RETEST_HISTORY_BARS", "100"))
    config = LevelRetestConfig(
        min_touches=int(os.getenv("LEVEL_RETEST_MIN_TOUCHES", "2")),
        level_cluster_tolerance_pct=float(os.getenv("LEVEL_RETEST_CLUSTER_TOLERANCE_PCT", "0.012")),
        base_touch_tolerance_pct=float(os.getenv("LEVEL_RETEST_BASE_ZONE_PCT", "0.005")),
        atr_zone_multiplier=float(os.getenv("LEVEL_RETEST_ATR_ZONE_MULTIPLIER", "0.30")),
        max_touch_tolerance_pct=float(os.getenv("LEVEL_RETEST_MAX_ZONE_PCT", "0.015")),
        min_breakout_pct=float(os.getenv("LEVEL_RETEST_MIN_BREAKOUT_PCT", "0.004")),
        min_bounce_pct=float(os.getenv("LEVEL_RETEST_MIN_BOUNCE_PCT", "0.003")),
        max_price_extension_pct=float(os.getenv("LEVEL_RETEST_MAX_EXTENSION_PCT", "0.025")),
        min_score=float(os.getenv("LEVEL_RETEST_MIN_SCORE", "55")),
    )
    if top_n < 0 or concurrency < 1 or not math.isfinite(min_volume) or min_volume < 0.0:
        raise ValueError("Invalid 12h level-retest scan configuration")
    if history_bars < 20 or history_bars > 1499:
        raise ValueError("LEVEL_RETEST_HISTORY_BARS must be between 20 and 1499")

    ingestion = BinanceFuturesIngestion(symbols=[])
    scan_started_ms = int(time.time() * 1000)
    output_path = data_dir / "level_retest_12h_latest.json"
    seen_path = data_dir / "level_retest_12h_seen.json"
    seen_order = _load_seen(seen_path)
    seen_keys = set(seen_order)
    alerter = TelegramAlerter(cooldown_sec=0)
    semaphore = asyncio.Semaphore(concurrency)
    scanned_symbols = 0
    failed_symbols = 0
    events: list[LevelRetestEvent] = []

    try:
        tickers = await ingestion.fetch_universe_tickers()
        eligible_universe = select_usdt_m_universe(
            tickers,
            min_24h_volume_usdt=min_volume,
            top_n_symbols=0,
        )
        eligible_count = len(eligible_universe)
        universe = (
            eligible_universe[:top_n]
            if top_n > 0
            else eligible_universe
        )
        logger.info(
            "level_retest_universe_selected eligible=%d selected=%d cap=%s min_24h_volume_usdt=%s history_bars=%d",
            eligible_count, len(universe), top_n or "none", min_volume, history_bars,
        )

        async def analyze(item: tuple[str, float, float]) -> Optional[LevelRetestEvent]:
            nonlocal scanned_symbols, failed_symbols
            symbol, _volume, ticker_price = item
            async with semaphore:
                try:
                    raw = await ingestion.fetch_symbol_12h_klines_with_current(
                        symbol, history_bars=history_bars,
                    )
                    if not raw:
                        failed_symbols += 1
                        return None
                    now_ms = int(time.time() * 1000)
                    candles = [_candle_from_binance(row) for row in raw]
                    completed: list[OHLCVCandle] = []
                    live: Optional[OHLCVCandle] = None
                    for row, candle in zip(raw, candles):
                        close_time_ms = int(row[6])
                        if close_time_ms < now_ms:
                            completed.append(candle)
                        else:
                            live = candle
                    if len(completed) < 20:
                        failed_symbols += 1
                        return None
                    scanned_symbols += 1
                    # Prefer the current 12h kline close if it is fresher than a ticker snapshot.
                    latest_price = live.close if live is not None else ticker_price
                    return detect_bullish_level_retest(
                        symbol=symbol,
                        completed_candles=completed,
                        live_candle=live,
                        current_price=latest_price,
                        config=config,
                    )
                except Exception as exc:
                    failed_symbols += 1
                    logger.warning("level_retest_symbol_failed symbol=%s error=%s", symbol, exc)
                    return None

        results = await asyncio.gather(*(analyze(item) for item in universe))
        events = sorted((event for event in results if event is not None), key=lambda event: event.score, reverse=True)
        payload = {
            "mode": "level_retest_12h",
            "timeframe": "12h",
            "scan_started_ms": scan_started_ms,
            "scan_completed_ms": int(time.time() * 1000),
            "eligible_universe_size": eligible_count,
            "universe_size": len(universe),
            "top_n_cap": top_n if top_n > 0 else None,
            "min_24h_volume_usdt": min_volume,
            "scanned_symbols": scanned_symbols,
            "failed_symbols": failed_symbols,
            "candidate_count": len(events),
            "config": asdict(config),
            "candidates": [event.to_dict() for event in events],
        }
        _atomic_write_json(output_path, payload)

        delivered = 0
        new_seen = list(seen_order)
        for event in events:
            if event.event_key in seen_keys:
                continue
            try:
                ok = await alerter.send_message(_alert_text(event))
            except Exception as exc:
                logger.warning("level_retest_alert_error symbol=%s error=%s", event.symbol, exc)
                ok = False
            if ok:
                delivered += 1
                if event.event_key not in seen_keys:
                    seen_keys.add(event.event_key)
                    new_seen.append(event.event_key)
        # Bounded persistent cache. Save only successful sends so failed alerts retry next run.
        _atomic_write_json(seen_path, {
            "updated_at_ms": int(time.time() * 1000),
            "event_keys": new_seen[-5000:],
        })
        print(
            f"[LEVEL_RETEST_12H] scanned={scanned_symbols}/{len(universe)} "
            f"failed={failed_symbols} candidates={len(events)} alerts_sent={delivered}"
        )
        for event in events[:20]:
            print(
                f"  {event.symbol:14s} score={event.score:5.1f} level={event.support_level:.8g} "
                f"price={event.price:.8g} touches={event.touch_count} "
                f"dist={event.distance_to_level_pct:+.2f}%"
            )
        print(f"[LEVEL_RETEST_12H] results saved to {output_path}")
        return payload
    finally:
        await ingestion.stop()
