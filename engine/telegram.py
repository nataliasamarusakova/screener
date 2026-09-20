"""
Telegram Alerter for Strong Quantitative Signals (|Score| >= 75)
and Synthetic Liquidation Cascades.
Zero-dependency beyond aiohttp.
Includes intelligent deduplication to prevent repetitive spam.
"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from contracts import SignalEvent, SyntheticLiquidation

logger = logging.getLogger("telegram_alerter")
ALERT_CACHE_FILE = Path("data/alert_cache.json")


def _get_chat_ids() -> List[str]:
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TG_CHAT_ID") or ""
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


def _get_bot_token() -> str:
    return os.environ.get("TG_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN") or ""


class TelegramAlerter:
    """
    Asynchronous Telegram notification service with stateful throttling.
    """

    def __init__(self, cooldown_sec: int = 3600) -> None:
        self.bot_token = _get_bot_token()
        self.chat_ids = _get_chat_ids()
        self.cooldown_sec = cooldown_sec
        self.cache: Dict[str, Dict[str, float]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        if not ALERT_CACHE_FILE.exists():
            return {}
        try:
            data = json.loads(ALERT_CACHE_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("alert cache root must be an object")
            validated: Dict[str, Dict[str, float]] = {}
            for symbol, record in data.items():
                if not isinstance(symbol, str) or not isinstance(record, dict):
                    continue
                try:
                    timestamp = float(record["time"])
                    score = float(record["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(timestamp) or not math.isfinite(score):
                    continue
                validated[symbol] = {"time": timestamp, "score": score}
            return validated
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("telegram_alert_cache_load_failed path=%s error=%s", ALERT_CACHE_FILE, exc)
            return {}

    def _save_cache(self) -> None:
        target = ALERT_CACHE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(self.cache, ensure_ascii=False, separators=(",", ":"))
        fd, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp:
                temp.write(raw)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_name, target)
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, ValueError, TypeError) as exc:
            logger.error("telegram_alert_cache_save_failed path=%s error=%s", target, exc)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def _should_alert(self, symbol: str, current_score: float) -> bool:
        """Throttles repeated alerts unless score moves significantly or cooldown expires."""
        now = time.time()
        record = self.cache.get(symbol)
        if not record:
            return True

        last_time = record.get("time", 0.0)
        last_score = record.get("score", 0.0)

        # Cooldown expired
        if now - last_time >= self.cooldown_sec:
            return True

        # Direction flip or significant change in score (>= 20 points)
        if (last_score > 0 and current_score < 0) or (last_score < 0 and current_score > 0):
            return True

        if abs(current_score - last_score) >= 20.0:
            return True

        return False

    def _record_alert(self, symbol: str, score: float) -> None:
        self.cache[symbol] = {"time": time.time(), "score": score}
        self._save_cache()

    async def send_message(self, text: str) -> bool:
        if not self.bot_token or not self.chat_ids:
            logger.info("Telegram notification skipped: TG_BOT_TOKEN or TG_CHAT_IDS not configured.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        success = True

        async with aiohttp.ClientSession() as session:
            for chat_id in self.chat_ids:
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                try:
                    async with session.post(url, json=payload, timeout=10) as resp:
                        if resp.status != 200:
                            success = False
                            body = await resp.text()
                            logger.error(f"Failed to send Telegram alert to {chat_id}: {body}")
                except Exception as exc:
                    success = False
                    logger.error(f"Exception sending Telegram message to {chat_id}: {exc}")

        return success

    async def process_and_dispatch_signals(
        self,
        signals: List[SignalEvent],
        synthetic_liqs: List[SyntheticLiquidation]
    ) -> int:
        """
        Dispatches alerts for qualifying STRONG signals (|Score| >= 75) and synthetic liquidations.
        Returns number of sent alerts.
        """
        if not self.bot_token or not self.chat_ids:
            return 0

        sent_count = 0

        # 1. Process Strong Signals
        for sig in signals:
            if sig.signal_type in ("STRONG_LONG", "STRONG_SHORT"):
                if not self._should_alert(sig.symbol, sig.composite_score):
                    continue

                msg = self.format_signal_html(sig)
                ok = await self.send_message(msg)
                if ok:
                    self._record_alert(sig.symbol, sig.composite_score)
                    sent_count += 1

        # 2. Process Critical Synthetic Liquidations
        for liq in synthetic_liqs:
            if liq.anomaly_ratio >= 1.5:
                liq_key = f"{liq.symbol}_LIQ"
                if not self._should_alert(liq_key, liq.anomaly_ratio):
                    continue

                msg = self.format_liquidation_html(liq)
                ok = await self.send_message(msg)
                if ok:
                    self._record_alert(liq_key, liq.anomaly_ratio)
                    sent_count += 1

        return sent_count

    @staticmethod
    def format_signal_html(sig: SignalEvent) -> str:
        is_long = sig.signal_type == "STRONG_LONG"
        badge = "🟢 <b>STRONG LONG SIGNAL</b>" if is_long else "🔴 <b>STRONG SHORT SIGNAL</b>"

        # Gate status badge
        gate_badge = "✅ PASSED" if sig.gate_status == "PASSED" else f"🚫 {html.escape(sig.gate_status[:40])}"

        # Wyckoff sweep/reclaim badge
        sweep_badge = "⚡ <b>WYCKOFF SPRING / SWEEP RECLAIM CONFIRMED</b>" if sig.sweep_reclaim else ""

        # Whale sentiment context
        if sig.z_whale_sentiment >= 1.0:
            whale_text = f"{sig.z_whale_sentiment:+.2f} 🐋 Smart Money LONG"
        elif sig.z_whale_sentiment <= -1.0:
            whale_text = f"{sig.z_whale_sentiment:+.2f} 🐋 Smart Money SHORT"
        else:
            whale_text = f"{sig.z_whale_sentiment:+.2f} ⚖️ Neutral"

        sweep_line = f"\n ⚡ <b>Wyckoff Sweep:</b> {sweep_badge}" if sig.sweep_reclaim else ""

        return (
            f"{badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Symbol:</b> <code>{sig.symbol}</code>\n"
            f"📊 <b>Composite Score:</b> <b>{sig.composite_score:+.1f} / 100</b>\n"
            f"💵 <b>Ref Price:</b> <code>${sig.price:,.4f}</code>\n"
            f"🎯 <b>Target (TP):</b> <code>${sig.target_price:,.4f}</code>\n"
            f"🛑 <b>Invalidation (SL):</b> <code>${sig.invalidation_price:,.4f}</code>\n"
            f"⚖️ <b>Risk / Reward:</b> <b>{sig.risk_reward_ratio:.1f}x</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔬 <b>Quantitative Drivers:</b>\n"
            f" • <b>Funding 8h:</b> <code>{sig.funding_8h*100:+.4f}%</code>\n"
            f" • <b>Spot-Perp Basis:</b> <code>{sig.basis_bps:+.2f} bps</code>\n"
            f" • <b>L2 Imbalance (OBI):</b> <code>{sig.obi:+.3f}</code>\n"
            f" • <b>Flow Toxicity (VPIN):</b> <code>{sig.vpin:.3f}</code>\n"
            f" • <b>Z(CVD Div):</b> <code>{sig.z_cvd_div:+.2f}</code> | <b>Z(Trap):</b> <code>{sig.z_fund_trap:+.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏦 <b>Institutional Filters:</b>\n"
            f" • <b>Whale/Retail Z:</b> <code>{whale_text}</code>\n"
            f" • <b>BTC Rel Strength:</b> <code>{sig.relative_strength:+.2f}%</code>"
            f"{sweep_line}\n"
            f" • <b>Gate Status:</b> {gate_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>Point-in-Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(sig.timestamp_ms / 1000))}</i>"
        )

    @staticmethod
    def format_liquidation_html(liq: SyntheticLiquidation) -> str:
        side_badge = "🚨 <b>LONG LIQUIDATION CASCADE</b>" if liq.side == "LONG_LIQUIDATION" else "⚡ <b>SHORT SQUEEZE CASCADE</b>"
        return (
            f"{side_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Symbol:</b> <code>{liq.symbol}</code>\n"
            f"⚠️ <b>Hidden Binance Liquidation Detected!</b>\n"
            f"📉 <b>ΔOpen Interest:</b> <code>{liq.delta_oi:,.1f} contracts</code>\n"
            f"🌊 <b>Taker Volume:</b> <code>{liq.taker_volume:,.1f}</code>\n"
            f"💥 <b>Anomaly Ratio:</b> <b>{liq.anomaly_ratio:.2f}x</b>\n"
            f"📦 <b>Reconstructed Volume:</b> <code>{liq.estimated_liquidation_volume:,.1f}</code>\n"
            f"💵 <b>Price at Event:</b> <code>${liq.price:,.4f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(liq.timestamp_ms / 1000))}</i>"
        )
