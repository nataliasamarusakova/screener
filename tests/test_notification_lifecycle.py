from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from screener.metrics import analyze_universe
from screener.models import Check, MarketRow, ScenarioResult
from screener.service import _collect_notifications
from screener.storage import Store
from screener.telegram import format_cancellation_message, format_reversal_message


def _market():
    return MarketRow(
        symbol="FILUSDT_PERP.A", name="Filecoin", price=0.765,
        chg24_pct=-1.0, market_cap_usd=1_000_000_000,
        volume24_usd=250_000_000, oi_usd=100_000_000,
        oi_chg24_pct=-2.0, oi_chg24_usd=-2_000_000,
        oi_chg4h_pct=-0.5, oi_chg4h_usd=-500_000,
        oi_vol_ratio=0.4, oi_mktcap_ratio=0.1, vol_mktcap_ratio=0.25,
        fr_avg_pct=0.001, fr_oiw_pct=0.001, pfr_oiw_pct=0.001,
        liq_total24_usd=1_000_000, liq_long24_usd=800_000, liq_short24_usd=200_000,
        ls_ratio_1d=1.4, futures_imbalance_4h=-5, spot_imbalance_4h=-3,
    )


def _settings():
    return SimpleNamespace(telegram_bot_token="test-token", telegram_chat_ids=["chat-1"])


def _confirmed_result(row, direction: str, scenario_code: str, *, level: float, close: float):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    hour_open = (now_ms // 3_600_000) * 3_600_000 - 3_600_000
    four_hour_open = (now_ms // 14_400_000) * 14_400_000 - 14_400_000
    is_long = direction == "LONG"
    scenario = (
        "LONG A — Trend Continuation" if scenario_code == "LONG A"
        else "SHORT D — Long Liquidation Cascade" if scenario_code == "SHORT D"
        else f"{scenario_code} — Test"
    )
    trigger_key = "trend_retest_long" if scenario_code == "LONG A" else "weak_bounce_rejection"
    trigger = (
        {"confirmed": True, "level": level, "retest_open_time_ms": hour_open - 3_600_000}
        if scenario_code == "LONG A"
        else {"confirmed": True, "resistance": level, "rejection_open_time_ms": hour_open - 3_600_000}
    )
    context_4h = {
        "last_close": close,
        "last_close_time_ms": four_hour_open + 14_400_000 - 1,
        "last_confirmed_support": min(close, level) - 0.02,
        "last_confirmed_resistance": max(close, level) + 0.02,
        "ema20": close,
        "ema50": close - 0.01 if is_long else close + 0.01,
    }
    technical = {
        "binance_symbol": "FILUSDT",
        trigger_key: trigger,
        "last_closed_1h": {
            "open_time_ms": hour_open,
            "close_time_ms": hour_open + 3_600_000 - 1,
            "open": close, "high": close + 0.01, "low": close - 0.01, "close": close,
        },
        "context_4h": context_4h,
    }
    checks = [Check(f"P1 condition {index}", True, "pass", "required") for index in range(6)]
    return ScenarioResult(
        symbol=row.symbol, name=row.market.name, scenario=scenario,
        direction=direction, checks=checks, quality_flags=[],
        status="CONFIRMED_SIGNAL", technical=technical,
    )


def _waiting_short(row, *, last_close: float):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    hour_open = (now_ms // 3_600_000) * 3_600_000 - 3_600_000
    return ScenarioResult(
        symbol=row.symbol, name=row.market.name,
        scenario="SHORT D — Long Liquidation Cascade", direction="SHORT",
        checks=[Check("cascade", True, "passed", "P1")], quality_flags=[], status="WAIT_BOUNCE",
        technical={
            "binance_symbol": "FILUSDT",
            "last_closed_1h": {
                "open_time_ms": hour_open,
                "close_time_ms": hour_open + 3_600_000 - 1,
                "open": last_close, "high": last_close + 0.002,
                "low": last_close - 0.002, "close": last_close,
            },
        },
    )


class FakeTelegram:
    sent: list[tuple[str, list[str]]] = []

    def __init__(self, token, chat_ids):
        self.chat_ids = list(chat_ids)
        self.configured = bool(token and chat_ids)

    def send(self, message, chat_ids=None):
        recipients = list(self.chat_ids if chat_ids is None else chat_ids)
        self.sent.append((message, recipients))
        return [
            {"chat_id": chat_id, "sent": True, "delivery_unknown": False, "message_id": len(self.sent)}
            for chat_id in recipients
        ]


def _enriched():
    return analyze_universe([_market()], {"thresholds": {}})[0]


def _activate_short(store: Store, row, *, level: float = 0.7659):
    group = f"{row.symbol}|SHORT D — Long Liquidation Cascade"
    store.record_notification(
        "old-short-event", 11, chat_id="chat-1", event_group=group,
        fingerprint="old-short-fingerprint", event_state="ACTIVE", direction="SHORT",
        invalidation_level=level, pair="FILUSDT", scenario_code="SHORT D",
        contract_type="PERPETUAL", delivery_status="SENT",
    )
    return group


def test_cancellation_message_has_requested_standalone_text_and_dot_decimal():
    assert format_cancellation_message("FILUSDT", "SHORT", "SHORT D", 0.7659) == (
        "⚪ <b>СЦЕНАРИЙ ОТМЕНЁН</b> — <code>FILUSDT</code>\n"
        "SHORT D: закрытие 1H выше <code>0.76590000</code>."
    )


def test_reversal_message_is_a_standalone_three_line_post():
    assert format_reversal_message("FILUSDT", "SHORT", "SHORT D", "LONG", "LONG A") == (
        "🔄 <b>РАЗВОРОТ</b> — <code>FILUSDT</code>\n"
        "SHORT D → LONG A\n"
        "Противоположный сценарий подтверждён закрытыми 1H-свечами."
    )


def test_active_short_is_cancelled_once_and_not_repeated(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "state")
    _activate_short(store, row)

    cancelled = _waiting_short(row, last_close=0.7660)
    first = _collect_notifications([cancelled], rows, _settings(), store, True, False)
    assert len(FakeTelegram.sent) == 1
    message, chats = FakeTelegram.sent[0]
    assert chats == ["chat-1"]
    assert message == (
        "⚪ <b>СЦЕНАРИЙ ОТМЕНЁН</b> — <code>FILUSDT</code>\n"
        "SHORT D: закрытие 1H выше <code>0.76590000</code>."
    )
    assert first[0]["type"] == "cancellation"
    assert store.active_notification_states(chat_ids=["chat-1"]) == []

    # The CANCELLED state is durable across the next GitHub Actions cron run.
    assert _collect_notifications([cancelled], rows, _settings(), store, True, False) == []
    assert len(FakeTelegram.sent) == 1


def test_long_scenario_is_cancelled_only_after_closed_1h_below_saved_support(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "long-state")
    group = f"{row.symbol}|LONG A — Trend Continuation"
    store.record_notification(
        "old-long-event", 10, chat_id="chat-1", event_group=group,
        fingerprint="old-long", event_state="ACTIVE", direction="LONG",
        invalidation_level=0.7500, pair="FILUSDT", scenario_code="LONG A",
        contract_type="PERPETUAL", delivery_status="SENT",
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    open_ms = (now_ms // 3_600_000) * 3_600_000 - 3_600_000
    waiting = ScenarioResult(
        symbol=row.symbol, name=row.market.name, scenario="LONG A — Trend Continuation",
        direction="LONG", checks=[], quality_flags=[], status="WAITING_FOR_1H_BREAKOUT_RETEST",
        technical={"binance_symbol": "FILUSDT", "last_closed_1h": {
            "open_time_ms": open_ms, "close_time_ms": open_ms + 3_599_999,
            "open": 0.749, "high": 0.751, "low": 0.748, "close": 0.749,
        }},
    )
    result = _collect_notifications([waiting], rows, _settings(), store, True, False)
    assert result[0]["type"] == "cancellation"
    assert FakeTelegram.sent[0][0] == (
        "⚪ <b>СЦЕНАРИЙ ОТМЕНЁН</b> — <code>FILUSDT</code>\n"
        "LONG A: закрытие 1H ниже <code>0.75000000</code>."
    )


def test_confirmed_opposite_scenario_sends_one_reversal_post_not_a_second_signal(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "state")
    old_group = _activate_short(store, row)

    # LONG A is fully confirmed and the closed 1H candle reclaimed the old SHORT level.
    new_long = _confirmed_result(row, "LONG", "LONG A", level=0.7500, close=0.7700)
    outcomes = _collect_notifications([new_long], rows, _settings(), store, True, False)

    assert len(FakeTelegram.sent) == 1
    message, chats = FakeTelegram.sent[0]
    assert chats == ["chat-1"]
    assert message.startswith("🔄 <b>РАЗВОРОТ</b> — <code>FILUSDT</code>")
    assert "SHORT D → LONG A" in message
    assert "СЦЕНАРИЙ ОТМЕНЁН" not in message
    assert "CONFIRMED" not in message
    assert outcomes[0]["type"] == "reversal"

    active = store.active_notification_states(chat_ids=["chat-1"])
    assert len(active) == 1
    assert active[0]["event_group"] == f"{row.symbol}|LONG A — Trend Continuation"
    assert active[0]["direction"] == "LONG"
    assert _collect_notifications([new_long], rows, _settings(), store, True, False) == []
    assert len(FakeTelegram.sent) == 1


def test_opposite_candidate_does_not_cancel_or_reverse_before_old_level_breaks(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "state")
    _activate_short(store, row)

    # The LONG A trigger exists, but the previous SHORT resistance has not been reclaimed.
    long_not_yet_reversed = _confirmed_result(row, "LONG", "LONG A", level=0.7400, close=0.7600)
    assert _collect_notifications([long_not_yet_reversed], rows, _settings(), store, True, False) == []
    assert FakeTelegram.sent == []
    assert len(store.active_notification_states(chat_ids=["chat-1"])) == 1


def test_unconfirmed_opposite_candidate_does_not_create_reversal(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "state")
    _activate_short(store, row)

    candidate = _confirmed_result(row, "LONG", "LONG A", level=0.7400, close=0.7600)
    candidate.status = "WAITING_FOR_1H_BREAKOUT_RETEST"
    candidate.technical["trend_retest_long"]["confirmed"] = False
    assert _collect_notifications([candidate], rows, _settings(), store, True, False) == []
    assert FakeTelegram.sent == []
    assert len(store.active_notification_states(chat_ids=["chat-1"])) == 1


def test_old_event_is_cancelled_by_its_saved_level_not_a_new_pivot(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "state")
    _activate_short(store, row)

    # Price remains below the old SHORT invalidation level; an unrelated new pivot
    # in the current chart must not cancel the previously delivered event.
    still_active = _waiting_short(row, last_close=0.7600)
    still_active.technical["weak_bounce_rejection"] = {"invalidated": True, "resistance": 0.7700}
    assert _collect_notifications([still_active], rows, _settings(), store, True, False) == []
    assert FakeTelegram.sent == []
    assert len(store.active_notification_states(chat_ids=["chat-1"])) == 1
