from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from screener.metrics import analyze_universe
from screener.models import Check, MarketRow, ScenarioResult
from screener.service import _collect_notifications
from screener.storage import Store
from screener.telegram import format_cancellation_message, format_reversal_message


def _market(symbol: str = "FILUSDT_PERP.A", name: str = "Filecoin"):
    return MarketRow(
        symbol=symbol, name=name, price=0.765,
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


def _recipient_key():
    from screener.service import _recipient_storage_key
    return _recipient_storage_key("chat-1", _settings())


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
        "old-short-event", 11, recipient_key=_recipient_key(), event_group=group,
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
    assert store.active_notification_states(recipient_keys=[_recipient_key()]) == []
    persisted = store.notifications_path.read_text(encoding="utf-8")
    assert '"chat_id"' not in persisted
    assert "chat-1" not in persisted
    assert "hmac-sha256:" in persisted

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
        "old-long-event", 10, recipient_key=_recipient_key(), event_group=group,
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

    active = store.active_notification_states(recipient_keys=[_recipient_key()])
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
    assert len(store.active_notification_states(recipient_keys=[_recipient_key()])) == 1


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
    assert len(store.active_notification_states(recipient_keys=[_recipient_key()])) == 1


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
    assert len(store.active_notification_states(recipient_keys=[_recipient_key()])) == 1



def test_two_new_confirmed_scenarios_are_sent_as_two_separate_telegram_messages(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    markets = [_market("FILUSDT_PERP.A", "Filecoin"), _market("ONDOUSDT_PERP.A", "Ondo")]
    rows, _ = analyze_universe(markets, {"thresholds": {}})
    by_symbol = {row.symbol: row for row in rows}
    store = Store(tmp_path / "two-separate-messages")

    short_d = _confirmed_result(
        by_symbol["FILUSDT_PERP.A"], "SHORT", "SHORT D", level=0.7800, close=0.7650
    )
    long_a = _confirmed_result(
        by_symbol["ONDOUSDT_PERP.A"], "LONG", "LONG A", level=0.7400, close=0.7650
    )
    # Give each signal its actual exchange contract; the helper defaults to FILUSDT.
    long_a.technical["binance_symbol"] = "ONDOUSDT"

    outcomes = _collect_notifications(
        [short_d, long_a], rows, _settings(), store,
        contract_alerts_allowed=True, dry_run=False,
    )

    # Exactly two sendMessage calls, each containing one complete signal card.
    from screener import telegram as telegram_module
    assert not hasattr(telegram_module, "format_alert_batch")
    assert len(FakeTelegram.sent) == 2
    assert len(outcomes) == 2
    assert all(item["type"] == "confirmed_signal" and item["sent"] for item in outcomes)
    messages = [message for message, recipients in FakeTelegram.sent]
    assert all(recipients == ["chat-1"] for _, recipients in FakeTelegram.sent)
    assert any("FILUSDT" in message and "SHORT D" in message for message in messages)
    assert any("ONDOUSDT" in message and "LONG A" in message for message in messages)
    assert all("НОВЫЕ ПОДТВЕРЖДЁННЫЕ СЦЕНАРИИ" not in message for message in messages)
    assert all("────────────" not in message for message in messages)
    assert all("P1 · НОВЫЕ СЦЕНАРИИ" not in message for message in messages)
    assert len(store.active_notification_states(recipient_keys=[_recipient_key()])) == 2

    # A following cron run must not resend either of the two separately posted cards.
    repeated = _collect_notifications(
        [short_d, long_a], rows, _settings(), store,
        contract_alerts_allowed=True, dry_run=False,
    )
    assert repeated == []
    assert len(FakeTelegram.sent) == 2


def test_active_signal_is_monitored_and_cancelled_when_symbol_disappears_from_coinalyze(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    row = _enriched()[0]
    store = Store(tmp_path / "missing-symbol")
    _activate_short(store, row)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    hour_open = (now_ms // 3_600_000) * 3_600_000 - 3_600_000
    technical = {
        "binance_symbol": "FILUSDT",
        "last_closed_1h": {
            "open_time_ms": hour_open,
            "close_time_ms": hour_open + 3_599_999,
            "open": 0.7660, "high": 0.7670, "low": 0.7655, "close": 0.7660,
        },
    }

    outcomes = _collect_notifications(
        [], [], _settings(), store, True, False,
        technical_by_symbol={row.symbol: technical},
    )

    assert len(outcomes) == 1 and outcomes[0]["type"] == "cancellation"
    assert FakeTelegram.sent[0][1] == ["chat-1"]
    assert store.active_notification_states(recipient_keys=[_recipient_key()]) == []


def test_missing_symbol_without_fresh_binance_candle_is_not_auto_cancelled(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    row = _enriched()[0]
    store = Store(tmp_path / "missing-symbol-unknown")
    _activate_short(store, row)

    outcomes = _collect_notifications([], [], _settings(), store, True, False)

    assert outcomes == []
    assert FakeTelegram.sent == []
    assert len(store.active_notification_states(recipient_keys=[_recipient_key()])) == 1


def test_active_symbol_absent_from_coinalyze_remains_a_chart_candidate(tmp_path):
    from screener.config import DEFAULT_P1_CONFIG
    from screener.service import _preliminary_candidate_symbols

    row = _enriched()[0]
    store = Store(tmp_path / "active-candidate")
    _activate_short(store, row)
    settings = SimpleNamespace(
        telegram_bot_token="test-token", telegram_chat_ids=["chat-1"],
        telegram_storage_hash_key="",
        p1=DEFAULT_P1_CONFIG,
    )
    candidates = _preliminary_candidate_symbols([], None, settings, store, contract_type="PERPETUAL")

    assert row.symbol in candidates


def test_contract_guard_blocks_new_signals_but_allows_confirmed_existing_cancellation(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    row = _enriched()[0]
    store = Store(tmp_path / "guarded-cancellation")
    _activate_short(store, row)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    hour_open = (now_ms // 3_600_000) * 3_600_000 - 3_600_000
    technical = {
        "binance_symbol": "FILUSDT",
        "last_closed_1h": {
            "open_time_ms": hour_open,
            "close_time_ms": hour_open + 3_599_999,
            "open": 0.7660, "high": 0.7670, "low": 0.7655, "close": 0.7660,
        },
    }

    outcomes = _collect_notifications(
        [], [], _settings(), store, False, False,
        contract_type="ALL_CONTRACTS",
        technical_by_symbol={row.symbol: technical},
    )

    assert len(outcomes) == 1 and outcomes[0]["type"] == "cancellation"
    assert FakeTelegram.sent[0][1] == ["chat-1"]
    records = [
        __import__("json").loads(line)
        for line in store.notifications_path.read_text(encoding="utf-8").splitlines()
    ]
    cancellation = records[-1]
    assert cancellation["event_state"] == "CANCELLED"
    assert cancellation["contract_type"] == "PERPETUAL"
    assert store.active_notification_states(recipient_keys=[_recipient_key()]) == []


def test_permanent_failure_for_new_signal_is_recorded_and_not_retried(tmp_path, monkeypatch):
    import screener.service as service
    from screener.service import _notification_identity

    class PermanentFailureTelegram(FakeTelegram):
        calls = 0

        def send(self, message, chat_ids=None):
            type(self).calls += 1
            return [{
                "chat_id": (chat_ids or self.chat_ids)[0], "sent": False,
                "delivery_unknown": False, "permanent_failure": True,
                "error": "Telegram HTTP 403",
            }]

    monkeypatch.setattr(service, "TelegramClient", PermanentFailureTelegram)
    PermanentFailureTelegram.calls = 0
    rows = _enriched()
    row = rows[0]
    result = _confirmed_result(row, "LONG", "LONG A", level=0.7500, close=0.7700)
    identity = _notification_identity(result)
    assert identity is not None
    store = Store(tmp_path / "terminal-send-failure")

    first = _collect_notifications([result], rows, _settings(), store, True, False, contract_type="PERPETUAL")
    second = _collect_notifications([result], rows, _settings(), store, True, False, contract_type="PERPETUAL")

    assert len(first) == 1 and len(second) == 0
    assert PermanentFailureTelegram.calls == 1
    assert store.notification_exists(identity[1], recipient_key=_recipient_key(), contract_type="PERPETUAL")
    assert store.active_notification_states(recipient_keys=[_recipient_key()], contract_type="PERPETUAL") == []


def test_permanent_cancellation_failure_is_terminal_without_clearing_active_alert(tmp_path, monkeypatch):
    import screener.service as service

    class PermanentFailureTelegram(FakeTelegram):
        calls = 0

        def send(self, message, chat_ids=None):
            type(self).calls += 1
            return [{
                "chat_id": (chat_ids or self.chat_ids)[0], "sent": False,
                "delivery_unknown": False, "permanent_failure": True,
                "error": "Telegram HTTP 403",
            }]

    monkeypatch.setattr(service, "TelegramClient", PermanentFailureTelegram)
    PermanentFailureTelegram.calls = 0
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "terminal-cancel-failure")
    _activate_short(store, row)
    cancelled = _waiting_short(row, last_close=0.7660)

    first = _collect_notifications([cancelled], rows, _settings(), store, True, False, contract_type="PERPETUAL")
    second = _collect_notifications([cancelled], rows, _settings(), store, True, False, contract_type="PERPETUAL")

    assert first == [] and second == []
    assert PermanentFailureTelegram.calls == 1
    active = store.active_notification_states(recipient_keys=[_recipient_key()], contract_type="PERPETUAL")
    assert len(active) == 1
    assert active[0]["event_key"] == "old-short-event"


def test_active_other_contract_does_not_suppress_current_contract_notification(tmp_path, monkeypatch):
    import screener.service as service
    from screener.service import _notification_identity

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    result = _confirmed_result(row, "LONG", "LONG A", level=0.7500, close=0.7700)
    identity = _notification_identity(result)
    assert identity is not None
    group, event_key, fingerprint, level = identity
    store = Store(tmp_path / "contract-isolated-alerts")
    store.record_notification(
        event_key, 11, recipient_key=_recipient_key(), event_group=group,
        fingerprint=fingerprint, event_state="ACTIVE", direction="LONG",
        invalidation_level=level, pair="FILUSDT", scenario_code="LONG A",
        contract_type="INVERSE", delivery_status="SENT",
    )

    outcomes = _collect_notifications(
        [result], rows, _settings(), store, True, False, contract_type="PERPETUAL"
    )

    assert len(FakeTelegram.sent) == 1
    assert len(outcomes) == 1 and outcomes[0]["type"] == "confirmed_signal"
    active = store.active_notification_states(
        recipient_keys=[_recipient_key()], contract_type="PERPETUAL"
    )
    assert len(active) == 1 and active[0]["contract_type"] == "PERPETUAL"


def test_permanent_reversal_failure_is_recorded_without_replacing_old_active_alert(tmp_path, monkeypatch):
    import screener.service as service

    class PermanentFailureTelegram(FakeTelegram):
        calls = 0

        def send(self, message, chat_ids=None):
            type(self).calls += 1
            return [{
                "chat_id": (chat_ids or self.chat_ids)[0], "sent": False,
                "delivery_unknown": False, "permanent_failure": True,
                "error": "Telegram HTTP 403",
            }]

    monkeypatch.setattr(service, "TelegramClient", PermanentFailureTelegram)
    PermanentFailureTelegram.calls = 0
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "terminal-reversal-failure")
    _activate_short(store, row)
    new_long = _confirmed_result(row, "LONG", "LONG A", level=0.7500, close=0.7700)

    first = _collect_notifications(
        [new_long], rows, _settings(), store, True, False, contract_type="PERPETUAL"
    )
    second = _collect_notifications(
        [new_long], rows, _settings(), store, True, False, contract_type="PERPETUAL"
    )

    assert len(first) == 1 and first[0]["type"] == "reversal" and not first[0]["sent"]
    assert second == []
    assert PermanentFailureTelegram.calls == 1
    records = list(Store._read_jsonl(store.notifications_path))
    assert any(
        str(item.get("event_key", "")).startswith("FAILED|REVERSAL|")
        and item.get("event_state") == "FAILED"
        for item in records
    )
    active = store.active_notification_states(
        recipient_keys=[_recipient_key()], contract_type="PERPETUAL"
    )
    assert len(active) == 1
    assert active[0]["event_key"] == "old-short-event"


def test_persisted_reversal_marker_recovers_partial_state_without_resending(tmp_path, monkeypatch):
    import screener.service as service

    FakeTelegram.sent = []
    monkeypatch.setattr(service, "TelegramClient", FakeTelegram)
    rows = _enriched()
    row = rows[0]
    store = Store(tmp_path / "reversal-recovery")
    old_group = _activate_short(store, row)
    new_long = _confirmed_result(row, "LONG", "LONG A", level=0.7500, close=0.7700)
    from screener.service import _notification_identity
    identity = _notification_identity(new_long)
    assert identity is not None
    new_group, new_event_key, new_fingerprint, _ = identity
    reversal_key = f"REVERSAL|{row.symbol}|SHORT|LONG|{new_event_key}"

    # Simulate a crash after the Telegram reversal record was persisted, but
    # before the old/new per-scenario lifecycle rows were appended.
    store.record_notification(
        reversal_key, 77, recipient_key=_recipient_key(),
        event_group=f"REVERSAL|{row.symbol}|SHORT|LONG",
        fingerprint=new_fingerprint, event_state="SENT", direction="LONG",
        pair="FILUSDT", scenario_code="LONG A", contract_type="PERPETUAL",
        delivery_status="SENT",
    )

    first = _collect_notifications(
        [new_long], rows, _settings(), store, True, False, contract_type="PERPETUAL"
    )
    second = _collect_notifications(
        [new_long], rows, _settings(), store, True, False, contract_type="PERPETUAL"
    )

    assert first == [] and second == []
    assert FakeTelegram.sent == []
    assert store.latest_notification_state(
        old_group, recipient_key=_recipient_key(), contract_type="PERPETUAL"
    )["event_state"] == "REVERSED"
    assert store.latest_notification_state(
        new_group, recipient_key=_recipient_key(), contract_type="PERPETUAL"
    )["event_state"] == "ACTIVE"
    active = store.active_notification_states(
        recipient_keys=[_recipient_key()], contract_type="PERPETUAL"
    )
    assert len(active) == 1 and active[0]["event_group"] == new_group
