from datetime import datetime, timedelta, timezone

from screener.models import MarketRow
from screener.storage import Store


def _row(raw: dict | None = None) -> MarketRow:
    return MarketRow(
        symbol="BTCUSDT_PERP.A",
        name="Bitcoin",
        price=70_000,
        chg24_pct=1.0,
        oi_usd=10_000_000,
        futures_imbalance_4h=2.0,
        spot_imbalance_4h=3.0,
        raw=raw or {},
    )


def test_identical_parsed_values_do_not_create_a_new_snapshot(tmp_path):
    store = Store(tmp_path / "data")
    first_time = datetime.now(timezone.utc)
    store.add_snapshot(first_time, "PERPETUAL", [_row({"row_number": 1})])

    assert store.snapshot_unchanged("PERPETUAL", [_row({"row_number": 99})]) is True
    assert store.snapshot_unchanged("ALL_CONTRACTS", [_row()]) is False

    changed = _row()
    changed.price = 70_100
    assert store.snapshot_unchanged("PERPETUAL", [changed]) is False


def test_spot_sequence_requires_distinct_snapshots_near_four_hour_spacing(tmp_path):
    store = Store(tmp_path / "data")
    base = datetime.now(timezone.utc) - timedelta(hours=12)
    for index in range(3):
        row = _row()
        row.spot_imbalance_4h = 2.0 + index
        store.add_snapshot(base + timedelta(hours=4 * index), "PERPETUAL", [row])

    as_of = base + timedelta(hours=8)
    assert store.spot_sequence_status(
        "BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10, as_of=as_of
    ) is True

    negative = _row()
    negative.spot_imbalance_4h = -1.0
    latest_at = base + timedelta(hours=12)
    store.add_snapshot(latest_at, "PERPETUAL", [negative])
    assert store.spot_sequence_status(
        "BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10, as_of=latest_at
    ) is False


def test_prune_retains_recent_market_history_and_drops_dom_raw(tmp_path):
    store = Store(tmp_path / "data")
    now = datetime.now(timezone.utc)
    old = _row({"row_number": 1, "price_raw": "$1"})
    store.add_snapshot(now - timedelta(days=40), "PERPETUAL", [old])
    current = _row({"row_number": 99, "price_raw": "$2"})
    store.add_snapshot(now, "PERPETUAL", [current])

    deleted = store.prune(snapshot_retention_days=30)
    assert deleted["snapshots_deleted"] == 1
    history = store.recent_symbol_history("BTCUSDT_PERP.A", limit=10)
    assert len(history) == 1
    assert "raw" not in history[0]


def test_storage_is_jsonl_only_and_records_committed_snapshot(tmp_path):
    import json

    data_dir = tmp_path / "data"
    store = Store(data_dir)
    store.add_snapshot(datetime.now(timezone.utc), "PERPETUAL", [_row()])

    expected = {
        "snapshots.jsonl",
        "market_rows.jsonl",
        "notifications.jsonl",
        "runs.jsonl",
    }
    assert {path.name for path in data_dir.iterdir()} == expected
    assert not list(data_dir.glob("*.sqlite*"))

    snapshot_records = [json.loads(line) for line in (data_dir / "snapshots.jsonl").read_text().splitlines()]
    row_records = [json.loads(line) for line in (data_dir / "market_rows.jsonl").read_text().splitlines()]
    assert snapshot_records[0]["record_type"] == "snapshot"
    assert snapshot_records[0]["row_count"] == 1
    assert row_records[0]["record_type"] == "market_row"
    assert row_records[0]["data"]["symbol"] == "BTCUSDT_PERP.A"


def test_uncommitted_rows_from_interrupted_snapshot_are_ignored(tmp_path):
    store = Store(tmp_path / "data")
    store.add_snapshot(datetime.now(timezone.utc), "PERPETUAL", [_row()])
    store._append_jsonl(store.market_rows_path, {
        "record_type": "market_row",
        "snapshot_id": "interrupted-without-manifest",
        "symbol": "BTCUSDT_PERP.A",
        "data": {"symbol": "BTCUSDT_PERP.A", "spot_imbalance_4h": 99.0},
    })

    history = store.recent_symbol_history("BTCUSDT_PERP.A", limit=10)
    assert len(history) == 1
    assert history[0]["spot_imbalance_4h"] == 3.0


def test_notifications_and_run_events_are_appended_as_jsonl(tmp_path):
    import json

    store = Store(tmp_path / "data")
    run_id = store.start_run()
    store.record_notification("BTC|LONG A|OBSERVATION", 123)
    store.finish_run(run_id, row_count=1, result_count=1, status="OK")

    notifications = [json.loads(line) for line in store.notifications_path.read_text().splitlines()]
    runs = [json.loads(line) for line in store.runs_path.read_text().splitlines()]
    assert store.notification_exists("BTC|LONG A|OBSERVATION")
    assert len(notifications) == 1
    assert notifications[0]["telegram_message_id"] == "123"
    assert [record["record_type"] for record in runs] == ["run_started", "run_finished"]
    assert runs[0]["run_id"] == runs[1]["run_id"] == run_id


def test_malformed_jsonl_line_is_skipped_without_losing_following_records(tmp_path):
    store = Store(tmp_path / "data")
    store._append_jsonl(store.notifications_path, {"event_key": "first", "sent_at": datetime.now(timezone.utc).isoformat()})
    with store.notifications_path.open("a", encoding="utf-8") as stream:
        stream.write("{this-is-not-json}\n")
    store._append_jsonl(store.notifications_path, {"event_key": "last", "sent_at": datetime.now(timezone.utc).isoformat()})

    assert store.notification_exists("first")
    assert store.notification_exists("last")



def test_append_repairs_a_truncated_trailing_jsonl_record(tmp_path):
    import json

    store = Store(tmp_path / "data")
    store.notifications_path.write_text('{"event_key":"valid"}\n{"event_key":"partial', encoding="utf-8")
    store._append_jsonl(store.notifications_path, {
        "record_type": "notification",
        "event_key": "next",
        "sent_at": datetime.now(timezone.utc).isoformat(),
    })

    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event_key"] for record in records] == ["valid", "next"]


def test_prune_removes_orphan_rows_even_when_no_snapshot_expired(tmp_path):
    store = Store(tmp_path / "data")
    store.add_snapshot(datetime.now(timezone.utc), "PERPETUAL", [_row()])
    store._append_jsonl(store.market_rows_path, {
        "record_type": "market_row",
        "snapshot_id": "orphan-without-manifest",
        "symbol": "BTCUSDT_PERP.A",
        "data": {"symbol": "BTCUSDT_PERP.A", "spot_imbalance_4h": 999.0},
    })

    deleted = store.prune()

    assert deleted["snapshots_deleted"] == 0
    assert len(list(store._read_jsonl(store.market_rows_path))) == 1
    assert store.recent_symbol_history("BTCUSDT_PERP.A", limit=10)[0]["spot_imbalance_4h"] == 3.0


def test_notification_delivery_state_is_per_recipient_and_legacy_rows_are_broadcast(tmp_path):
    store = Store(tmp_path / "data")
    event = "XRP|SHORT D|WAIT_BOUNCE"
    store.record_notification(event, 1001, chat_id="chat-a")

    assert store.notification_exists(event, "chat-a") is True
    assert store.notification_exists(event, "chat-b") is False
    assert store.notification_due(event, 4, "chat-a") is False
    assert store.notification_due(event, 4, "chat-b") is True

    legacy_event = "BTC|LONG A|OBSERVATION"
    store.record_notification(legacy_event, 1002)
    assert store.notification_exists(legacy_event, "chat-a") is True
    assert store.notification_exists(legacy_event, "chat-b") is True
    assert store.notification_due(legacy_event, 4, "chat-a") is False
    assert store.notification_due(legacy_event, 4, "chat-b") is False


def test_snapshot_due_uses_fixed_utc_four_hour_slots_and_is_idempotent(tmp_path):
    base = datetime(2026, 11, 10, 4, 5, tzinfo=timezone.utc)
    store = Store(tmp_path / "sample-data")
    store.add_snapshot(base, "PERPETUAL", [_row()])

    assert store.snapshot_due(base + timedelta(minutes=30), interval_hours=4, tolerance_minutes=10) is False
    next_slot = base + timedelta(hours=3, minutes=55)
    assert store.snapshot_due(next_slot, interval_hours=4, tolerance_minutes=10) is True
    store.add_snapshot(next_slot, "PERPETUAL", [_row()])
    assert store.snapshot_due(next_slot + timedelta(minutes=5), interval_hours=4, tolerance_minutes=10) is False
    assert store.snapshot_due(next_slot + timedelta(hours=4), interval_hours=4, tolerance_minutes=10) is True


def test_latest_notification_state_tracks_transition_per_recipient(tmp_path):
    store = Store(tmp_path / "data")
    group = "STRKUSDT_PERP.A|LONG A — Trend Continuation"
    store.record_notification("event-A", 1, chat_id="chat-1", event_group=group, fingerprint="A")
    assert store.latest_notification_state(group, "chat-1")["fingerprint"] == "A"
    assert store.latest_notification_state(group, "chat-2") is None
    store.record_notification("event-B", 2, chat_id="chat-1", event_group=group, fingerprint="B")
    assert store.latest_notification_state(group, "chat-1")["fingerprint"] == "B"
    store.record_notification("event-A-again", 3, chat_id="chat-1", event_group=group, fingerprint="A")
    assert store.latest_notification_state(group, "chat-1")["fingerprint"] == "A"


def test_spot_sequence_requires_latest_samples_and_freshness(tmp_path):
    store = Store(tmp_path / "fresh-spot")
    now = datetime.now(timezone.utc).replace(minute=5, second=0, microsecond=0)
    base = now - timedelta(days=2, hours=8)
    for index in range(3):
        row = _row()
        row.spot_imbalance_4h = 2.0
        store.add_snapshot(base + timedelta(hours=4 * index), "PERPETUAL", [row])
    assert store.spot_sequence_status("BTCUSDT_PERP.A", as_of=now) is None

    latest_negative = _row()
    latest_negative.spot_imbalance_4h = -9.0
    store.add_snapshot(now, "PERPETUAL", [latest_negative])
    assert store.spot_sequence_status("BTCUSDT_PERP.A", as_of=now) is False


def test_history_index_rejects_snapshot_with_hash_or_row_count_mismatch(tmp_path):
    import json

    store = Store(tmp_path / "integrity")
    second = _row()
    second.symbol = "ETHUSDT_PERP.A"
    second.name = "Ethereum"
    store.add_snapshot(datetime.now(timezone.utc), "PERPETUAL", [_row(), second])

    records = [json.loads(line) for line in store.market_rows_path.read_text(encoding="utf-8").splitlines()]
    records[0]["data"]["price"] = 123.45
    store._rewrite_jsonl(store.market_rows_path, records)

    assert store.recent_symbol_history("BTCUSDT_PERP.A") == []
    assert store.recent_symbol_history("ETHUSDT_PERP.A") == []


def test_prune_preserves_old_active_notification_but_removes_old_cancelled(tmp_path):
    import json

    store = Store(tmp_path / "notification-retention")
    old = (datetime.now(timezone.utc) - timedelta(days=181)).isoformat()
    active_group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    cancelled_group = "ETHUSDT_PERP.A|LONG A — Trend Continuation"
    store.record_notification("active-event", 1, chat_id="recipient-a", event_group=active_group,
        fingerprint="active", event_state="ACTIVE", direction="LONG", invalidation_level=10,
        pair="BTCUSDT", scenario_code="LONG A")
    store.record_notification("cancelled-event", 2, chat_id="recipient-a", event_group=cancelled_group,
        fingerprint="cancelled", event_state="CANCELLED", direction="LONG", invalidation_level=10,
        pair="ETHUSDT", scenario_code="LONG A")
    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["sent_at"] = old
    store._rewrite_jsonl(store.notifications_path, records)

    deleted = store.prune(notification_retention_days=180)

    assert deleted["notifications_deleted"] == 1
    active = store.active_notification_states(chat_ids=["recipient-a"])
    assert len(active) == 1
    assert active[0]["event_group"] == active_group


def test_notification_recipient_migration_stores_hmac_not_raw_chat_id(tmp_path):
    import json
    import hmac
    import hashlib

    store = Store(tmp_path / "recipient-migration")
    store.record_notification("event", 123, chat_id="test-chat-id-12345", event_group="BTC|LONG A",
        fingerprint="fingerprint", event_state="ACTIVE", direction="LONG")
    secret = "stable-private-key"
    migrated = store.migrate_notification_recipient_ids(
        lambda chat_id: hmac.new(secret.encode(), chat_id.encode(), hashlib.sha256).hexdigest()
    )

    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    assert migrated == 1
    assert "chat_id" not in records[0]
    assert records[0]["recipient_key"] == hmac.new(secret.encode(), b"test-chat-id-12345", hashlib.sha256).hexdigest()


def test_packaged_legacy_notification_hides_recipient_and_expands_explicitly(tmp_path):
    import json
    from pathlib import Path

    data_dir = Path(__file__).parents[1] / "data"
    raw = (data_dir / "notifications.jsonl").read_text(encoding="utf-8")
    assert '"chat_id"' not in raw
    assert "LEGACY_REDACTED_BROADCAST" in raw
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert records and all(item.get("recipient_scope") == "LEGACY_REDACTED_BROADCAST" for item in records)
    assert all("telegram_message_id" not in item for item in records)

    store = Store(tmp_path / "redacted-legacy-copy")
    store.notifications_path.write_text(raw, encoding="utf-8")
    # A redacted legacy recipient must never be broadcast to every configured
    # chat. It remains unresolved until exactly one recipient is configured.
    states = store.active_notification_states(recipient_keys=["recipient-a", "recipient-b"], contract_type="PERPETUAL")
    assert states == []
    store.notifications_path.write_text(raw, encoding="utf-8")
    assert store.migrate_redacted_broadcast_notifications(["recipient-a", "recipient-b"]) == 0
    assert store.migrate_redacted_broadcast_notifications(["recipient-a"]) == 1
    states = store.active_notification_states(recipient_keys=["recipient-a"], contract_type="PERPETUAL")
    assert len(states) == 1
    assert states[0]["recipient_key"] == "recipient-a"


def test_redacted_legacy_notification_is_materialized_per_recipient_once(tmp_path):
    import json

    store = Store(tmp_path / "redacted-legacy")
    record = {
        "record_type": "notification", "event_key": "legacy-event",
        "event_group": "BTC|LONG A — Trend Continuation",
        "fingerprint": "legacy-fingerprint", "event_state": "ACTIVE",
        "delivery_status": "SENT", "contract_type": "PERPETUAL",
        "recipient_scope": "LEGACY_REDACTED_BROADCAST",
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    store._append_jsonl(store.notifications_path, record)

    assert store.active_notification_states(recipient_keys=["recipient-a", "recipient-b"]) == []
    assert store.migrate_redacted_broadcast_notifications(["recipient-a", "recipient-b"]) == 0
    assert store.migrate_redacted_broadcast_notifications(["recipient-a"]) == 1
    assert store.migrate_redacted_broadcast_notifications(["recipient-a"]) == 0
    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert {item["recipient_key"] for item in records} == {"recipient-a"}
    assert all("recipient_scope" not in item and "chat_id" not in item for item in records)
    assert all("telegram_message_id" not in item for item in records)

    store.record_notification(
        "cancel-a", 2, recipient_key="recipient-a",
        event_group=record["event_group"], fingerprint=record["fingerprint"],
        event_state="CANCELLED", delivery_status="SENT",
    )
    active = store.active_notification_states(recipient_keys=["recipient-a", "recipient-b"])
    assert active == []
