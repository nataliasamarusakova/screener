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
    store.record_notification("BTC|LONG A|OBSERVATION", 123, recipient_key="recipient-a", contract_type="PERPETUAL")
    store.finish_run(run_id, row_count=1, result_count=1, status="OK")

    notifications = [json.loads(line) for line in store.notifications_path.read_text().splitlines()]
    runs = [json.loads(line) for line in store.runs_path.read_text().splitlines()]
    assert store.notification_exists("BTC|LONG A|OBSERVATION", recipient_key="recipient-a", contract_type="PERPETUAL")
    assert len(notifications) == 1
    assert notifications[0]["telegram_message_id"] == "123"
    assert [record["record_type"] for record in runs] == ["run_started", "run_finished"]
    assert runs[0]["run_id"] == runs[1]["run_id"] == run_id


def test_malformed_jsonl_line_is_skipped_without_losing_following_records(tmp_path):
    store = Store(tmp_path / "data")
    current_record = {
        "record_type": "notification", "recipient_key": "recipient-a",
        "contract_type": "PERPETUAL", "delivery_status": "SENT",
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    store._append_jsonl(store.notifications_path, {**current_record, "event_key": "first"})
    with store.notifications_path.open("a", encoding="utf-8") as stream:
        stream.write("{this-is-not-json}\n")
    store._append_jsonl(store.notifications_path, {**current_record, "event_key": "last"})

    assert store.notification_exists("first", recipient_key="recipient-a", contract_type="PERPETUAL")
    assert store.notification_exists("last", recipient_key="recipient-a", contract_type="PERPETUAL")



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


def test_notification_delivery_state_is_isolated_per_recipient_and_contract(tmp_path):
    store = Store(tmp_path / "data")
    event = "XRP|SHORT D|WAIT_BOUNCE"
    store.record_notification(event, 1001, recipient_key="recipient-a", contract_type="PERPETUAL")

    assert store.notification_exists(event, recipient_key="recipient-a", contract_type="PERPETUAL") is True
    assert store.notification_exists(event, recipient_key="recipient-b", contract_type="PERPETUAL") is False
    assert store.notification_exists(event, recipient_key="recipient-a", contract_type="INVERSE") is False
    assert store.notification_due(event, 4, recipient_key="recipient-a", contract_type="PERPETUAL") is False
    assert store.notification_due(event, 4, recipient_key="recipient-b", contract_type="PERPETUAL") is True


def test_snapshot_due_requires_full_rolling_interval_and_is_idempotent(tmp_path):
    base = datetime(2026, 11, 10, 4, 5, tzinfo=timezone.utc)
    store = Store(tmp_path / "sample-data")
    store.add_snapshot(base, "PERPETUAL", [_row()])

    assert store.snapshot_due(base + timedelta(minutes=30), interval_hours=4, tolerance_minutes=10) is False
    assert store.snapshot_due(base + timedelta(hours=3, minutes=55), interval_hours=4, tolerance_minutes=10) is False
    next_observation = base + timedelta(hours=4)
    assert store.snapshot_due(next_observation, interval_hours=4, tolerance_minutes=10) is True
    store.add_snapshot(next_observation, "PERPETUAL", [_row()])
    assert store.snapshot_due(next_observation + timedelta(minutes=5), interval_hours=4, tolerance_minutes=10) is False
    assert store.snapshot_due(next_observation + timedelta(hours=4), interval_hours=4, tolerance_minutes=10) is True


def test_latest_notification_state_tracks_transition_per_recipient(tmp_path):
    store = Store(tmp_path / "data")
    group = "STRKUSDT_PERP.A|LONG A — Trend Continuation"
    store.record_notification("event-A", 1, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=group, fingerprint="A", event_state="ACTIVE", direction="LONG", invalidation_level=70_000)
    assert store.latest_notification_state(group, recipient_key="recipient-a", contract_type="PERPETUAL")["fingerprint"] == "A"
    assert store.latest_notification_state(group, recipient_key="recipient-b", contract_type="PERPETUAL") is None
    store.record_notification("event-B", 2, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=group, fingerprint="B", event_state="ACTIVE", direction="LONG", invalidation_level=70_000)
    assert store.latest_notification_state(group, recipient_key="recipient-a", contract_type="PERPETUAL")["fingerprint"] == "B"
    store.record_notification("event-A-again", 3, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=group, fingerprint="A", event_state="ACTIVE", direction="LONG", invalidation_level=70_000)
    assert store.latest_notification_state(group, recipient_key="recipient-a", contract_type="PERPETUAL")["fingerprint"] == "A"


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
    store.record_notification("active-event", 1, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=active_group,
        fingerprint="active", event_state="ACTIVE", direction="LONG", invalidation_level=10,
        pair="BTCUSDT", scenario_code="LONG A")
    store.record_notification("cancelled-event", 2, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=cancelled_group,
        fingerprint="cancelled", event_state="CANCELLED", direction="LONG", invalidation_level=10,
        pair="ETHUSDT", scenario_code="LONG A")
    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["sent_at"] = old
    store._rewrite_jsonl(store.notifications_path, records)

    deleted = store.prune(notification_retention_days=180)

    assert deleted["notifications_deleted"] == 1
    active = store.active_notification_states(recipient_keys=["recipient-a"])
    assert len(active) == 1
    assert active[0]["event_group"] == active_group


def test_read_jsonl_skips_invalid_utf8_line_and_continues(tmp_path):
    path = tmp_path / "notifications.jsonl"
    path.write_bytes(b'{"id":1}\n{\"id\":\xff}\n{\"id\":2}\n')

    records = list(Store._read_jsonl(path))

    assert records == [{"id": 1}, {"id": 2}]


def test_notification_lookup_is_scoped_by_recipient_and_contract_type(tmp_path):
    store = Store(tmp_path / "contract-scope")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    for contract_type, event_key in (("PERPETUAL", "perp-event"), ("INVERSE", "inverse-event")):
        store.record_notification(
            event_key, 1, recipient_key="recipient-a", event_group=group,
            fingerprint=event_key, event_state="ACTIVE", direction="LONG",
            invalidation_level=70_000, contract_type=contract_type, delivery_status="SENT",
        )

    assert store.notification_exists("perp-event", recipient_key="recipient-a", contract_type="PERPETUAL")
    assert not store.notification_exists("inverse-event", recipient_key="recipient-a", contract_type="PERPETUAL")
    assert store.latest_notification_state(group, recipient_key="recipient-a", contract_type="PERPETUAL")["contract_type"] == "PERPETUAL"
    assert [item["contract_type"] for item in store.active_notification_states(
        recipient_keys=["recipient-a"], contract_type="PERPETUAL"
    )] == ["PERPETUAL"]




def test_new_and_canceled_statuses_are_not_treated_as_active(tmp_path):
    store = Store(tmp_path / "non-active-states")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    store.record_notification(
        "active-event", 1, recipient_key="recipient-a", event_group=group,
        fingerprint="active-fingerprint", event_state="ACTIVE", direction="LONG", invalidation_level=70_000,
        contract_type="PERPETUAL", delivery_status="SENT",
    )
    store.record_notification(
        "new-event", None, recipient_key="recipient-b", event_group=group,
        fingerprint="new-fingerprint", event_state="NEW", direction="LONG",
        contract_type="PERPETUAL", delivery_status="SENT",
    )
    store.record_notification(
        "cancelled-event", 2, recipient_key="recipient-a", event_group=group,
        fingerprint="active-fingerprint", event_state="CANCELED", direction="LONG",
        invalidation_level=70_000, contract_type="PERPETUAL", delivery_status="SENT",
    )

    assert store.active_notification_states(recipient_keys=["recipient-a", "recipient-b"]) == []


def test_active_notifications_keep_same_group_separate_across_contract_types(tmp_path):
    store = Store(tmp_path / "active-contract-isolation")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    for contract_type, event_key in (("PERPETUAL", "perp-event"), ("INVERSE", "inverse-event")):
        store.record_notification(
            event_key, 1, recipient_key="recipient-a", event_group=group,
            fingerprint=event_key, event_state="ACTIVE", direction="LONG",
            invalidation_level=70_000, contract_type=contract_type, delivery_status="SENT",
        )

    all_active = store.active_notification_states(recipient_keys=["recipient-a"])
    perpetual_active = store.active_notification_states(
        recipient_keys=["recipient-a"], contract_type="PERPETUAL"
    )
    inverse_active = store.active_notification_states(
        recipient_keys=["recipient-a"], contract_type="INVERSE"
    )

    assert {item["event_key"] for item in all_active} == {"perp-event", "inverse-event"}
    assert [item["event_key"] for item in perpetual_active] == ["perp-event"]
    assert [item["event_key"] for item in inverse_active] == ["inverse-event"]


def test_prune_preserves_active_notifications_per_contract_type(tmp_path):
    from datetime import datetime, timedelta, timezone
    import json

    store = Store(tmp_path / "prune-contract-isolation")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    old_sent_at = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    for contract_type, event_key in (("PERPETUAL", "perp-event"), ("INVERSE", "inverse-event")):
        store.record_notification(
            event_key, 1, recipient_key="recipient-a", event_group=group,
            fingerprint=event_key, event_state="ACTIVE", direction="LONG",
            invalidation_level=70_000, contract_type=contract_type, delivery_status="SENT",
        )
    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["sent_at"] = old_sent_at
    store._rewrite_jsonl(store.notifications_path, records)

    store.prune(notification_retention_days=180)

    retained = list(Store._read_jsonl(store.notifications_path))
    assert {item["event_key"] for item in retained} == {"perp-event", "inverse-event"}


def test_prune_recognizes_cancellation_for_same_contract(tmp_path):
    from datetime import datetime, timedelta, timezone
    import json

    store = Store(tmp_path / "prune-current-transition")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    old_sent_at = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    store.record_notification(
        "active-event", 1, recipient_key="recipient-a", event_group=group,
        fingerprint="active-fingerprint", event_state="ACTIVE", direction="LONG", invalidation_level=70_000,
        contract_type="PERPETUAL", delivery_status="SENT",
    )
    store.record_notification(
        "cancel-event", 2, recipient_key="recipient-a", contract_type="PERPETUAL", event_group=group,
        fingerprint="active-fingerprint", event_state="CANCELLED", direction="LONG",
        invalidation_level=70_000, delivery_status="SENT",
    )
    records = [json.loads(line) for line in store.notifications_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["sent_at"] = old_sent_at
    store._rewrite_jsonl(store.notifications_path, records)

    store.prune(notification_retention_days=180)

    assert list(Store._read_jsonl(store.notifications_path)) == []


def test_notification_record_requires_current_recipient_and_contract_scope(tmp_path):
    import pytest

    store = Store(tmp_path / "strict-notification-schema")
    with pytest.raises(TypeError):
        store.record_notification("event", 1, recipient_key="recipient-a")
    with pytest.raises(TypeError):
        store.record_notification("event", 1, contract_type="PERPETUAL")


def test_active_notifications_ignore_records_without_current_schema_fields(tmp_path):
    store = Store(tmp_path / "strict-active-schema")
    group = "BTCUSDT_PERP.A|LONG A — Trend Continuation"
    now = datetime.now(timezone.utc).isoformat()
    store._append_jsonl(store.notifications_path, {
        "record_type": "notification", "event_key": "untyped-event",
        "recipient_key": "recipient-a", "event_group": group,
        "fingerprint": "untyped", "event_state": "ACTIVE",
        "direction": "LONG", "invalidation_level": 70_000,
        "delivery_status": "SENT", "sent_at": now,
    })
    store._append_jsonl(store.notifications_path, {
        "record_type": "notification", "event_key": "chat-id-event",
        "chat_id": "recipient-a", "contract_type": "PERPETUAL",
        "event_group": group, "fingerprint": "chat-id", "event_state": "ACTIVE",
        "direction": "LONG", "invalidation_level": 70_000,
        "delivery_status": "SENT", "sent_at": now,
    })
    store._append_jsonl(store.notifications_path, {
        "record_type": "notification", "event_key": "incomplete-event",
        "recipient_key": "recipient-a", "contract_type": "PERPETUAL",
        "event_group": group, "fingerprint": "incomplete", "event_state": "ACTIVE",
        "delivery_status": "SENT", "sent_at": now,
    })

    active = store.active_notification_states(recipient_keys=["recipient-a"])

    assert active == []
    store.prune()
    assert list(Store._read_jsonl(store.notifications_path)) == []


def test_notification_storage_exposes_only_current_schema_api(tmp_path):
    store = Store(tmp_path / "current-schema-api")

    assert not any("migrate" in name for name in dir(store))

    import inspect
    assert "chat_ids" not in inspect.signature(store.active_notification_states).parameters
    assert "chat_id" not in inspect.signature(store.record_notification).parameters


def test_active_notification_write_rejects_incomplete_lifecycle_state(tmp_path):
    import pytest

    store = Store(tmp_path / "required-active-fields")
    with pytest.raises(ValueError, match="event_group"):
        store.record_notification(
            "active-event", recipient_key="recipient-a", contract_type="PERPETUAL",
            event_state="ACTIVE", direction="LONG", invalidation_level=70_000,
        )
    with pytest.raises(ValueError, match="invalidation_level"):
        store.record_notification(
            "active-event", recipient_key="recipient-a", contract_type="PERPETUAL",
            event_group="BTCUSDT_PERP.A|LONG A", fingerprint="fingerprint",
            event_state="ACTIVE", direction="LONG",
        )
    with pytest.raises(ValueError, match="direction"):
        store.record_notification(
            "active-event", recipient_key="recipient-a", contract_type="PERPETUAL",
            event_group="BTCUSDT_PERP.A|LONG A", fingerprint="fingerprint",
            event_state="ACTIVE", direction="UNKNOWN", invalidation_level=70_000,
        )
