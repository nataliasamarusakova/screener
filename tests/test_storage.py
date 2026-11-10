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
    base = datetime.now(timezone.utc) - timedelta(hours=8)
    for index in range(3):
        row = _row()
        row.spot_imbalance_4h = 2.0 + index
        store.add_snapshot(base + timedelta(hours=4 * index), "PERPETUAL", [row])

    assert store.spot_sequence_status("BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10) is True

    negative = _row()
    negative.spot_imbalance_4h = -1.0
    store.add_snapshot(base + timedelta(hours=12), "PERPETUAL", [negative])
    assert store.spot_sequence_status("BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10) is False


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


def test_snapshot_due_samples_only_near_the_four_hour_p1_interval(tmp_path):
    base = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
    store = Store(tmp_path / "sample-data")
    store.add_snapshot(base, "PERPETUAL", [_row()])
    assert store.snapshot_due(base + timedelta(hours=3, minutes=49), interval_hours=4) is False
    assert store.snapshot_due(base + timedelta(hours=3, minutes=50), interval_hours=4) is False
    assert store.snapshot_due(base + timedelta(hours=3, minutes=59), interval_hours=4) is False
    assert store.snapshot_due(base + timedelta(hours=4), interval_hours=4) is True


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


def test_spot_sequence_does_not_reuse_an_older_run_after_latest_snapshot_gap(tmp_path):
    store = Store(tmp_path / "spot-sequence-gap-data")
    base = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    for index, hours in enumerate((0, 4, 8, 16, 20)):
        row = _row()
        row.spot_imbalance_4h = 2.0 + index
        store.add_snapshot(base + timedelta(hours=hours), "PERPETUAL", [row])
    # The older t0/t4/t8 sequence was positive, but the *latest* three snapshots
    # contain an 8-hour gap and cannot be treated as a consecutive P1 sequence.
    assert store.spot_sequence_status("BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10) is None


def test_spot_sequence_rejects_overlapping_windows_but_allows_bounded_late_poll(tmp_path):
    store = Store(tmp_path / "spot-sequence-boundary-data")
    base = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    # 3h55 gaps overlap the rolling 4H windows and cannot count as independent.
    for index, hour in enumerate((0, 3.92, 7.84)):
        row = _row()
        row.spot_imbalance_4h = 1.0 + index
        store.add_snapshot(base + timedelta(hours=hour), "PERPETUAL", [row])
    assert store.spot_sequence_status("BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10) is None

    # Replace with clean 4h gaps; a small late scheduler poll is tolerated.
    clean = Store(tmp_path / "spot-sequence-late-data")
    for index, hour in enumerate((0, 4.05, 8.10)):
        row = _row()
        row.spot_imbalance_4h = 1.0 + index
        clean.add_snapshot(base + timedelta(hours=hour), "PERPETUAL", [row])
    assert clean.spot_sequence_status("BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10) is True


def test_spot_sequence_does_not_mix_all_contracts_with_perpetual_history(tmp_path):
    store = Store(tmp_path / "contract-type-history-data")
    base = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    for hours, contract_type in ((0, "ALL_CONTRACTS"), (4, "ALL_CONTRACTS"), (8, "PERPETUAL"), (12, "PERPETUAL")):
        row = _row()
        row.spot_imbalance_4h = 2.0
        store.add_snapshot(base + timedelta(hours=hours), contract_type, [row])

    # The unfiltered history has three consecutive positive observations at t=4/8/12,
    # but only two belong to the currently selected Perpetual universe.
    assert len(store.recent_symbol_history("BTCUSDT_PERP.A", limit=10)) == 4
    assert len(store.recent_symbol_history("BTCUSDT_PERP.A", limit=10, contract_type="PERPETUAL")) == 2
    assert store.spot_sequence_status(
        "BTCUSDT_PERP.A", required=3, interval_hours=4, tolerance_minutes=10,
        contract_type="PERPETUAL",
    ) is None


def test_snapshot_due_starts_new_history_immediately_after_contract_view_change(tmp_path):
    store = Store(tmp_path / "contract-switch-sampling-data")
    base = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    store.add_snapshot(base, "ALL_CONTRACTS", [_row()])

    # A different contracts universe cannot be compared to the prior snapshot;
    # it starts a separate series rather than waiting for the old series' 4H timer.
    assert store.snapshot_due(
        base + timedelta(hours=1), interval_hours=4, contract_type="PERPETUAL"
    ) is True
    assert store.snapshot_due(
        base + timedelta(hours=1), interval_hours=4, contract_type="ALL_CONTRACTS"
    ) is False
