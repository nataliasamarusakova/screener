#!/usr/bin/env python3
"""Safely re-arm one permanently failed Telegram notification for a later scan.

Run with ``python -m screener.rearm_failed_notification --help``.
The command is read-only by default. Use --apply only after fixing the cause of
Telegram's permanent rejection (for example, chat ID or bot permissions).
Always copy the exact event_key, recipient_key and contract_type from
``data/notifications.jsonl``. The recipient_key is the stored HMAC value, not
the raw Telegram chat ID. With --apply, the command backs up the JSONL file and
removes only records whose event_state and delivery_status are both FAILED.
The next scan can retry only while the market trigger is still valid and fresh.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .merge_state import _canonical, _read_jsonl, _write_atomic


def rearm_failed_notification(
    data_dir: Path,
    event_key: str,
    recipient_key: str,
    contract_type: str,
    *,
    apply: bool = False,
) -> dict[str, str | int | bool]:
    """Remove only the exact terminal FAILED record, retaining a backup."""
    event_key = str(event_key).strip()
    recipient_key = str(recipient_key).strip()
    contract_type = str(contract_type).strip().upper()
    if not event_key or not recipient_key or not contract_type:
        raise ValueError("event_key, recipient_key and contract_type must be non-empty")

    path = Path(data_dir) / "notifications.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Notification history does not exist: {path}")
    records = _read_jsonl(path)
    matching_indices = [
        index for index, record in enumerate(records)
        if record.get("event_key") == event_key
        and record.get("recipient_key") == recipient_key
        and str(record.get("contract_type") or "").upper() == contract_type
    ]
    if not matching_indices:
        raise ValueError("No notification record matches the exact event, recipient and contract")

    matching = [records[index] for index in matching_indices]
    if any(
        str(record.get("event_state") or "").upper() != "FAILED"
        or str(record.get("delivery_status") or "").upper() != "FAILED"
        for record in matching
    ):
        raise ValueError("A matching notification is not exclusively terminal FAILED; refusing to re-arm it")

    result: dict[str, str | int | bool] = {
        "event_key": event_key,
        "recipient_key": recipient_key,
        "contract_type": contract_type,
        "matching_records": len(matching_indices),
        "applied": bool(apply),
    }
    if not apply:
        return result

    backup = path.with_name(
        f"{path.name}.before-rearm-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.bak"
    )
    backup.write_bytes(path.read_bytes())
    remove = set(matching_indices)
    retained = [record for index, record in enumerate(records) if index not in remove]
    _write_atomic(path, "".join(_canonical(record) + "\n" for record in retained))
    result["backup"] = str(backup)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "First run without --apply to inspect the exact match. After applying, "
            "review and commit data/notifications.jsonl to the scanner branch. "
            "Re-arming does not force-send an expired signal."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--event-key", required=True)
    parser.add_argument("--recipient-key", required=True, help="Stored HMAC recipient key, not the Telegram chat ID")
    parser.add_argument("--contract-type", required=True)
    parser.add_argument("--apply", action="store_true", help="Write the change; otherwise only show what would be removed")
    args = parser.parse_args()
    result = rearm_failed_notification(
        args.data_dir, args.event_key, args.recipient_key, args.contract_type, apply=args.apply,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    if not args.apply:
        print("Dry-run only. Add --apply after correcting the Telegram configuration/problem.")


if __name__ == "__main__":
    main()
