from __future__ import annotations

import argparse
import logging

from .config import load_settings
from .service import run_scan_once


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Analyze the saved Coinalyze Perpetual table and send Telegram observation alerts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze without sending Telegram messages")
    args = parser.parse_args()

    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    dry_run = args.dry_run or settings.dry_run
    result = run_scan_once(settings, dry_run=dry_run)
    print(
        f"Scan: {result['status']} | rows={result['rows']} | "
        f"contract={result['contract_type']} | dry_run={dry_run}"
    )
    for key, path in zip(("report", "csv", "json"), result["output_paths"]):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
