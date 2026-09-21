#!/usr/bin/env python3
"""Export the idempotent live feature recorder to the backtest JSONL format."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

from engine.research_recorder import ResearchRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/research/features.sqlite3")
    parser.add_argument("--shard-dir", default="data/research/shards")
    parser.add_argument("--output", default="data/research/features.jsonl")
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--end-ms", type=int)
    args = parser.parse_args()
    n = ResearchRecorder(Path(args.db), Path(args.shard_dir)).export_jsonl(Path(args.output), args.start_ms, args.end_ms)
    print(f"rows={n} output={args.output}")


if __name__ == "__main__":
    main()
