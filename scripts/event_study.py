#!/usr/bin/env python3
"""Run forward-return event studies over all clean, signal-ready observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.factor_research import analyze_feature_event_study, summarize_feature_event_study
from engine.research import ResearchDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = ResearchDataset.from_jsonl(args.input) if args.input.suffix.lower() == ".jsonl" else ResearchDataset.from_csv(args.input)
    provenance = dataset.provenance_summary()
    if not provenance["clean"]:
        raise SystemExit("Refusing event study on mixed/legacy provenance dataset")

    observations = analyze_feature_event_study(dataset)
    payload = {
        "provenance": provenance,
        "observation_count": len(observations),
        "summary": summarize_feature_event_study(observations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
