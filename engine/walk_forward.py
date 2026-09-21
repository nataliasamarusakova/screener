"""Walk-forward validation with explicit train/OOS separation and parameter stability."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from engine.research import BacktestConfig, QuantBacktester, ResearchDataset, TradeRecord
from engine.signals import QuantSignalEngine


@dataclass(frozen=True)
class ParameterConfig:
    name: str
    params: dict[str, Any]

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ParameterConfig":
        name = str(row.get("name", "config"))
        params = dict(row.get("params", {}))
        return cls(name=name, params=params)


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 30 * 24 * 12
    validation_bars: int = 7 * 24 * 12
    test_bars: int = 7 * 24 * 12
    embargo_bars: int = 12
    min_train_trades: int = 50
    min_test_trades: int = 20


@dataclass
class FoldResult:
    fold: int
    train_start_ms: int
    train_end_ms: int
    validation_start_ms: int
    validation_end_ms: int
    test_start_ms: int
    test_end_ms: int
    selected_config: str | None
    train_candidates: list[dict[str, Any]]
    oos_stats: dict[str, Any]



def load_parameter_grid(path: Path) -> list[ParameterConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("configs", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("WFA parameter grid must contain a non-empty list")
    configs = [ParameterConfig.from_mapping(row) for row in rows]
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("WFA configuration names must be unique")
    if len(configs) > 50:
        raise ValueError("Refusing >50 WFA configurations; control the multiple-testing surface")
    return configs


def _stats(trades: Sequence[TradeRecord]) -> dict[str, Any]:
    values = [float(t.net_return) for t in trades if math.isfinite(float(t.net_return))]
    if not values:
        return {"n": 0, "mean": None, "std": None, "t_stat": None, "hit_rate": None, "profit_factor": None}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    std = math.sqrt(variance)
    t = mean / (std / math.sqrt(len(values))) if std > 0.0 and len(values) > 1 else None
    gains = sum(x for x in values if x > 0.0)
    losses = -sum(x for x in values if x < 0.0)
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "t_stat": t,
        "hit_rate": sum(x > 0.0 for x in values) / len(values),
        "profit_factor": gains / losses if losses > 0.0 else None,
    }


def _make_engine(params: dict[str, Any]) -> QuantSignalEngine:
    allowed = {
        "w_cvd", "w_fund", "w_oi", "w_micro", "w_whale",
        "funding_weight", "basis_weight", "z_history_min_samples", "score_tanh_scale",
        "sweep_booster_points", "strong_signal_threshold", "swing_buffer_bps",
        "atr_stop_multiplier", "max_atr_multiplier", "target_risk_reward",
        "friction_round_trip_pct", "min_effective_rrr", "trailing_activation_r",
        "trailing_distance_r", "risk_per_trade_pct", "max_leverage",
    }
    clean = {k: v for k, v in params.items() if k in allowed}
    return QuantSignalEngine(**clean)


def _all_timestamps(dataset: ResearchDataset) -> list[int]:
    values = {bar.timestamp_ms for symbol in dataset.symbols() for bar in dataset.bars(symbol)}
    return sorted(values)


def _slice_bounds(timestamps: Sequence[int], start_index: int, length: int) -> tuple[int, int] | None:
    if start_index < 0 or start_index >= len(timestamps) or length <= 0:
        return None
    end_index = min(len(timestamps) - 1, start_index + length - 1)
    return timestamps[start_index], timestamps[end_index]


def run_wfa(dataset: ResearchDataset, configs: Sequence[ParameterConfig], cfg: WalkForwardConfig) -> list[FoldResult]:
    timestamps = _all_timestamps(dataset)
    if len(timestamps) < cfg.train_bars + cfg.embargo_bars + cfg.validation_bars + cfg.embargo_bars + cfg.test_bars:
        return []

    results: list[FoldResult] = []
    cursor = 0
    fold = 0
    while cursor + cfg.train_bars + cfg.embargo_bars + cfg.validation_bars + cfg.embargo_bars + cfg.test_bars <= len(timestamps):
        train_start = timestamps[cursor]
        train_end = timestamps[cursor + cfg.train_bars - 1]
        validation_start = timestamps[cursor + cfg.train_bars + cfg.embargo_bars]
        validation_end = timestamps[cursor + cfg.train_bars + cfg.embargo_bars + cfg.validation_bars - 1]
        test_start = timestamps[cursor + cfg.train_bars + cfg.embargo_bars + cfg.validation_bars + cfg.embargo_bars]
        test_end = timestamps[cursor + cfg.train_bars + cfg.embargo_bars + cfg.validation_bars + cfg.embargo_bars + cfg.test_bars - 1]

        ranked: list[tuple[float, ParameterConfig, dict[str, Any]]] = []
        for candidate in configs:
            engine = _make_engine(candidate.params)
            backtester = QuantBacktester(engine, BacktestConfig())
            trades = backtester.run(dataset, signal_start_ms=train_start, signal_end_ms=validation_end, require_exit_within_end=True)
            train_trades = [t for t in trades if train_start <= t.signal_timestamp_ms <= train_end]
            val_trades = [t for t in trades if validation_start <= t.signal_timestamp_ms <= validation_end]
            train_stats = _stats(train_trades)
            val_stats = _stats(val_trades)
            if train_stats["n"] < cfg.min_train_trades or val_stats["n"] == 0:
                score = float("-inf")
            else:
                # Selection uses validation t-stat only after a minimum train sample gate.
                score = float(val_stats["t_stat"] if val_stats["t_stat"] is not None else float("-inf"))
            ranked.append((score, candidate, {"train": train_stats, "validation": val_stats}))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[0] if ranked and math.isfinite(ranked[0][0]) else None
        selected_name = selected[1].name if selected else None
        oos_stats: dict[str, Any]
        if selected is None:
            oos_stats = {"n": 0, "reason": "NO_CONFIG_PASSED_TRAIN_GATE"}
        else:
            engine = _make_engine(selected[1].params)
            backtester = QuantBacktester(engine, BacktestConfig())
            oos_trades = backtester.run(dataset, signal_start_ms=test_start, signal_end_ms=test_end, require_exit_within_end=True)
            oos_stats = _stats(oos_trades)
            oos_stats["passed_min_test_trades"] = oos_stats["n"] >= cfg.min_test_trades

        results.append(
            FoldResult(
                fold=fold,
                train_start_ms=train_start,
                train_end_ms=train_end,
                validation_start_ms=validation_start,
                validation_end_ms=validation_end,
                test_start_ms=test_start,
                test_end_ms=test_end,
                selected_config=selected_name,
                train_candidates=[{"name": c.name, "score": score, **stats} for score, c, stats in ranked],
                oos_stats=oos_stats,
            )
        )
        cursor += cfg.test_bars
        fold += 1
    return results


def save_results(results: Sequence[FoldResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
