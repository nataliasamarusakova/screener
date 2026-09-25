"""Research/backtest primitives for Quant Screener.

The research layer is deliberately separated from live ingestion. It consumes a
point-in-time feature dataset and never reaches Binance by itself, so historical
results are reproducible and auditable.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from engine.signals import QuantSignalEngine

INTERVAL_MS = 5 * 60 * 1000
SUPPORTED_FORWARD_MINUTES = (5, 15, 30, 60)


@dataclass(frozen=True)
class ResearchBar:
    timestamp_ms: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float
    delta_oi_pct: float
    funding_rate_8h: float
    basis_bps: float
    obi: float
    vpin: float
    cvd_divergence_score: float
    whale_divergence_score: float
    atr_pct: float
    recent_high: float
    recent_low: float
    spread_bps: float = 0.0
    gate_long_status: str = "PASSED"
    gate_short_status: str = "PASSED"
    relative_strength: float = 0.0
    sweep_reclaim: bool = False
    sweep_pattern: str = "NONE"
    recorded_signal: dict[str, Any] | None = None
    candidate_signal: dict[str, Any] | None = None
    strategy_revision: str = "LEGACY_UNKNOWN"
    code_revision: str = "UNKNOWN"
    config_fingerprint: str = "UNKNOWN"
    research_schema_version: int = 0

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ResearchBar":
        required = [
            "timestamp_ms", "symbol", "open", "high", "low", "close", "volume",
            "open_interest", "delta_oi_pct", "funding_rate_8h", "basis_bps", "obi",
            "vpin", "cvd_divergence_score", "whale_divergence_score", "atr_pct",
            "recent_high", "recent_low",
        ]
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Research bar missing required fields: {missing}")

        def f(key: str, default: float | None = None) -> float:
            value = row.get(key, default)
            if value is None:
                if default is None:
                    raise ValueError(f"Missing numeric field: {key}")
                value = default
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite research field: {key}")
            return value

        def s(key: str, default: str) -> str:
            value = str(row.get(key, default))
            return value

        timestamp_ms = int(row["timestamp_ms"])
        if timestamp_ms <= 0 or timestamp_ms % INTERVAL_MS != INTERVAL_MS - 1:
            raise ValueError(f"timestamp_ms must be a 5m candle close: {timestamp_ms}")
        symbol = str(row["symbol"]).upper().strip()
        if not symbol:
            raise ValueError("symbol is required")

        open_px, high, low, close = (f(x) for x in ("open", "high", "low", "close"))
        if min(open_px, high, low, close) <= 0.0 or high < max(open_px, close) or low > min(open_px, close):
            raise ValueError(f"Invalid OHLC for {symbol}@{timestamp_ms}")
        if f("volume") < 0.0 or f("open_interest") <= 0.0:
            raise ValueError(f"Invalid volume/open_interest for {symbol}@{timestamp_ms}")
        atr_pct = f("atr_pct")
        if atr_pct <= 0.0:
            raise ValueError(f"atr_pct must be > 0 for {symbol}@{timestamp_ms}")
        vpin = f("vpin")
        if not 0.0 <= vpin <= 1.0:
            raise ValueError(f"vpin must be in [0,1] for {symbol}@{timestamp_ms}")

        recorded_signal = row.get("signal_event_final")
        candidate_signal = row.get("signal_event")
        if recorded_signal is not None and not isinstance(recorded_signal, dict):
            raise ValueError("signal_event_final must be an object when present")
        if candidate_signal is not None and not isinstance(candidate_signal, dict):
            raise ValueError("signal_event must be an object when present")
        return cls(
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            open=open_px,
            high=high,
            low=low,
            close=close,
            volume=f("volume"),
            open_interest=f("open_interest"),
            delta_oi_pct=f("delta_oi_pct"),
            funding_rate_8h=f("funding_rate_8h"),
            basis_bps=f("basis_bps"),
            obi=f("obi"),
            vpin=vpin,
            cvd_divergence_score=f("cvd_divergence_score"),
            whale_divergence_score=f("whale_divergence_score"),
            atr_pct=atr_pct,
            recent_high=f("recent_high"),
            recent_low=f("recent_low"),
            spread_bps=max(0.0, f("spread_bps", 0.0)),
            gate_long_status=s("gate_long_status", "PASSED"),
            gate_short_status=s("gate_short_status", "PASSED"),
            relative_strength=f("relative_strength", 0.0),
            sweep_reclaim=str(row.get("sweep_reclaim", "False")).strip().lower() in {"1", "true", "yes", "y"},
            sweep_pattern=s("sweep_pattern", "NONE"),
            recorded_signal=recorded_signal,
            candidate_signal=candidate_signal,
            strategy_revision=s("strategy_revision", "LEGACY_UNKNOWN"),
            code_revision=s("code_revision", "UNKNOWN"),
            config_fingerprint=s("config_fingerprint", "UNKNOWN"),
            research_schema_version=int(row.get("research_schema_version", 0) or 0),
        )


class ResearchDataset:
    """Strict point-in-time research dataset grouped by symbol."""

    def __init__(self, bars: Iterable[ResearchBar]):
        grouped: dict[str, list[ResearchBar]] = {}
        for bar in bars:
            grouped.setdefault(bar.symbol, []).append(bar)
        self.by_symbol = grouped
        self._segments: dict[str, list[list[ResearchBar]]] = {}
        self._gap_count = 0
        for symbol, rows in grouped.items():
            rows.sort(key=lambda x: x.timestamp_ms)
            seen: set[int] = set()
            segments: list[list[ResearchBar]] = []
            current_segment: list[ResearchBar] = []
            previous = None
            for row in rows:
                if row.timestamp_ms in seen:
                    raise ValueError(f"Duplicate timestamp for {symbol}: {row.timestamp_ms}")
                seen.add(row.timestamp_ms)
                if previous is not None and row.timestamp_ms - previous != INTERVAL_MS:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = []
                    self._gap_count += 1
                current_segment.append(row)
                previous = row.timestamp_ms
            if current_segment:
                segments.append(current_segment)
            self._segments[symbol] = segments

    @classmethod
    def from_jsonl(cls, path: Path) -> "ResearchDataset":
        bars: list[ResearchBar] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            bars.append(ResearchBar.from_mapping(row))
        return cls(bars)

    @classmethod
    def from_csv(cls, path: Path) -> "ResearchDataset":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return cls(ResearchBar.from_mapping(row) for row in csv.DictReader(fh))

    def symbols(self) -> list[str]:
        return sorted(self.by_symbol)

    def bars(self, symbol: str) -> list[ResearchBar]:
        return list(self.by_symbol.get(symbol.upper(), ()))

    def segments(self, symbol: str) -> list[list[ResearchBar]]:
        return [list(segment) for segment in self._segments.get(symbol.upper(), ())]

    @property
    def gap_count(self) -> int:
        return self._gap_count

    def provenance_summary(self) -> dict[str, Any]:
        revisions: set[tuple[str, str, int]] = set()
        legacy_rows = 0
        for symbol in self.symbols():
            for bar in self.by_symbol[symbol]:
                revisions.add((bar.strategy_revision, bar.config_fingerprint, bar.research_schema_version))
                if bar.strategy_revision == "LEGACY_UNKNOWN" or bar.config_fingerprint == "UNKNOWN" or bar.research_schema_version <= 0:
                    legacy_rows += 1
        return {
            "unique_provenance": len(revisions),
            "provenance": sorted([list(x) for x in revisions]),
            "legacy_rows": legacy_rows,
            "clean": legacy_rows == 0 and len(revisions) == 1,
        }

    def contiguous_suffix(self, symbol: str, cutoff_timestamp_ms: int) -> list[ResearchBar]:
        """Return the latest contiguous segment ending at/before cutoff."""
        segments = self._segments.get(symbol.upper(), ())
        eligible = [seg for seg in segments if seg and seg[0].timestamp_ms <= cutoff_timestamp_ms]
        if not eligible:
            return []
        for segment in reversed(eligible):
            rows = [row for row in segment if row.timestamp_ms <= cutoff_timestamp_ms]
            if rows:
                return rows
        return []

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for symbol in self.symbols():
                for bar in self.by_symbol[symbol]:
                    fh.write(json.dumps(asdict(bar), separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class BacktestConfig:
    entry_mode: str = "next_open"
    entry_delay_bars: int = 1
    max_holding_bars: int = 12
    stop_first_on_ambiguous_bar: bool = True
    commission_rt_pct: float = 0.0010
    slippage_atr_fraction: float = 0.15
    max_slippage_rt_pct: float = 0.0030
    replay_recorded_signals: bool = True

    def __post_init__(self) -> None:
        if self.entry_mode not in {"next_open", "close"}:
            raise ValueError("entry_mode must be next_open or close")
        if self.entry_mode == "close" and self.entry_delay_bars != 0:
            raise ValueError("close entry mode requires entry_delay_bars=0")
        if self.entry_delay_bars < 0 or self.max_holding_bars < 1:
            raise ValueError("Invalid backtest horizon configuration")
        if min(self.commission_rt_pct, self.slippage_atr_fraction, self.max_slippage_rt_pct) < 0:
            raise ValueError("Backtest costs cannot be negative")


@dataclass
class TradeRecord:
    signal_timestamp_ms: int
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    symbol: str
    side: str
    signal_score: float
    entry_price: float
    exit_price: float
    invalidation_price: float
    target_price: float
    stop_hit: bool
    target_hit: bool
    exit_reason: str
    gross_return: float
    round_trip_cost: float
    net_return: float
    holding_bars: int
    z_cvd: float
    z_fund: float
    z_oi: float
    z_micro: float
    z_whale: float
    atr_pct: float
    spread_bps: float


class QuantBacktester:
    """Event-driven, bar-based backtester using only information known at signal time."""

    def __init__(self, signal_engine: QuantSignalEngine, config: BacktestConfig | None = None):
        self.signal_engine = signal_engine
        self.config = config or BacktestConfig()

    def _friction(self, entry_bar: ResearchBar, exit_bar: ResearchBar) -> float:
        """Explicit commission only; spread/slippage are already in executed fill prices."""
        return float(self.config.commission_rt_pct)

    @staticmethod
    def _signal_from_recorded(row: ResearchBar) -> Any | None:
        if not row.recorded_signal:
            return None
        from contracts import SignalEvent
        payload = dict(row.recorded_signal)
        try:
            return SignalEvent(**payload)
        except (TypeError, ValueError, KeyError):
            return None

    def _build_signal(self, rows: Sequence[ResearchBar], index: int):
        current = rows[index]
        if self.config.replay_recorded_signals:
            recorded = self._signal_from_recorded(current)
            if recorded is not None:
                z_values = (recorded.z_cvd_div, recorded.z_fund_trap, recorded.z_delta_oi, recorded.z_micro, recorded.z_whale_sentiment)
                if recorded.signal_type in {"STRONG_LONG", "STRONG_SHORT"}:
                    return recorded, z_values
                return None
            # A clean (versioned) research row must never be silently re-generated.
            # Recomputing it here would turn a recorded-decision replay into a counterfactual backtest.
            if current.research_schema_version > 0 and current.strategy_revision not in {"", "LEGACY_UNKNOWN"}:
                return None
        if index < self.signal_engine.z_history_min_samples:
            return None
        prior = rows[:index]

        def values(name: str) -> list[float]:
            return [float(getattr(row, name)) for row in prior[-self.signal_engine.z_history_min_samples * 3 :]]

        funding_history = values("funding_rate_8h")
        basis_history = values("basis_bps")
        delta_history = values("delta_oi_pct")
        micro_history = [
            r.obi * (1.0 - r.vpin)
            for r in prior[-self.signal_engine.z_history_min_samples * 3 :]
        ]
        cvd_history = values("cvd_divergence_score")
        whale_history = values("whale_divergence_score")
        try:
            z_cvd, z_fund, z_oi, z_micro, z_whale = self.signal_engine.calculate_factor_zscores(
                funding_rate_8h=current.funding_rate_8h,
                basis_spread_bps=current.basis_bps,
                delta_oi_pct=current.delta_oi_pct,
                obi=current.obi,
                vpin=current.vpin,
                cvd_divergence_score=current.cvd_divergence_score,
                whale_divergence_score=current.whale_divergence_score,
                funding_history=funding_history,
                basis_history=basis_history,
                delta_oi_pct_history=delta_history,
                micro_factor_history=micro_history,
                cvd_history=cvd_history,
                whale_history=whale_history,
            )
        except ValueError:
            return None

        active_factors = self.signal_engine.factor_activity(
            funding_history=funding_history,
            basis_history=basis_history,
            delta_oi_pct_history=delta_history,
            micro_factor_history=micro_history,
            whale_history=whale_history,
        )

        signal = self.signal_engine.compute_signal(
            symbol=current.symbol,
            current_price=current.close,
            funding_rate_8h=current.funding_rate_8h,
            basis_spread_bps=current.basis_bps,
            delta_oi=current.delta_oi_pct,
            oi_total=current.open_interest,
            obi=current.obi,
            vpin=current.vpin,
            cvd_divergence_score=current.cvd_divergence_score,
            recent_high=current.recent_high,
            recent_low=current.recent_low,
            z_cvd_override=z_cvd,
            z_fund_override=z_fund,
            z_delta_oi_override=z_oi,
            z_micro_override=z_micro,
            z_whale_override=z_whale,
            z_whale_sentiment=current.whale_divergence_score,
            relative_strength=current.relative_strength,
            sweep_reclaim=current.sweep_reclaim,
            sweep_pattern=current.sweep_pattern,
            atr_pct=current.atr_pct,
            gate_long_status=current.gate_long_status,
            gate_short_status=current.gate_short_status,
            timestamp_ms=current.timestamp_ms,
            decision_timestamp_ms=current.timestamp_ms,
            account_equity=10_000.0,
            whale_history_length=len(whale_history),
            friction_round_trip_pct=self._friction(current, current),
            active_factors=active_factors,
        )
        return signal, (z_cvd, z_fund, z_oi, z_micro, z_whale)


    @staticmethod
    def _gross_return(entry_reference: float, exit_reference: float, side: str) -> float:
        if entry_reference <= 0.0 or exit_reference <= 0.0:
            raise ValueError("Reference prices must be positive")
        if side == "LONG":
            return exit_reference / entry_reference - 1.0
        if side == "SHORT":
            return 1.0 - exit_reference / entry_reference
        raise ValueError(f"Unknown side: {side}")

    def _execution_price(
        self,
        reference: float,
        side: str,
        bar: ResearchBar,
        is_entry: bool,
        *,
        volatility_pct: float | None = None,
    ) -> float:
        """Adverse execution relative to a reference price.

        ``bar`` supplies the spread snapshot. ``volatility_pct`` must come from
        information available no later than the fill decision; callers use the
        prior completed bar for intrabar fills to avoid using the exit/entry bar's
        own completed high/low/close as a slippage input.
        """
        vol = bar.atr_pct if volatility_pct is None else float(volatility_pct)
        if not math.isfinite(vol) or vol < 0.0:
            raise ValueError("volatility_pct must be finite and non-negative")
        half_spread = max(0.0, bar.spread_bps) / 20_000.0
        half_slip = min(vol * self.config.slippage_atr_fraction / 2.0, self.config.max_slippage_rt_pct / 2.0)
        adverse = half_spread + half_slip
        # LONG entry and SHORT exit are buys; LONG exit and SHORT entry are sells.
        is_buy = (side == "LONG" and is_entry) or (side == "SHORT" and not is_entry)
        return reference * (1.0 + adverse) if is_buy else reference * (1.0 - adverse)

    def _run_trade(self, rows: Sequence[ResearchBar], signal_index: int, signal: Any, z_values: tuple[float, ...]) -> TradeRecord | None:
        if signal.signal_type not in {"STRONG_LONG", "STRONG_SHORT"}:
            return None
        side = "LONG" if signal.signal_type == "STRONG_LONG" else "SHORT"
        entry_index = signal_index + self.config.entry_delay_bars
        if entry_index >= len(rows):
            return None
        entry_bar = rows[entry_index]
        entry_ref = entry_bar.open if self.config.entry_mode == "next_open" else rows[signal_index].close
        entry_timestamp_ms = (
            entry_bar.timestamp_ms - (5 * 60 * 1000 - 1)
            if self.config.entry_mode == "next_open"
            else rows[signal_index].timestamp_ms
        )
        # Entry-cost inputs must be known at the entry decision. For a next-open
        # fill this is the fully closed signal bar; for a close fill it is the
        # same signal bar at its close. Do not use the next bar's ATR.
        entry_cost_bar = rows[signal_index]
        entry_px = self._execution_price(
            entry_ref,
            side,
            entry_cost_bar,
            is_entry=True,
            volatility_pct=entry_cost_bar.atr_pct,
        )

        stop = float(signal.invalidation_price)
        target = float(signal.target_price)
        for j in range(entry_index, min(len(rows), entry_index + self.config.max_holding_bars)):
            bar = rows[j]
            stop_hit = bar.low <= stop if side == "LONG" else bar.high >= stop
            target_hit = bar.high >= target if side == "LONG" else bar.low <= target
            if stop_hit or target_hit:
                if stop_hit and target_hit:
                    exit_ref = stop if self.config.stop_first_on_ambiguous_bar else target
                    reason = "STOP_AND_TARGET_SAME_BAR_STOP_FIRST" if self.config.stop_first_on_ambiguous_bar else "STOP_AND_TARGET_SAME_BAR_TARGET_FIRST"
                    stop_won = self.config.stop_first_on_ambiguous_bar
                elif stop_hit:
                    exit_ref = stop
                    reason = "STOP"
                    stop_won = True
                else:
                    exit_ref = target
                    reason = "TARGET"
                    stop_won = False
                # A stop/target can trigger intrabar before ``bar`` is complete.
                # Use the previous completed bar for the slippage estimate; the
                # fill's reference itself remains the stop/target level.
                exit_cost_bar = rows[j - 1] if j > signal_index else rows[signal_index]
                exit_px = self._execution_price(
                    exit_ref,
                    side,
                    exit_cost_bar,
                    is_entry=False,
                    volatility_pct=exit_cost_bar.atr_pct,
                )
                gross = self._gross_return(entry_px, exit_px, side)
                cost = self._friction(entry_bar, bar)
                net = gross - cost
                return TradeRecord(
                    signal_timestamp_ms=rows[signal_index].timestamp_ms,
                    entry_timestamp_ms=entry_timestamp_ms,
                    exit_timestamp_ms=bar.timestamp_ms,
                    symbol=signal.symbol,
                    side=side,
                    signal_score=float(signal.composite_score),
                    entry_price=entry_px,
                    exit_price=exit_px,
                    invalidation_price=stop,
                    target_price=target,
                    stop_hit=stop_hit,
                    target_hit=target_hit,
                    exit_reason=reason,
                    gross_return=gross,
                    round_trip_cost=cost,
                    net_return=net,
                    holding_bars=j - entry_index + 1,
                    z_cvd=z_values[0],
                    z_fund=z_values[1],
                    z_oi=z_values[2],
                    z_micro=z_values[3],
                    z_whale=z_values[4],
                    atr_pct=entry_bar.atr_pct,
                    spread_bps=entry_bar.spread_bps,
                )

        # Time-based exit at the last available bar inside the holding window.
        exit_index = min(len(rows) - 1, entry_index + self.config.max_holding_bars - 1)
        if exit_index <= entry_index:
            return None
        exit_bar = rows[exit_index]
        # At a time-based close the current bar is fully known at the fill, but
        # using the previous completed bar keeps the cost model conservative and
        # prevents any same-bar volatility/lookahead leakage.
        exit_cost_bar = rows[exit_index - 1] if exit_index > signal_index else rows[signal_index]
        exit_px = self._execution_price(
            exit_bar.close,
            side,
            exit_cost_bar,
            is_entry=False,
            volatility_pct=exit_cost_bar.atr_pct,
        )
        gross = self._gross_return(entry_px, exit_px, side)
        cost = self._friction(entry_bar, exit_bar)
        return TradeRecord(
            signal_timestamp_ms=rows[signal_index].timestamp_ms,
            entry_timestamp_ms=entry_timestamp_ms,
            exit_timestamp_ms=exit_bar.timestamp_ms,
            symbol=signal.symbol,
            side=side,
            signal_score=float(signal.composite_score),
            entry_price=entry_px,
            exit_price=exit_px,
            invalidation_price=stop,
            target_price=target,
            stop_hit=False,
            target_hit=False,
            exit_reason="TIME_EXIT",
            gross_return=gross,
            round_trip_cost=cost,
            net_return=gross - cost,
            holding_bars=exit_index - entry_index + 1,
            z_cvd=z_values[0],
            z_fund=z_values[1],
            z_oi=z_values[2],
            z_micro=z_values[3],
            z_whale=z_values[4],
            atr_pct=entry_bar.atr_pct,
            spread_bps=entry_bar.spread_bps,
        )

    def run(
        self,
        dataset: ResearchDataset,
        signal_start_ms: int | None = None,
        signal_end_ms: int | None = None,
        require_exit_within_end: bool = False,
    ) -> list[TradeRecord]:
        trades: list[TradeRecord] = []
        for symbol in dataset.symbols():
            for rows in dataset.segments(symbol):
                for index in range(self.signal_engine.z_history_min_samples, len(rows)):
                    if signal_start_ms is not None and rows[index].timestamp_ms < signal_start_ms:
                        continue
                    if signal_end_ms is not None and rows[index].timestamp_ms > signal_end_ms:
                        continue
                    built = self._build_signal(rows, index)
                    if built is None:
                        continue
                    signal, z_values = built
                    trade = self._run_trade(rows, index, signal, z_values)
                    if trade is not None:
                        if require_exit_within_end and signal_end_ms is not None and trade.exit_timestamp_ms > signal_end_ms:
                            continue
                        trades.append(trade)
        trades.sort(key=lambda x: (x.entry_timestamp_ms, x.symbol))
        return trades


def trade_to_dict(trade: TradeRecord) -> dict[str, Any]:
    return asdict(trade)
