from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MarketRow:
    symbol: str
    name: str
    price: float | None = None
    chg24_pct: float | None = None
    market_cap_usd: float | None = None
    fdv_usd: float | None = None
    volume24_usd: float | None = None
    oi_usd: float | None = None
    oi_chg24_pct: float | None = None
    oi_chg24_usd: float | None = None
    oi_chg4h_pct: float | None = None
    oi_chg4h_usd: float | None = None
    oi_share_pct: float | None = None
    oi_vol_ratio: float | None = None
    oi_mktcap_ratio: float | None = None
    vol_mktcap_ratio: float | None = None
    fr_avg_pct: float | None = None
    pfr_avg_pct: float | None = None
    fr_oiw_pct: float | None = None
    pfr_oiw_pct: float | None = None
    liq_total24_usd: float | None = None
    liq_short24_usd: float | None = None
    liq_long24_usd: float | None = None
    liq_vol_ratio: float | None = None
    liq_oi_ratio: float | None = None
    ls_ratio_1h: float | None = None
    long_accounts_1h_pct: float | None = None
    short_accounts_1h_pct: float | None = None
    ls_ratio_1d: float | None = None
    long_accounts_1d_pct: float | None = None
    short_accounts_1d_pct: float | None = None
    futures_imbalance_4h: float | None = None
    spot_imbalance_4h: float | None = None
    btc_corr7d: float | None = None
    source_symbol: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None


@dataclass(slots=True)
class Check:
    name: str
    passed: bool | None
    actual: str
    rule: str


@dataclass(slots=True)
class ScenarioResult:
    symbol: str
    name: str
    scenario: str
    direction: str
    checks: list[Check]
    quality_flags: list[str]
    status: str
    reason: str = ""
    technical: dict[str, Any] = field(default_factory=dict)
    reason_code: str = ""
    diagnostic_status: str = ""
    failed_checks: list[dict[str, Any]] = field(default_factory=list)
    unknown_checks: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        """Attach durable diagnostics without changing legacy scenario status semantics."""
        self.failed_checks = [
            {"name": check.name, "passed": False, "actual": check.actual, "rule": check.rule}
            for check in self.checks if check.passed is False
        ]
        self.unknown_checks = [
            {"name": check.name, "passed": None, "actual": check.actual, "rule": check.rule}
            for check in self.checks if check.passed is None
        ]

        if self.failed_checks:
            self.diagnostic_status = "FAILED"
        elif self.unknown_checks:
            self.diagnostic_status = "UNKNOWN"
        elif self.checks:
            self.diagnostic_status = "PASS"
        else:
            self.diagnostic_status = "NO_CHECKS"

        if len(self.failed_checks) == 1:
            failed_name = self.failed_checks[0]["name"].strip().lower()
            reason_codes = {
                "price change 24h": "PRICE_CHANGE_24H_OUT_OF_RANGE",
                "strong positive spot": "SPOT_FLOW_CONFIRMATION_FAILED",
                "spot positive": "SPOT_FLOW_CONFIRMATION_FAILED",
                "s negative": "SPOT_FLOW_CONFIRMATION_FAILED",
                "f/s positive; at least one strong": "FLOW_CONFIRMATION_FAILED",
                "f/s negative; at least one strong": "FLOW_CONFIRMATION_FAILED",
                "quality flags": "QUALITY_GATE_FAILED",
                "oi 4h growth": "OI_4H_GROWTH_OUT_OF_RANGE",
                "oi 24h decline material": "OI_24H_DECLINE_NOT_MATERIAL",
            }
            inferred_reason_code = reason_codes.get(failed_name, "CHECK_FAILED")
        elif len(self.failed_checks) > 1:
            inferred_reason_code = "MULTIPLE_CHECKS_FAILED"
        elif self.unknown_checks:
            inferred_reason_code = "REQUIRED_CHECKS_UNKNOWN"
        elif not self.checks:
            inferred_reason_code = "NO_CHECKS_AVAILABLE"
        else:
            inferred_reason_code = "SCENARIO_STATUS_" + (self.status or "UNKNOWN").upper()

        if not self.reason_code:
            self.reason_code = inferred_reason_code

        failed_text = [
            f"{item['name']} (actual: {item['actual']}; rule: {item['rule']})"
            for item in self.failed_checks
        ]
        unknown_text = [
            f"{item['name']} (actual: {item['actual']}; required: {item['rule']})"
            for item in self.unknown_checks
        ]
        parts: list[str] = []
        if failed_text:
            parts.append("Failed checks: " + "; ".join(failed_text))
        if unknown_text:
            parts.append("Unknown checks: " + "; ".join(unknown_text))

        if self.status == "NOT_MET":
            # The public status remains NOT_MET for backwards compatibility.
            # diagnostic_status distinguishes a known failure from an unknown-only result.
            if parts:
                self.summary = ". ".join(parts) + "."
                self.reason = self.summary
            elif not self.reason:
                self.reason = "No scenario predicate passed or failed decisively; inspect the scenario-level gate."
                self.summary = self.reason
            else:
                self.summary = self.reason
        elif self.reason:
            self.summary = self.reason
        elif parts:
            self.summary = ". ".join(parts) + "."
            self.reason = self.summary
        else:
            self.summary = f"Scenario status: {self.status}."

    @property
    def passed_count(self) -> int:
        return sum(c.passed is True for c in self.checks)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def unknown_count(self) -> int:
        return sum(c.passed is None for c in self.checks)

    @property
    def near_candidate(self) -> bool:
        if self.status.startswith("BLOCKED_") or self.quality_flags or self.total_count <= 0:
            return False
        return self.passed_count >= max(0, self.total_count - 1)

    @property
    def complete(self) -> bool:
        return self.total_count > 0 and self.passed_count == self.total_count and self.unknown_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "scenario": self.scenario,
            "direction": self.direction,
            "checks": [asdict(c) for c in self.checks],
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "unknown_count": self.unknown_count,
            "quality_flags": list(self.quality_flags),
            "status": self.status,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "diagnostic_status": self.diagnostic_status,
            "failed_checks": list(self.failed_checks),
            "unknown_checks": list(self.unknown_checks),
            "summary": self.summary,
            "technical": self.technical,
        }


@dataclass(slots=True)
class MarketStats:
    row_count: int
    f_count: int
    s_count: int
    median_f: float | None
    median_s: float | None
    median_oi24_pct: float | None
    median_oi4h_pct: float | None
    median_oi24_usd: float | None
    median_oi4h_usd: float | None
    f_negative_count: int
    s_negative_count: int
    oi24_negative_count: int
    oi24_count: int
    price_positive_count: int
    price_negative_count: int
    price_count: int
    btc_chg24_pct: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
