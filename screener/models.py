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
