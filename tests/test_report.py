import csv
import json
from datetime import datetime, timezone

from screener.metrics import analyze_universe
from screener.models import Check, MarketRow, ScenarioResult
from screener.report import write_outputs


def test_report_writes_market_ranks_calculations_and_csv_limit(tmp_path):
    markets = [
        MarketRow(
            symbol=f"ALT{i}", name=f"ALT{i}", price=10 + i, chg24_pct=1.0,
            market_cap_usd=1_000_000_000, volume24_usd=200_000_000, oi_usd=80_000_000,
            oi_chg24_pct=-1.0, oi_chg24_usd=-1_000_000, oi_chg4h_pct=1.5,
            oi_chg4h_usd=2_500_000, fr_avg_pct=0.001, fr_oiw_pct=0.002,
            pfr_oiw_pct=0.003, liq_total24_usd=100_000, liq_long24_usd=80_000,
            liq_short24_usd=20_000, ls_ratio_1h=1.2, ls_ratio_1d=1.3,
            futures_imbalance_4h=float(i - 10), spot_imbalance_4h=float(i - 8),
        )
        for i in range(12)
    ]
    rows, stats = analyze_universe(markets, {"thresholds": {"strong_flow_quantile": 0.15}})
    results = [
        ScenarioResult(
            symbol=row.symbol, name=row.market.name, scenario="LONG A — test", direction="LONG",
            checks=[Check("F positive", True, "+1", ">0")], quality_flags=[],
            status="OBSERVATION", reason="test only",
        )
        for row in rows[:12]
    ]

    md_path, csv_path, json_path = write_outputs(
        tmp_path, datetime.now(timezone.utc), "PERPETUAL", rows, stats, results, []
    )

    report = md_path.read_text(encoding="utf-8")
    assert "Ранги потоков по всей таблице" in report
    assert "Futures Buy" in report
    assert "RelStrength" in report
    assert "Полные расчёты по каждой монете" in report
    assert "FR−FR OI_W" in report
    assert "OI, ликвидации относительно падения OI" in report
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 10
    assert "версия_параметров" in csv_rows[0]
    assert {row["версия_параметров"] for row in csv_rows} == {"P1"}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["application_version"] == "1.0.1"
    assert payload["parameter_version"] == "P1"
    assert payload["p1_backtested"] is False
    assert len(payload["universe_calculations"]) == len(markets)
    assert payload["universe_calculations"][0]["futures_buy_share_pct"] is not None


def test_no_signal_csv_contains_explicit_no_signal_row(tmp_path):
    markets = [MarketRow(symbol="BTCUSDT_PERP", name="Bitcoin", futures_imbalance_4h=1, spot_imbalance_4h=2)]
    rows, stats = analyze_universe(markets, {"thresholds": {}})
    _, csv_path, _ = write_outputs(
        tmp_path, datetime.now(timezone.utc), "PERPETUAL", rows, stats, [], []
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["symbol"] == "нет сигнала"
    assert csv_rows[0]["версия_параметров"] == "P1"
