"""
Rich Terminal Dashboard and Live Quant Monitor.
Visualizes market regime, BTC correlation, institutional filters,
top opportunities, and synthetic liquidations.
Exports latest scan results to JSON for external APIs/frontends.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import msgspec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contracts import SignalEvent, SyntheticLiquidation
from engine.screener import ScreenerResult

console = Console()


def render_dashboard(
    signals: List[SignalEvent],
    synthetic_liqs: List[SyntheticLiquidation],
    summary: ScreenerResult,
) -> None:
    """Renders a rich, colored terminal dashboard of the 5-minute quant scan."""
    now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(summary.timestamp_ms / 1000))

    # Calculate Market Sentiment
    positive_scores = sum(1 for s in signals if s.composite_score > 15.0)
    negative_scores = sum(1 for s in signals if s.composite_score < -15.0)
    neutral_scores = len(signals) - positive_scores - negative_scores

    if positive_scores > negative_scores * 1.5:
        regime_text = Text("🟢 BULLISH ABSORPTION REGIME", style="bold green")
    elif negative_scores > positive_scores * 1.5:
        regime_text = Text("🔴 BEARISH SUPPLY REGIME", style="bold red")
    else:
        regime_text = Text("⚖️ BALANCED / MIXED ROTATION", style="bold yellow")

    btc_status_style = "bold green" if summary.btc_regime == "IMPULSE_PUMP" else ("bold red" if summary.btc_regime == "IMPULSE_DUMP" else "bold yellow")
    btc_regime_text = Text(f"{summary.btc_regime} ({summary.btc_change_5m_pct:+.2f}% 5m)", style=btc_status_style)

    # Header Panel
    header_content = (
        f"⏱️ [bold white]Timestamp:[/bold white] {now_utc}  |  "
        f"⚡ [bold white]Scan Speed:[/bold white] [cyan]{summary.duration_sec:.2f}s[/cyan]  |  "
        f"🌐 [bold white]Universe:[/bold white] [bold cyan]{summary.total_scanned}[/bold cyan] pairs\n"
        f"👑 [bold white]BTC Macro Gate:[/bold white] {btc_regime_text}  |  "
        f"🧭 [bold white]Alt Structure:[/bold white] {regime_text}\n"
        f"📊 [bold white]Signals:[/bold white] [green]LONG: {summary.strong_longs_count}[/green] | "
        f"[red]SHORT: {summary.strong_shorts_count}[/red] | "
        f"[yellow]NEUTRAL: {neutral_scores}[/yellow]  |  "
        f"🚨 [bold white]Synthetic Liqs:[/bold white] [bold magenta]{summary.synthetic_liqs_count}[/bold magenta]"
    )
    console.print(Panel(header_content, title="[bold cyan]INSTITUTIONAL QUANT DERIVATIVES COCKPIT[/bold cyan]", border_style="cyan"))

    # Top 5 Long Candidates Table
    long_table = Table(
        title="[bold green]🟢 TOP LONG CANDIDATES (Negative Funding / Whales Accumulation / Limit Absorption)[/bold green]",
        border_style="green",
        header_style="bold green",
        expand=True,
    )
    long_table.add_column("Symbol", justify="left", style="bold white")
    long_table.add_column("Price", justify="right")
    long_table.add_column("Score", justify="right", style="bold green")
    long_table.add_column("Fund 8h", justify="right")
    long_table.add_column("Basis", justify="right")
    long_table.add_column("OBI", justify="right")
    long_table.add_column("RS (BTC)", justify="right", style="cyan")
    long_table.add_column("Whale Z", justify="right", style="bold magenta")
    long_table.add_column("Spring", justify="center", style="bold yellow")
    long_table.add_column("Gate", justify="center")
    long_table.add_column("Status", justify="center")

    top_longs = [s for s in signals if s.composite_score > 0][:5]
    for s in top_longs:
        status = "[bold green]STRONG LONG[/bold green]" if s.signal_type == "STRONG_LONG" else "[dim]LEAN LONG[/dim]"
        spring_badge = "⚡ YES" if s.sweep_reclaim else "—"
        gate_badge = "[green]PASS[/green]" if s.gate_status == "PASSED" else f"[red]{s.gate_status[:10]}[/red]"
        long_table.add_row(
            s.symbol,
            f"${s.price:,.4f}",
            f"{s.composite_score:+.1f}",
            f"{s.funding_8h*100:+.4f}%",
            f"{s.basis_bps:+.1f}",
            f"{s.obi:+.3f}",
            f"{s.relative_strength:+.2f}%",
            f"{s.z_whale_sentiment:+.2f}",
            spring_badge,
            gate_badge,
            status,
        )
    console.print(long_table)

    # Top 5 Short Candidates Table
    short_table = Table(
        title="[bold red]🔴 TOP SHORT CANDIDATES (Overheated Funding / Retail Traps / Supply Walls)[/bold red]",
        border_style="red",
        header_style="bold red",
        expand=True,
    )
    short_table.add_column("Symbol", justify="left", style="bold white")
    short_table.add_column("Price", justify="right")
    short_table.add_column("Score", justify="right", style="bold red")
    short_table.add_column("Fund 8h", justify="right")
    short_table.add_column("Basis", justify="right")
    short_table.add_column("OBI", justify="right")
    short_table.add_column("RS (BTC)", justify="right", style="cyan")
    short_table.add_column("Whale Z", justify="right", style="bold magenta")
    short_table.add_column("Upthrust", justify="center", style="bold yellow")
    short_table.add_column("Gate", justify="center")
    short_table.add_column("Status", justify="center")

    top_shorts = [s for s in signals if s.composite_score < 0][::-1][:5]
    for s in top_shorts:
        status = "[bold red]STRONG SHORT[/bold red]" if s.signal_type == "STRONG_SHORT" else "[dim]LEAN SHORT[/dim]"
        upthrust_badge = "⚡ YES" if s.sweep_reclaim else "—"
        gate_badge = "[green]PASS[/green]" if s.gate_status == "PASSED" else f"[red]{s.gate_status[:10]}[/red]"
        short_table.add_row(
            s.symbol,
            f"${s.price:,.4f}",
            f"{s.composite_score:+.1f}",
            f"{s.funding_8h*100:+.4f}%",
            f"{s.basis_bps:+.1f}",
            f"{s.obi:+.3f}",
            f"{s.relative_strength:+.2f}%",
            f"{s.z_whale_sentiment:+.2f}",
            upthrust_badge,
            gate_badge,
            status,
        )
    console.print(short_table)

    # Synthetic Liquidations Table
    if synthetic_liqs:
        liq_table = Table(
            title="[bold yellow]⚡ RECONSTRUCTED SYNTHETIC LIQUIDATIONS (Hidden Binance Cascades)[/bold yellow]",
            border_style="yellow",
            header_style="bold yellow",
            expand=True,
        )
        liq_table.add_column("Symbol", style="bold white")
        liq_table.add_column("Type", justify="center")
        liq_table.add_column("Price", justify="right")
        liq_table.add_column("ΔOI (5m)", justify="right", style="bold red")
        liq_table.add_column("Taker Vol", justify="right")
        liq_table.add_column("Anomaly Ratio", justify="center", style="bold magenta")
        liq_table.add_column("Estimated Liq Vol", justify="right", style="cyan")

        for lq in synthetic_liqs[:8]:
            side_badge = "[red]LONG LIQ (DUMP)[/red]" if lq.side == "LONG_LIQUIDATION" else "[green]SHORT SQUEEZE[/green]"
            liq_table.add_row(
                lq.symbol,
                side_badge,
                f"${lq.price:,.4f}",
                f"{lq.delta_oi:,.1f}",
                f"{lq.taker_volume:,.1f}",
                f"{lq.anomaly_ratio:.2f}x",
                f"{lq.estimated_liquidation_volume:,.1f}",
            )
        console.print(liq_table)


def export_json(
    signals: List[SignalEvent],
    synthetic_liqs: List[SyntheticLiquidation],
    summary: ScreenerResult,
    target_path: Path = Path(".signals_latest.json"),
) -> None:
    """Exports structured results for external frontends or REST API consumption."""
    payload = {
        "summary": {
            "timestamp_ms": summary.timestamp_ms,
            "duration_sec": summary.duration_sec,
            "total_scanned": summary.total_scanned,
            "strong_longs": summary.strong_longs_count,
            "strong_shorts": summary.strong_shorts_count,
            "synthetic_liqs": summary.synthetic_liqs_count,
            "btc_regime": summary.btc_regime,
            "btc_change_5m_pct": summary.btc_change_5m_pct,
        },
        "signals": [
            {
                "symbol": s.symbol,
                "type": s.signal_type,
                "score": s.composite_score,
                "price": s.price,
                "funding_8h": s.funding_8h,
                "basis_bps": s.basis_bps,
                "obi": s.obi,
                "vpin": s.vpin,
                "z_cvd_div": s.z_cvd_div,
                "z_fund_trap": s.z_fund_trap,
                "z_whale_sentiment": s.z_whale_sentiment,
                "relative_strength": s.relative_strength,
                "sweep_reclaim": s.sweep_reclaim,
                "gate_status": s.gate_status,
                "invalidation_price": s.invalidation_price,
                "target_price": s.target_price,
                "risk_reward_ratio": s.risk_reward_ratio,
            }
            for s in signals
        ],
        "liquidations": [
            {
                "symbol": lq.symbol,
                "side": lq.side,
                "price": lq.price,
                "delta_oi": lq.delta_oi,
                "taker_volume": lq.taker_volume,
                "anomaly_ratio": lq.anomaly_ratio,
                "estimated_volume": lq.estimated_liquidation_volume,
            }
            for lq in synthetic_liqs
        ],
    }
    raw = msgspec.json.encode(payload)
    target_path.write_bytes(raw)
