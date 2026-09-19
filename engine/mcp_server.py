"""
Model Context Protocol (MCP) Server & External AI Agent Interface.
Exposes Quantitative Analytics, Signals, Risk Calculations, and Backtests
to external AI Agents via standard JSON-RPC over stdio.
Zero-Docker, Pure Python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import msgspec
import numpy as np
import polars as pl

from engine.backtester import QuantBacktester
from engine.risk import DynamicRiskEngine

SIGNALS_STATE_FILE = Path(".signals_latest.json")
MARKET_STATE_FILE = Path(".market_state.bin")


class MCPServer:
    """
    Standard Model Context Protocol (MCP) server handling JSON-RPC requests over stdio.
    """

    def __init__(self) -> None:
        self.risk_engine = DynamicRiskEngine()
        self.backtester = QuantBacktester()

    def get_screener_signals(self, min_abs_score: float = 0.0) -> Dict[str, Any]:
        """Tool: Returns latest ranked signals filtered by minimum absolute score."""
        if not SIGNALS_STATE_FILE.exists():
            return {"error": "No scan results available yet. Run cron_runner.py first."}

        try:
            data = json.loads(SIGNALS_STATE_FILE.read_text(encoding="utf-8"))
            signals = data.get("signals", [])
            filtered = [s for s in signals if abs(s.get("score", 0)) >= min_abs_score]
            return {
                "summary": data.get("summary", {}),
                "count": len(filtered),
                "signals": filtered,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_synthetic_liquidations(self) -> Dict[str, Any]:
        """Tool: Returns detected hidden Binance synthetic liquidations."""
        if not SIGNALS_STATE_FILE.exists():
            return {"liquidations": [], "count": 0}

        try:
            data = json.loads(SIGNALS_STATE_FILE.read_text(encoding="utf-8"))
            liqs = data.get("liquidations", [])
            return {"count": len(liqs), "liquidations": liqs}
        except Exception as exc:
            return {"error": str(exc)}

    def calculate_position_size(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        invalidation_price: float,
        target_price: float,
        account_capital_usdt: float = 10000.0,
        risk_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """Tool: Computes mathematically sound position sizing based on invalidation distance."""
        self.risk_engine.risk_per_trade_pct = risk_pct
        rec = self.risk_engine.calculate_sizing(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            invalidation_price=invalidation_price,
            target_price=target_price,
            account_capital=account_capital_usdt,
        )
        if not rec:
            return {"error": "Failed to calculate position sizing. Check price parameters."}

        return {
            "symbol": rec.symbol,
            "side": rec.side,
            "entry_price": rec.entry_price,
            "invalidation_price": rec.invalidation_price,
            "target_price": rec.target_price,
            "stop_distance_pct": rec.stop_distance_pct,
            "target_distance_pct": rec.target_distance_pct,
            "risk_reward_ratio": rec.risk_reward_ratio,
            "recommended_quantity": rec.recommended_quantity,
            "recommended_notional_usdt": rec.recommended_notional_usdt,
            "effective_leverage": rec.effective_leverage,
            "capital_at_risk_usdt": rec.risk_per_trade_usdt,
        }

    def run_strategy_backtest(
        self,
        long_threshold: float = 75.0,
        short_threshold: float = -75.0,
        holding_bars: int = 6,
    ) -> Dict[str, Any]:
        """Tool: Runs Walk-Forward Backtester with Deflated Sharpe Ratio on synthetic sample."""
        np.random.seed(42)
        n = 1000
        # Synthetic random walk with momentum and mean-reverting scores
        prices = 60000.0 + np.cumsum(np.random.randn(n) * 50.0)
        scores = np.random.randn(n) * 45.0

        df = pl.DataFrame({
            "close": prices,
            "composite_score": scores,
        })

        metrics = self.backtester.run_backtest(
            df=df,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            holding_bars=holding_bars,
        )

        return {
            "total_trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "profit_factor": metrics.profit_factor,
            "total_net_pnl_pct": metrics.total_net_pnl_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "deflated_sharpe_ratio": metrics.deflated_sharpe_ratio,
            "is_statistically_significant": metrics.is_statistically_significant,
        }

    def handle_request(self, request_json: str) -> str:
        """Processes a single JSON-RPC 2.0 MCP request."""
        try:
            req = json.loads(request_json)
        except Exception as exc:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}})

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "tools/list":
            tools = [
                {
                    "name": "get_screener_signals",
                    "description": "Returns latest ranked quant signals for 100+ crypto futures.",
                    "parameters": {"type": "object", "properties": {"min_abs_score": {"type": "number"}}},
                },
                {
                    "name": "get_synthetic_liquidations",
                    "description": "Returns reconstructed hidden Binance liquidation cascades.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "calculate_position_size",
                    "description": "Calculates position sizing and leverage based on structural invalidation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "signal_type": {"type": "string"},
                            "entry_price": {"type": "number"},
                            "invalidation_price": {"type": "number"},
                            "target_price": {"type": "number"},
                            "account_capital_usdt": {"type": "number"},
                        },
                        "required": ["symbol", "signal_type", "entry_price", "invalidation_price", "target_price"],
                    },
                },
                {
                    "name": "run_strategy_backtest",
                    "description": "Runs Walk-Forward Backtester with Deflated Sharpe Ratio (DSR).",
                    "parameters": {"type": "object", "properties": {"long_threshold": {"type": "number"}}},
                },
            ]
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})

        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})

            if name == "get_screener_signals":
                res = self.get_screener_signals(min_abs_score=float(arguments.get("min_abs_score", 0.0)))
            elif name == "get_synthetic_liquidations":
                res = self.get_synthetic_liquidations()
            elif name == "calculate_position_size":
                res = self.calculate_position_size(**arguments)
            elif name == "run_strategy_backtest":
                res = self.run_strategy_backtest(**arguments)
            else:
                return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})

            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}})

        return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})

    def run_stdio(self) -> None:
        """Runs the MCP server loop over standard input/output."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_request(line)
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    server = MCPServer()
    server.run_stdio()
