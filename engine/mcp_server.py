"""
Local Model Context Protocol (MCP) server over stdio.

Security/semantic invariants:
- Input arguments are schema-validated before dispatch, preventing TypeError from unknown kwargs.
- Backtests consume actual local CSV/Parquet data; synthetic/random market paths are not exposed as a backtest result.
- The process is stdio-only and must not be exposed as a network service.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Set

import polars as pl

from engine.backtester import QuantBacktester
from engine.risk import DynamicRiskEngine

SIGNALS_STATE_FILE = Path(".signals_latest.json")

logger = logging.getLogger("mcp")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


class MCPServer:
    """JSON-RPC 2.0 MCP server using trusted local stdio transport."""

    def __init__(self) -> None:
        self.risk_engine = DynamicRiskEngine()
        self.backtester = QuantBacktester()
        self.backtest_root = Path(os.getenv("MCP_BACKTEST_ROOT", ".")).resolve()

    def get_screener_signals(self, min_abs_score: float = 0.0) -> Dict[str, Any]:
        if not math.isfinite(float(min_abs_score)) or min_abs_score < 0.0:
            return {"error": "min_abs_score must be finite and >= 0"}
        if not SIGNALS_STATE_FILE.exists():
            return {"error": "No scan results available yet. Run cron_runner.py first."}
        try:
            data = json.loads(SIGNALS_STATE_FILE.read_text(encoding="utf-8"))
            signals = data.get("signals", [])
            filtered = []
            for signal in signals:
                try:
                    score = float(signal["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(score) and abs(score) >= min_abs_score:
                    filtered.append(signal)
            return {"summary": data.get("summary", {}), "count": len(filtered), "signals": filtered}
        except (OSError, ValueError, TypeError) as exc:
            logger.error("signals_read_failed error=%s", exc)
            return {"error": str(exc)}

    def get_synthetic_liquidations(self) -> Dict[str, Any]:
        if not SIGNALS_STATE_FILE.exists():
            return {"liquidations": [], "count": 0}
        try:
            data = json.loads(SIGNALS_STATE_FILE.read_text(encoding="utf-8"))
            liqs = data.get("liquidations", [])
            return {"count": len(liqs), "liquidations": liqs}
        except (OSError, ValueError, TypeError) as exc:
            logger.error("liquidations_read_failed error=%s", exc)
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
        available_liquidity_usdt: float | None = None,
    ) -> Dict[str, Any]:
        if not isinstance(symbol, str) or not symbol.strip():
            return {"error": "symbol must be a non-empty string"}
        if signal_type not in ("STRONG_LONG", "STRONG_SHORT"):
            return {"error": "signal_type must be STRONG_LONG or STRONG_SHORT"}

        try:
            values = [
                float(entry_price),
                float(invalidation_price),
                float(target_price),
                float(account_capital_usdt),
                float(risk_pct),
            ]
            liquidity = None if available_liquidity_usdt is None else float(available_liquidity_usdt)
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid numeric parameters: {exc}"}

        if not all(math.isfinite(x) for x in values):
            return {"error": "All numeric parameters must be finite"}
        if values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] <= 0.0:
            return {"error": "Price and capital values must be positive"}
        if not (0.0 < values[4] <= 100.0):
            return {"error": "risk_pct must be > 0 and <= 100"}
        if liquidity is not None and (not math.isfinite(liquidity) or liquidity <= 0.0):
            return {"error": "available_liquidity_usdt must be finite and > 0 when supplied"}

        engine = DynamicRiskEngine(
            default_account_capital=self.risk_engine.default_account_capital,
            risk_per_trade_pct=values[4],
            max_leverage=self.risk_engine.max_leverage,
            max_liquidity_impact_pct=self.risk_engine.max_liquidity_impact_pct,
        )
        rec = engine.calculate_sizing(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=values[0],
            invalidation_price=values[1],
            target_price=values[2],
            account_capital=values[3],
            available_liquidity_usdt=liquidity,
        )
        if rec is None:
            return {"error": "Invalid risk contract: direction/stop/target/liquidity rejected"}

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
            "liquidity_constraint_applied": rec.liquidity_constraint_applied,
        }

    def _resolve_backtest_path(self, data_path: str) -> Path:
        path = Path(data_path).expanduser().resolve()
        if self.backtest_root != path and self.backtest_root not in path.parents:
            raise ValueError(f"data_path must be under MCP_BACKTEST_ROOT={self.backtest_root}")
        if not path.is_file():
            raise ValueError(f"Backtest data file does not exist: {path}")
        if path.suffix.lower() not in (".csv", ".parquet"):
            raise ValueError("Only .csv and .parquet backtest files are supported")
        return path

    def run_strategy_backtest(
        self,
        data_path: str,
        score_column: str = "composite_score",
        price_column: str = "close",
        open_price_column: str = "open",
        timestamp_column: str = "timestamp_ms",
        long_threshold: float = 75.0,
        short_threshold: float = -75.0,
        holding_bars: int = 6,
        n_trials: int | None = None,
        var_trials_sr: float | None = None,
    ) -> Dict[str, Any]:
        """Run on user-supplied historical bars; no synthetic/random data is permitted."""
        try:
            path = self._resolve_backtest_path(data_path)
            if not all(math.isfinite(float(x)) for x in (long_threshold, short_threshold)):
                raise ValueError("thresholds must be finite")
            if long_threshold <= short_threshold:
                raise ValueError("long_threshold must be greater than short_threshold")
            if holding_bars < 1:
                raise ValueError("holding_bars must be >= 1")
            if n_trials is not None and n_trials < 1:
                raise ValueError("n_trials must be >= 1")
            if path.suffix.lower() == ".parquet":
                df = pl.read_parquet(path)
            else:
                df = pl.read_csv(path)

            metrics = self.backtester.run_backtest(
                df=df,
                score_column=score_column,
                price_column=price_column,
                open_price_column=open_price_column,
                timestamp_column=timestamp_column,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                holding_bars=holding_bars,
                n_trials=n_trials,
                var_trials_sr=var_trials_sr,
            )
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            logger.error("backtest_failed path=%s error=%s", data_path, exc)
            return {"error": str(exc)}

        return {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate_pct": metrics.win_rate_pct,
            "profit_factor": metrics.profit_factor,
            "total_net_pnl_pct": metrics.total_net_pnl_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "annualized_return_pct": metrics.annualized_return_pct,
            "annualized_volatility_pct": metrics.annualized_volatility_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "deflated_sharpe_ratio": metrics.deflated_sharpe_ratio,
            "is_statistically_significant": metrics.is_statistically_significant,
            "n_trials": metrics.n_trials,
            "note": "DSR is zero/not significant when actual trial count is not supplied.",
        }

    @staticmethod
    def _validate_arguments(
        name: str, arguments: Any, allowed: Set[str], required: Set[str] | None = None
    ) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(f"Unknown arguments for {name}: {sorted(unknown)}")
        missing = (required or set()) - set(arguments)
        if missing:
            raise ValueError(f"Missing required arguments for {name}: {sorted(missing)}")
        return arguments

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_screener_signals",
                "description": "Returns latest ranked quant signals.",
                "inputSchema": {"type": "object", "properties": {"min_abs_score": {"type": "number", "minimum": 0.0}}},
            },
            {
                "name": "get_synthetic_liquidations",
                "description": "Returns reconstructed synthetic liquidation events.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "calculate_position_size",
                "description": "Calculates position sizing from validated stop/target and real available liquidity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "signal_type": {"type": "string", "enum": ["STRONG_LONG", "STRONG_SHORT"]},
                        "entry_price": {"type": "number"},
                        "invalidation_price": {"type": "number"},
                        "target_price": {"type": "number"},
                        "account_capital_usdt": {"type": "number"},
                        "risk_pct": {"type": "number"},
                        "available_liquidity_usdt": {"type": "number"},
                    },
                    "required": ["symbol", "signal_type", "entry_price", "invalidation_price", "target_price", "account_capital_usdt", "risk_pct", "available_liquidity_usdt"],
                },
            },
            {
                "name": "run_strategy_backtest",
                "description": "Runs the backtester on actual local CSV/Parquet 5m bars; requires explicit trial metadata for DSR.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data_path": {"type": "string"},
                        "score_column": {"type": "string"},
                        "price_column": {"type": "string"},
                        "open_price_column": {"type": "string"},
                        "timestamp_column": {"type": "string"},
                        "long_threshold": {"type": "number"},
                        "short_threshold": {"type": "number"},
                        "holding_bars": {"type": "integer", "minimum": 1},
                        "n_trials": {"type": "integer", "minimum": 1},
                        "var_trials_sr": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    "required": ["data_path", "n_trials", "timestamp_column"],
                },
            },
        ]

    def handle_request(self, request_json: str) -> str:
        try:
            req = json.loads(request_json)
        except (TypeError, ValueError) as exc:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}})

        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0":
            return json.dumps({"jsonrpc": "2.0", "id": req.get("id") if isinstance(req, dict) else None, "error": {"code": -32600, "message": "Invalid JSON-RPC request"}})

        is_notification = "id" not in req
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        def response(payload: dict[str, Any]) -> str:
            return "" if is_notification else json.dumps(payload)

        try:
            if method == "tools/list":
                result = {"tools": self._tools()}
                return response({"jsonrpc": "2.0", "id": req_id, "result": result})

            if method != "tools/call" or not isinstance(params, dict):
                raise ValueError("Unsupported method or invalid params")

            name = params.get("name")
            arguments = params.get("arguments", {})
            if name == "get_screener_signals":
                args = self._validate_arguments(name, arguments, {"min_abs_score"})
                res = self.get_screener_signals(**args)
            elif name == "get_synthetic_liquidations":
                args = self._validate_arguments(name, arguments, set())
                res = self.get_synthetic_liquidations(**args)
            elif name == "calculate_position_size":
                allowed = {"symbol", "signal_type", "entry_price", "invalidation_price", "target_price", "account_capital_usdt", "risk_pct", "available_liquidity_usdt"}
                args = self._validate_arguments(
                    name, arguments, allowed,
                    required=allowed,
                )
                res = self.calculate_position_size(**args)
            elif name == "run_strategy_backtest":
                allowed = {"data_path", "score_column", "price_column", "open_price_column", "timestamp_column", "long_threshold", "short_threshold", "holding_bars", "n_trials", "var_trials_sr"}
                args = self._validate_arguments(
                    name, arguments, allowed,
                    required={"data_path", "n_trials"},
                )
                res = self.run_strategy_backtest(**args)
            else:
                return response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})

            return response({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2, allow_nan=False)}]}})
        except (TypeError, ValueError) as exc:
            logger.error("mcp_invalid_params method=%s error=%s", method, exc)
            return response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(exc)}})

    def run_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if line:
                response = self.handle_request(line)
                if response:
                    sys.stdout.write(response + "\n")
                    sys.stdout.flush()


if __name__ == "__main__":
    MCPServer().run_stdio()
