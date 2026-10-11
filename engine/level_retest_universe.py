"""Pure selection helper for the standalone 12h level-retest universe."""
from __future__ import annotations

import math
from typing import Any


def select_usdt_m_universe(
    tickers: dict[str, dict[str, Any]],
    *,
    min_24h_volume_usdt: float = 0.0,
    top_n_symbols: int = 0,
) -> list[tuple[str, float, float]]:
    """Select USDT-margined Futures ticker symbols.

    A zero top-N means no symbol-count cap. A zero volume floor includes
    low-volume and zero-volume tickers when price/market data are valid.
    The caller receives (symbol, quote_volume, last_price), sorted by volume.
    """
    if (
        not math.isfinite(min_24h_volume_usdt)
        or min_24h_volume_usdt < 0.0
        or top_n_symbols < 0
    ):
        raise ValueError("Invalid universe selection configuration")

    universe: list[tuple[str, float, float]] = []
    for symbol, ticker in tickers.items():
        # This endpoint is USDⓈ-M Futures; symbol suffix keeps this mode on
        # USDT-quoted perpetual-style names, excluding e.g. USDC/dated contracts.
        if not isinstance(symbol, str) or not symbol.endswith("USDT") or not isinstance(ticker, dict):
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", "nan"))
            price = float(ticker.get("lastPrice", "nan"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(quote_volume)
            or not math.isfinite(price)
            or quote_volume < min_24h_volume_usdt
            or price <= 0.0
        ):
            continue
        universe.append((symbol, quote_volume, price))

    universe.sort(key=lambda item: item[1], reverse=True)
    if top_n_symbols > 0:
        universe = universe[:top_n_symbols]
    return universe
