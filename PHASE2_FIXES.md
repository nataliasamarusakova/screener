# Phase 2 / Phase 3 — Quant Screener Fixes

Дата: 2026-09-19

## Scope

Исправлены root causes из аудита без изменения публичных `msgspec.Struct` полей в середине схемы: новые поля добавлены только в конец и имеют defaults для backward compatibility.

## Закрытые critical findings

| ID | Finding | Status |
|---|---|---|
| C1 | Silent substitution missing OI/funding/orderbook/trades/sentiment | **FULL** — missing/stale/non-finite inputs are now explicit failures; no numeric neutral fallbacks on the production signal path. |
| C2 | Fake 5m CVD/VPIN based on latest N trades | **PARTIAL** — CVD uses closed 5m OHLC/taker-buy volume and VPIN uses an explicit 5m trade window. A 1000-trade saturation is fail-closed rather than paginated, so correctness is preserved but availability for extremely liquid 5m windows is incomplete. |
| C3 | Wyckoff sweep used 24h extremes | **FULL** — sweep now consumes prior closed 5m high/low structure and preserves detector direction end-to-end. |
| C4 | Risk engine silently rewrote invalid stop/target | **FULL** — invalid directional risk contracts are rejected. |
| C5 | Fees omitted from backtest PnL | **FULL** — entry and exit fees are included explicitly in PnL. |
| C6 | Sentiment API call without required key / fallback to zero | **FULL** — authenticated request path required; missing sentiment fails closed. |
| C7 | Next-bar execution silently used close | **FULL** — backtester requires `open` and executes at next-bar open. |
| C8 | DSR kurtosis/trial metadata invalid | **FULL** — raw kurtosis convention is explicit; trial count and trial-variance are required metadata instead of hardcoded values. |

Итого: **7 critical полностью закрыто, 1 partial, 0 deferred**.

## Important fixes

- Pseudo-z-scores заменены на empirical z-scores по PIT history с явным warm-up.
- Fixed `beta_alt_btc=1.6` больше не используется в production path; для alt symbols требуется rolling beta из 5m истории.
- `nextFundingTime=0`/missing теперь `UNKNOWN_FUNDING_TIME`, не `PASSED`.
- `NaN`/`Inf` и отрицательные market-data values теперь fail closed.
- Per-symbol exceptions логируются и учитываются в scan health.
- MCP arguments валидируются; unknown kwargs и missing required arguments возвращают JSON-RPC `-32602`.
- MCP backtest больше не использует synthetic/random prices: нужен реальный `data_path` под `MCP_BACKTEST_ROOT` и явный `n_trials`.
- Request weight учитывается центральным limiter'ом; 429/418 переводят клиент в cooldown вместо blind retry.
- State persistence: temp file + flush + fsync + atomic replace + directory fsync.
- GitHub Actions больше не сохраняет checkout credentials; scheduled runs отменяют устаревший запуск.

## Намеренно не скрытые ограничения

1. **VPIN high-volume window:** если внутри 5m окна Binance отдаёт 1000 aggregate trades и окно выглядит truncated, фактор не вычисляется. Нельзя подменять это `limit=N` без анализа request-weight budget; следующий этап — pagination или persistent trade stream.
2. **Execution-cost calibration:** spread/latency/adverse-selection constants всё ещё heuristic. Код не притворяется empirically calibrated; нужен execution dataset.
3. **Sweep/ATR threshold calibration:** значения strategy-level thresholds сохраняются, но явно помечены TODO на OOS calibration.
4. **GitHub repository security settings:** branch protection/rulesets невозможно проверить из ZIP; workflow-level credentials hardening сделан, repo settings требуют отдельной проверки.
5. **Full pytest:** в текущем execution environment отсутствуют `polars` и `msgspec`, а внешний package installation недоступен. Поэтому collection полного suite здесь невозможен. Syntax/static verification выполнена успешно.

## Regression tests added/updated

Проверяются конкретные значения/структуры для:

- true time-windowed aggTrades;
- explicit 5m sweep direction;
- exact risk sizing and invalid input rejection;
- unknown funding fail-closed;
- NaN rejection;
- empirical z-score math;
- next-open execution;
- fee-inclusive PnL;
- DSR metadata requirement and raw kurtosis;
- sentiment PIT selection and authentication;
- MCP unknown/missing kwargs;
- backward-compatible `MarketStateSnapshot` decode.

## Verification

### Static/syntax

- `python -m compileall -q .` — PASS, 20 Python files.
- Engine bare `print()` scan — PASS; no bare `print()` calls under `engine/`.
- Targeted source-contract checks — PASS for true 5m windowing, fail-closed fallbacks, 5m sweep, risk validation, fee-inclusive PnL, DSR metadata, rolling beta, MCP required args, atomic durable state write, workflow checkout credential handling.

### Full tests

Expected command once CI dependencies are available:

```bash
pip install -r requirements.txt
python -m pytest -q
```

Expected key numeric assertions include:

- VPIN fixture: `0.45`;
- liquidation ratio fixture: `4.0` and estimated volume `800.0`;
- empirical z-score fixture: exact `1.0` for the constructed sample;
- risk sizing fixture: `200.0` base units before liquidity cap;
- capped liquidity fixture: `100.0` base units;
- backtest fixture: next-open entry plus both-side fees, net PnL approximately `0.86%`;
- no DSR without actual trial metadata.

## Production decision today

Не подключать эти scores к automated order execution до полного CI run и replay/testnet validation. Alert-only consumer допустим только при явном signal-health monitoring и после warm-up.

При fresh state после deploy signal path остаётся cold-started: требуется минимум 24 предыдущих валидных 5m observations для empirical z-scores; после time-series gap history также сбрасывается и прогревается заново. Это сознательный fail-closed behaviour.

## Remaining P0

1. Реализовать bounded pagination / persistent trade capture для VPIN в 5m окнах, не нарушая Binance request-weight budget.
2. Провести replay на реальном 5m dataset с injected API failures, gaps, stale funding timestamps и crossed books, чтобы подтвердить end-to-end fail-closed semantics.
