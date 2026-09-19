# FINAL FULL PROJECT AUDIT

Дата: 2026-09-19

Проверен итоговый проект `screener-main` после предыдущего Phase-2 patch.

## Scope

Проверены:
- все 20 Python-файлов проекта;
- `tests/test_quant_engine.py`;
- GitHub Actions workflow;
- `requirements.txt`, `pytest.ini`, contracts/state schema;
- REST/WS data flow, FSM, financial math, risk, backtest, MCP, Telegram persistence.

Критерий включения в список ошибок: только воспроизводимый дефект, противоречие математическому/протокольному контракту или подтверждённое fail-open/fail-silent поведение.

Не исправлялись эвристики только потому, что их параметры можно было бы калибровать иначе.

## Исправленные ошибки этого полного повторного аудита

### P0/P1

1. `engine/backtester.py:233-247`
   `bar_returns` расходились с фактическим execution PnL: использовались reference prices вместо executed entry/exit prices. Из-за этого equity curve, PnL и Sharpe могли не соответствовать `_net_return_long/_short`.
   Исправлено: per-bar decomposition теперь точно суммируется в execution-aware net trade return.

2. `engine/liquidations.py:59`
   При `taker_buy_vol=taker_sell_vol=0` использовался искусственный denominator `1e-6`, создавая бесконечно большой anomaly ratio.
   Исправлено: zero visible volume = неизвестный denominator => событие отклоняется; также введена finite/non-negative validation.

3. `engine/execution_cost.py:69-92`
   Zero spread превращался в искусственные `0.5 bps`, а invalid reference/spread/latency/config values не валидировались.
   Исправлено: zero spread остаётся zero; invalid input rejected.

4. `engine/telegram.py:45-78`
   Alert deduplication cache записывался напрямую и ошибки чтения/записи проглатывались.
   Исправлено: atomic temp+fsync+replace; ошибки логируются.

5. `engine/mcp_server.py:289-342`
   JSON-RPC notifications получали response, что нарушает semantics JSON-RPC 2.0.
   Исправлено: валидные requests без `id` выполняются без ответа; stdio не пишет пустой response.

6. `binance_ingestion.py:502-540`
   После WS sequence gap FSM переходил в `DISCONNECTED`, но новый snapshot task автоматически не создавался.
   Исправлено: per-symbol resync task; gap -> BUFFERING -> snapshot synchronization.

7. `binance_ingestion.py:570-624`
   Malformed WebSocket `aggTrade`/`markPriceUpdate` поля через `.get(..., 0)` превращались в валидоподобные zero values.
   Исправлено: required fields и finite/positive constraints проверяются до создания normalized contracts.

8. `engine/screener.py:181-253`, `binance_ingestion.py:699-714`
   Funding normalization всегда считалась как 8h, хотя Binance имеет symbols с изменённым `fundingIntervalHours`.
   Исправлено: `fundingInfo` загружается отдельно; для adjusted symbols используется `(1 + raw) ** (8 / interval_h) - 1`; при недоступности metadata scan fail-closed.

9. `engine/sentiment.py:124-132`
   Retail/Top-Trader/Taker observations могли быть взяты из разных 5m timestamps.
   Исправлено: все три timestamp должны совпадать и находиться внутри PIT window.

10. `engine/signals.py:166-186`
    `NaN/Inf` в Z-score overrides не проверялись; `_clip()` мог преобразовать `NaN` в числовой крайний score.
    Исправлено: overrides включены в finite-input contract.

11. `engine/signals.py:39-50`
    Non-finite значения в историческом распределении silently удалялись, искусственно уменьшая effective sample size.
    Исправлено: history с NaN/Inf отвергается.

12. `engine/funding_filter.py:35-55`
    Non-finite funding мог пройти gate как `PASSED`, потому что comparisons с NaN были false.
    Исправлено: invalid funding -> оба направления blocked, `INVALID_FUNDING_RATE`.

13. `binance_ingestion.py:125-170`
    Binance order-book first-event condition использовала `U <= lastUpdateId <= u`; контракт требует coverage следующего update: `U <= lastUpdateId + 1 <= u`.
    Исправлено, regression test обновлён.

14. `binance_ingestion.py:198-214`
    Negative/non-finite WebSocket depth levels могли попасть во внутреннюю книгу.
    Исправлено: invalid price/quantity вызывает resync path.

15. `.github/workflows/quant-screener.yml:101`
    После исчерпания push retries workflow завершался кодом 0 даже при фактически неперсистнутом state.
    Исправлено: после последней неудачи job завершается `exit 1`.

## Ошибки из предыдущего аудита, которые сохранены в финальном проекте

Они также входят в текущую исправленную версию:

- silent substitution missing OI/funding/orderbook/trades -> fail-closed;
- настоящий 5m CVD по closed klines;
- 5m aggregate-trade window для VPIN с pagination;
- Wyckoff sweep по предыдущей 5m OHLC structure;
- directional sweep pattern сохраняется до composite score;
- risk engine reject'ит invalid stop/target вместо silent rewrite;
- fee-inclusive execution PnL;
- next-bar OPEN entry вместо silent next-close substitution;
- DSR не использует выдуманный `n_trials=20` и принимает raw kurtosis;
- rolling beta для alt/BTC;
- `nextFundingTime=0` fail-closed;
- NaN/Inf quality gate fail-closed;
- MCP unknown kwargs/missing required args reject'ятся;
- state persistence atomic + fsync;
- API key передаётся для MARKET_DATA sentiment endpoint;
- workflow checkout `persist-credentials: false` и concurrency cancellation.

## Binance protocol points verified

`/fapi/v1/aggTrades` поддерживает `startTime/endTime`, `fromId`, limit до 1000; interval query ограничен одним часом. Это делает временную агрегацию и pagination обязательной для полного 5m window.

Binance также публикует `fundingIntervalHours` в `/fapi/v1/fundingInfo` для symbols, где funding interval был изменён; поэтому unconditional 8h normalization была реальной ошибкой.

## Tests / verification

Добавлены или обновлены regression tests для:
- next-open + execution-fee PnL;
- exact bar-return decomposition;
- zero-volume liquidation rejection;
- invalid execution-cost inputs / zero spread;
- non-finite signal overrides;
- non-finite factor history;
- invalid funding gate input;
- order-book `lastUpdateId + 1` rule;
- MCP notification semantics.

Итого в `tests/test_quant_engine.py`: 32 test functions.

Machine checks in the provided environment:

- Python AST parse: PASS, 20 files.
- `python -m compileall -q .`: PASS.
- `pytest`: НЕ ПОЛУЧИЛОСЬ ПРОГНАТЬ COLLECTION, потому что в sandbox отсутствуют `polars` и `msgspec`. Это инфраструктурное ограничение среды, не успешный test run.

CI workflow содержит `pip install -r requirements.txt` и `python -m pytest -q`; именно этот run должен быть authoritative verification после установки dependencies.

## Residual risks — НЕ НАЗВАНЫ «ошибками» без доказательства

1. Execution cost constants (`50ms`, adverse factor, fee defaults) требуют calibration на реальном fill/execution dataset. Их изменение без данных не является доказанным fix.

2. Signal scaling constants / sweep thresholds требуют OOS calibration, но статический аудит не доказывает их неправильность.

3. `compute_volatility_compression_jit()` и `StreamingVPINState` сейчас не находятся на production screener path. Для первого также не определена требуемая формула volume normalization; поэтому поведение не выдумывалось и не переписывалось.

4. Cron REST budget остаётся практически насыщенным для 100 symbols и может расти из-за aggregate-trade pagination. Это подтверждённый capacity risk, но не deterministic correctness bug. Для устранения нужен отдельный throughput design, а не новый magic timeout/limit.

5. Full pytest remains environment-blocked until `polars`/`msgspec` are installed. Production rollout должен блокироваться отсутствием CI test pass.
