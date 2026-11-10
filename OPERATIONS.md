# Screener operations

## External schedule

The market scan workflow is triggered by the external Cron-job.org job through the GitHub `repository_dispatch` event type `run_screener`.

Recommended Cron-job.org schedule: every 5 minutes, timezone `Europe/Moscow` (or keep the existing scheduler timezone if it is intentionally configured). Do not add a second independent cron trigger in GitHub Actions while Cron-job.org is active.

The workflow uses a single concurrency group with `queue: max`: one scan runs at a time and up to 100 pending runs can wait. This avoids GitHub's default behavior of replacing the previous pending run, but it is not an unlimited or real-time queue. If a scan consistently takes longer than five minutes, the queue grows. If the queue reaches 100 pending runs, GitHub can reject additional queued runs. Monitor the oldest queued run and the queue depth in Actions.

## State persistence

- `data/snapshots.jsonl` and `data/market_rows.jsonl` retain changed Coinalyze observations at approximately four-hour intervals, matching the P1 snapshot-comparison window (± configured tolerance).
- `data/notifications.jsonl` persists successful Telegram deliveries per recipient and the latest delivered scenario fingerprint.
- `data/runs.jsonl` is per-invocation diagnostics and is deliberately not committed on every scan.
- Git state commits therefore occur when durable market history or notification state changes, not merely because a run started or ended.

## Telegram behavior

- The whole-market summary is emitted at most once for each four-hour UTC observation bucket, and only when a new durable market snapshot is saved.
- Scenario alerts are keyed by the actual state of P1 checks and relevant chart confirmations/invalidation levels. A transition such as `retest not confirmed` → `retest confirmed` creates a new event even when the high-level scenario status remains `OBSERVATION`.
- Successful sends are recorded per chat. Failed recipients are retried on later runs without re-sending to recipients that already received the same state.
- Telegram 429 and transient 5xx/network failures receive bounded retries.

## Limits and known boundaries

- The 15-minute candle is shown as entry-timing context only. P1 does not define a standalone numeric 15m trigger threshold, so the screener does not invent one or silently turn it into a hard gate.
- P1 does not quantify the distance meant by “near the 24H low”. The screener displays the 24H low and distance derived from 24 closed Binance 1H candles, while retaining the conservative cascade guard; it does not invent a percentage cutoff.
- GitHub-hosted runners are ephemeral. Persisted P1 state is still stored in the repository's `data/` files; for much longer retention or strict delivery guarantees, migrate this state to a dedicated persistent database.
- The GitHub Actions test workflow runs on Python 3.11. The live market workflow still requires configured Coinalyze, WireGuard and Telegram secrets.
