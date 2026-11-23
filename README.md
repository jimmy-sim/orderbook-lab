# orderbook-lab

Research project: can the Coinbase BTC-USD order book predict short-horizon (1–60s) price
moves, and does any edge survive trading costs? Research only. Nothing here places orders.

## Recording data

The recorder listens to Coinbase Exchange's free public WebSocket feed. No account or API key
is needed. It saves:

- **book**: top 10 bid/ask levels (price and size) every 100ms
- **trades**: every trade, with the aggressor (taker) side

Data goes to Parquet files in hourly folders under `data/` (not committed), for example
`data/book/product=BTC-USD/date=2026-11-23/hour=21/part-*.parquet`. Reconnects, book
resyncs and missed trades are appended to `data/gaps.jsonl`.

```bash
uv sync
./recorder.sh start     # runs in the background under `caffeinate`
./recorder.sh status    # running?, disk used, gaps, last log lines
./recorder.sh stop      # stops cleanly and writes out buffered rows
```

Rows are buffered and written every 5 minutes, so a crash loses at most 5 minutes.
Closing a MacBook's lid on battery will still sleep it; that shows up as a gap.

For a short foreground test: `uv run python recorder.py --products BTC-USD --minutes 2 --flush 30`.

## Notes on the feed

- `level2_batch` is the unauthenticated book channel: the same updates as `level2`, sent in
  50ms batches (Coinbase reports it as `level2_50`).
- In Coinbase `match` messages, `side` is the **maker's** side. The recorder converts it to
  `taker_side` (who crossed the spread), which is what trade-flow features need.
