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

### Always-on machine (e.g. a Mac Mini)

```bash
gh repo clone jimmy-sim/orderbook-lab ~/orderbook-lab   # outside Desktop/Documents (macOS blocks background access there)
cd ~/orderbook-lab && uv sync
./recorder.sh install    # starts now, at every login, and again after a crash
```

Also, in System Settings, turn on *Prevent automatic sleeping when the display is off* and
*Start up automatically after a power failure*. The recorder starts when you log in, so turn
on automatic login too, or log in once after each restart. `./recorder.sh uninstall` turns
always-on mode off.

For a short foreground test: `uv run python recorder.py --products BTC-USD --minutes 2 --flush 30`.

## Notes on the feed

- `level2_batch` is the unauthenticated book channel: the same updates as `level2`, sent in
  50ms batches (Coinbase reports it as `level2_50`).
- In Coinbase `match` messages, `side` is the **maker's** side. The recorder converts it to
  `taker_side` (who crossed the spread), which is what trade-flow features need.

## Features and evaluation

```bash
uv run python run_all.py     # features -> walk-forward models -> results/*.csv
uv run pytest -q             # checks on hand-built data (no look-ahead, gaps respected)
```

- [features.py](features.py): spread, order book imbalance at depths 1/3/5/10, microprice,
  trade-flow imbalance (1/5/30s), recent returns and realised volatility. Labels are future
  mid returns at 1/5/30/60s, in basis points. Data is split into *segments* at every hole
  or resync, and no window or label crosses a segment edge.
- [evaluate.py](evaluate.py): expanding walk-forward in time order (never shuffled), with
  a 120s embargo between train and test. Models: imbalance-only linear regression, all-feature
  linear and logistic regression, gradient-boosted trees. Reports information coefficient
  (rank correlation) and hit rate by horizon.
- [costs.py](costs.py): a toy one-position-at-a-time strategy on the out-of-sample
  predictions. It trades only the top q% most confident signals, with the cut-off taken from
  earlier test folds only. It reports the gross edge per trade, the spread paid, the
  break-even taker and maker fees, and net P&L under a grid of Coinbase fee tiers. Coinbase
  doesn't publish one public fee table (your rates appear on your own Advanced fees page),
  so set `REFERENCE` in `costs.py` to your tier.
