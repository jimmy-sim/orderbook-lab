# orderbook-lab

**Can the live Coinbase order book predict where Bitcoin's price moves in the next few seconds, and is that prediction worth anything once you pay to trade?**

This repo is my end-to-end answer. It covers an always-on recorder for Coinbase's live market data, a feature and labeling pipeline that won't let the future leak into the past, time-ordered model evaluation, and a cost model that checks whether any edge survives the spread and exchange fees.

Research only. Nothing here places orders or needs an account. It only listens to Coinbase's free public feed.

*Built by [Jimmy Sim](https://github.com/Jimmy-Sim), UC Berkeley (Applied Math & CS).*

---

## The short answer (so far)

**Yes, the order book predicts the next second or two. No, the edge isn't worth trading at any fee a normal account pays.**

> Preliminary: based on about 20 hours of data (BTC-USD and ETH-USD, Nov 24 2026). The recorder is still running, and these numbers will be updated as weeks of data accumulate.

How well the models predict, measured only on data they never trained on:

| | 1 s ahead | 5 s ahead | 30 s ahead | 60 s ahead |
|---|---|---|---|---|
| **BTC-USD** rank correlation | **0.36** | 0.21 | 0.08 | 0.04 |
| BTC-USD right direction* | 69% | 59% | 53% | 52% |
| **ETH-USD** rank correlation | 0.10 | 0.06 | 0.03 | ~0 |

\*Counting only the times the price actually moved. 50% is a coin flip.

![Average price move one second later, by order book imbalance: a rising curve from about −0.4 bps when sellers dominate the best price to about +0.4 bps when buyers dominate](docs/signal.png)

![Out-of-sample rank correlation by horizon: BTC falls from 0.36 at 1 second to 0.04 at 60 seconds; ETH from 0.10 to near zero](docs/decay.png)

- **Fast decay:** the signal is strong at 1 second and almost gone by 60 seconds.
- **One feature does most of the work:** the imbalance between bid and ask size at the best price reaches 0.32 on its own, against 0.36 for all features combined. Gradient-boosted trees do no better.
- **The fees swamp the edge:** even the most confident signals earn about 1–4 basis points (0.01–0.04%) per trade before costs. A round trip at a typical retail taker fee costs about 120 bps. Even the top-volume fee tier (10 bps round trip) loses money. The only scenario that comes out positive is an optimistic maker case that assumes every passive order fills, which real markets don't allow.

![Best BTC gross edge per trade, 3.8 bps, next to round-trip taker fees from 10 bps (high-volume tier) to 240 bps (highest retail tier)](docs/edge_vs_fees.png)

The takeaway: the edge lives at a fraction of a basis point. That's why short-horizon trading is a business of fee rebates, speed and enormous volume, not clever models alone.

---

## How it works

```mermaid
flowchart LR
    A[Coinbase public<br/>WebSocket feed] --> B[recorder.py<br/>always-on, 24/7]
    B --> C[(Parquet files<br/>hourly folders)]
    B --> G[(gaps.jsonl<br/>every outage logged)]
    C --> D[features.py<br/>features + labels]
    G --> D
    D --> E[evaluate.py<br/>walk-forward models]
    E --> F[costs.py<br/>does it survive fees?]
    F --> R[results/*.csv]
```

| File | What it does |
|---|---|
| [`recorder.py`](recorder.py) | Keeps a live copy of the BTC and ETH order books. Saves the top 10 price levels every 100 ms and every trade. Reconnects automatically and logs every gap. |
| [`recorder.sh`](recorder.sh) | Start, stop and status commands. `install` registers it with macOS so it runs 24/7 and restarts after a crash. |
| [`features.py`](features.py) | Turns raw snapshots into predictors (spread, order book imbalance, microprice, trade flow, recent returns, volatility) and into labels (future price moves at 1/5/30/60 s). |
| [`evaluate.py`](evaluate.py) | Trains four models (from a one-feature baseline to gradient-boosted trees) and scores them on future data only. |
| [`costs.py`](costs.py) | Simulates a simple one-position-at-a-time strategy on the predictions and computes profit after spread and fees across six fee tiers. |
| [`run_all.py`](run_all.py) | One command runs the whole pipeline, from raw data to results, in about 90 seconds. |
| [`plots.py`](plots.py) | Redraws the charts in this README from the latest results. |
| [`tests/`](tests) | Checks the subtle parts on hand-built data (see below). |

---

## Design decisions that are easy to get wrong

Most "price prediction" projects look great because of a mistake somewhere. These are the ones this project guards against, each with a test:

1. **No looking into the future.** Every feature uses only data available at that moment. *(`test_forward_return_and_no_lookahead_in_features`)*
2. **No reaching across outages.** If the connection drops for 20 minutes, a "60-second return" spanning the outage is meaningless. The data is split into segments at every gap or resync, and no window or label crosses a segment edge. *(`test_labels_and_windows_do_not_cross_gaps`, `test_resync_breaks_segment_even_without_time_gap`)*
3. **Test on the future, never on shuffled data.** Models train on earlier data and are tested on later data, with a 120-second embargo so labels that look ahead into the test period can't leak into training. *(`test_walk_forward_is_time_ordered_with_embargo`)*
4. **Count who actually traded.** Coinbase reports the *maker's* side of each trade. The recorder flips it to the *taker's* side, the one who crossed the spread, which is what trade-flow signals need. *(`test_trade_binning_and_flow_sign`)*
5. **No peeking when choosing which signals to trade.** The confidence cut-off is set from earlier test periods only, and the first period is never traded. *(`test_threshold_uses_only_earlier_folds`)*

---

## Run it yourself

Requires macOS or Linux, Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jimmy-sim/orderbook-lab && cd orderbook-lab
uv sync

uv run python recorder.py --products BTC-USD --minutes 2 --flush 30   # 2-minute test recording
uv run pytest -q                                                      # 10 tests

./recorder.sh start      # record in the background
./recorder.sh status     # running? disk used? gaps?
uv run python run_all.py # rebuild every result into results/
uv run python plots.py   # redraw the charts in docs/
```

For 24/7 recording on an always-on machine such as a Mac Mini, run `./recorder.sh install`. It starts at login and restarts after a crash. Keep the repo outside Desktop and Documents, because macOS blocks background access there. In System Settings, also turn on *Prevent automatic sleeping when the display is off* and *Start up automatically after a power failure*.

Data goes to `data/` and results to `results/`. Neither is committed. The recorder adds about 200 MB a day for two products.

---

## Limitations and what's next

- **More data.** About 20 hours covers one market mood. Weeks of data will show whether the signal is stable across days, weekends and volatile periods.
- **A realistic maker simulation.** The cost model's maker case assumes passive orders always fill. The real question for a market maker is where an order sits in the queue and whether it fills just before the price moves against it. Modeling that is the natural next step.
- **One exchange, two products.** Other venues and less liquid coins may behave differently.

---

<details>
<summary>Notes on the Coinbase feed</summary>

- `level2_batch` is the unauthenticated order book channel: the same updates as `level2`, sent in 50 ms batches.
- In Coinbase `match` messages, `side` is the **maker's** side. The recorder converts it to `taker_side`.
- Missed trades are detected from gaps in Coinbase's consecutive trade IDs. Book resyncs, reconnects and missed trades are all appended to `data/gaps.jsonl`.
- Fees: Coinbase doesn't publish a single public fee table, so `costs.py` uses a grid of scenarios. Set `REFERENCE` in `costs.py` to your own tier.

</details>
