"""Does the prediction survive trading costs?

Input: the out-of-sample predictions written by run_all.py (results/predictions_*.parquet).

Toy strategy, one position at a time
------------------------------------
Walk through the test period in time order. When flat and |prediction| is above a
threshold, buy (prediction > 0) or sell short (< 0) one unit, hold for exactly the label
horizon H, then close. Signals that fire while a position is open are ignored, so trades
never overlap and each one is a separate bet.

Threshold: "trade only on the top q% most confident predictions". To avoid peeking, the
cut-off used in test fold k is the quantile of |prediction| over the *earlier* test folds
only, so the first fold is never traded.

Costs per round trip, in basis points of price (1 bp = 0.01%)
------------------------------------------------------------
gross       sign(prediction) x mid-to-mid return over H. What the signal "earns".
spread      half the spread at entry + half at exit. Crossing the spread is how a taker
            pays for immediacy: buy at the ask, sell at the bid.
taker net   gross - spread - 2 x taker fee
maker net   gross + spread - 2 x maker fee. An optimistic upper bound: assumes both legs fill
            instantly as passive limit orders at the touch, earning the half-spread. In
            reality passive orders fill mostly when the price is about to move against
            them (adverse selection), and often don't fill at all.

Break-even fees (per side)
--------------------------
taker:  (gross - spread) / 2       the most you could pay per trade as a taker and still break even
maker:  (gross + spread) / 2       the same for the optimistic maker

Fees
----
Coinbase does not publish one public table: your rates show on your own Coinbase Advanced
fees page once signed in, and they have changed several times. FEE_TIERS below are
scenarios spanning what public sources report (late 2026). Set REFERENCE to your own tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

RESULTS = Path(__file__).parent / "results"

# (name, maker %, taker %)  -- per side, as a percent of notional
FEE_TIERS = [
    ("retail high (1.20/0.60)", 0.60, 1.20),
    ("US entry 2026 (0.90/0.50)", 0.50, 0.90),
    ("retail typical (0.60/0.40)", 0.40, 0.60),
    ("mid tier (0.25/0.15)", 0.15, 0.25),
    ("high volume (0.05/0.00)", 0.00, 0.05),
    ("zero fees", 0.00, 0.00),
]
REFERENCE = "retail typical (0.60/0.40)"
TOP_Q = (1.0, 0.10, 0.01, 0.001)  # trade on all / top 10% / top 1% / top 0.1% of |prediction|


def pct_to_bps(x: float) -> float:
    return x * 100.0


def simulate(ts_ns: np.ndarray, pred: np.ndarray, fold: np.ndarray, horizon_s: float,
             top_q: float) -> np.ndarray:
    """Indices of rows where a trade is opened (no overlap, no look-ahead threshold)."""
    order = np.argsort(ts_ns, kind="stable")
    ts_ns, pred, fold = ts_ns[order], pred[order], fold[order]
    a = np.abs(pred)
    thr = np.full(len(a), np.inf)
    for k in np.unique(fold):
        past = a[fold < k]
        if len(past):
            thr[fold == k] = np.quantile(past, 1 - top_q) if top_q < 1 else 0.0
    cand = np.flatnonzero(a > thr)
    hold = int(horizon_s * 1e9)
    opened, free_at = [], -1
    for i in cand:
        if ts_ns[i] >= free_at:
            opened.append(i)
            free_at = ts_ns[i] + hold
    return order[np.array(opened, dtype=int)]


def trade_table(preds: pl.DataFrame) -> pl.DataFrame:
    """One row per (horizon, model, top_q): trade count and average economics in bp."""
    out = []
    for (horizon, model), g in preds.group_by(["horizon", "model"], maintain_order=True):
        h = float(horizon.removeprefix("fwd_").removesuffix("s"))
        ts, p, y = g["ts_ns"].to_numpy(), g["pred"].to_numpy(), g["y"].to_numpy()
        fold = g["fold"].to_numpy()
        sp = (g["spread_bps"].to_numpy() + g["spread_exit_bps"].to_numpy()) / 2
        traded_days = (ts[fold > 0].max() - ts[fold > 0].min()) / 8.64e13 if (fold > 0).any() else np.nan
        for q in TOP_Q:
            idx = simulate(ts, p, fold, h, q)
            if len(idx) == 0:
                continue
            gross = np.sign(p[idx]) * y[idx]
            spread = sp[idx]
            row = {
                "horizon": horizon, "model": model, "top_q": q, "trades": len(idx),
                "trades_per_day": len(idx) / traded_days if traded_days else np.nan,
                "hit": float((gross > 0).sum() / max((gross != 0).sum(), 1)),
                "gross_bps": float(gross.mean()),
                "spread_bps": float(spread.mean()),
                "breakeven_taker_fee_bps": float((gross - spread).mean() / 2),
                "breakeven_maker_fee_bps": float((gross + spread).mean() / 2),
            }
            for name, maker, taker in FEE_TIERS:
                if name == REFERENCE:
                    row["taker_net_bps"] = float((gross - spread - 2 * pct_to_bps(taker)).mean())
                    row["maker_net_bps_optimistic"] = float((gross + spread - 2 * pct_to_bps(maker)).mean())
            out.append(row)
    return pl.DataFrame(out)


def fee_grid(trades: pl.DataFrame) -> pl.DataFrame:
    """Average net bp per trade under every fee tier, for the best row of each horizon
    (highest gross edge per trade with at least 30 trades)."""
    rows = []
    for (horizon,), g in trades.filter(pl.col("trades") >= 30).group_by(["horizon"], maintain_order=True):
        best = g.sort("gross_bps", descending=True).row(0, named=True)
        for name, maker, taker in FEE_TIERS:
            rows.append({
                "horizon": horizon, "model": best["model"], "top_q": best["top_q"],
                "gross_bps": best["gross_bps"], "tier": name,
                "taker_net_bps": best["gross_bps"] - best["spread_bps"] - 2 * pct_to_bps(taker),
                "maker_net_bps_optimistic": best["gross_bps"] + best["spread_bps"] - 2 * pct_to_bps(maker),
            })
    return pl.DataFrame(rows)


def run(product: str, results: Path = RESULTS) -> tuple[pl.DataFrame, pl.DataFrame]:
    preds = pl.read_parquet(results / f"predictions_{product}.parquet")
    t = trade_table(preds).with_columns(product=pl.lit(product))
    g = fee_grid(t).with_columns(product=pl.lit(product)) if t.height else pl.DataFrame()
    return t, g


if __name__ == "__main__":
    import sys
    product = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    t, g = run(product)
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        print(t)
        print(g)
