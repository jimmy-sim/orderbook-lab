"""Turn recorded book snapshots and trades into features and labels.

Everything lives on the recorder's 100ms grid: one row per book snapshot.

Segments
--------
The grid has holes: reconnects, the laptop sleeping, recorder restarts. A feature or
label must never reach across a hole (a "60s return" spanning a 20-minute outage is
not a 60s return). So rows are split into *segments*: a new segment starts whenever
the gap to the previous row is more than MAX_GAP_MS, or a logged resync falls in
between. All rolling windows and look-aheads are computed within a segment, and rows
whose window would cross a segment edge get null.

Features (all known at time t, using only data up to t)
--------
spread_bps        ask1 - bid1, in basis points of mid (1 bp = 0.01%)
imb_k             (bid size - ask size) / (bid size + ask size), top k levels, k in DEPTHS.
                  +1 = all size on the bid, -1 = all on the ask.
micro_bps         microprice minus mid, in bp. Microprice = (bid1*ask_sz1 + ask1*bid_sz1)
                  / (bid_sz1 + ask_sz1): it leans toward the side with *less* size, i.e.
                  toward where the price is more likely to tick next.
tfi_Ns            trade-flow imbalance over the last N seconds:
                  (taker buy volume - taker sell volume) / total volume. 0 when no trades.
vol_Ns            traded volume (base units) over the last N seconds
ret_Ns            log return of mid over the last N seconds, in bp
rv_Ns             realised volatility: std of 100ms mid log returns over the last N s, in bp

Labels (the future; never used as inputs)
------
fwd_Hs            log return of mid from t to t+H seconds, in bp, H in HORIZONS
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent / "data"
STEP_MS = 100
MAX_GAP_MS = 1_000                   # a hole longer than this starts a new segment
DEPTHS = (1, 3, 5, 10)
FLOW_WINDOWS_S = (1, 5, 30)
RET_WINDOWS_S = (1, 5, 30)
VOL_WINDOWS_S = (30, 300)
HORIZONS_S = (1, 5, 30, 60)
NS = 1_000_000


def rows(seconds: float) -> int:
    return int(round(seconds * 1000 / STEP_MS))


# ---- loading -------------------------------------------------------------------------
def load(kind: str, product: str, root: Path = DATA) -> pl.DataFrame:
    files = sorted((root / kind / f"product={product}").glob("date=*/hour=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} data for {product} under {root}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    key = "ts_ns" if kind == "book" else "trade_id"
    return df.unique(key, keep="first").sort("ts_ns")


def load_resyncs(product: str, root: Path = DATA) -> list[tuple[int, int]]:
    """(start_ns, end_ns) of every logged book resync for this product."""
    path = root / "gaps.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        g = json.loads(line)
        if g.get("kind") == "book_resync" and g.get("product") == product:
            out.append((g["start_ns"], g["end_ns"]))
    return out


# ---- segments ------------------------------------------------------------------------
def add_segments(book: pl.DataFrame, resyncs: list[tuple[int, int]] = ()) -> pl.DataFrame:
    gap = book["ts_ns"].diff().fill_null(0) > MAX_GAP_MS * NS
    brk = gap
    if resyncs:
        # a resync whose end falls between row i-1 and row i breaks the series at i
        ends = pl.Series(sorted(e for _, e in resyncs), dtype=pl.Int64)
        idx = book["ts_ns"].search_sorted(ends, side="left")
        hit = pl.zeros(book.height, dtype=pl.Boolean, eager=True).scatter(
            idx.filter(idx < book.height), True)
        brk = brk | hit
    return book.with_columns(seg=brk.cast(pl.Int32).cum_sum())


# ---- trades onto the grid --------------------------------------------------------------
def trades_to_grid(book: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Sum taker buy/sell volume into each 100ms bin. A trade received in (t-100ms, t]
    is credited to the snapshot at t, so the snapshot at t 'knows' about it."""
    step = STEP_MS * NS
    binned = (
        trades.with_columns(ts_ns=((pl.col("ts_ns") + step - 1) // step) * step)
        .group_by("ts_ns")
        .agg(
            buy_vol=pl.col("size").filter(pl.col("taker_side") == "buy").sum(),
            sell_vol=pl.col("size").filter(pl.col("taker_side") == "sell").sum(),
            n_trades=pl.len(),
        )
    )
    return book.join(binned, on="ts_ns", how="left").with_columns(
        pl.col("buy_vol", "sell_vol").fill_null(0.0), pl.col("n_trades").fill_null(0))


# ---- features and labels -----------------------------------------------------------------
def build(book: pl.DataFrame, trades: pl.DataFrame | None = None,
          resyncs: list[tuple[int, int]] = ()) -> pl.DataFrame:
    df = add_segments(book.sort("ts_ns"), resyncs)
    if trades is not None and trades.height:
        df = trades_to_grid(df, trades)
    else:
        df = df.with_columns(buy_vol=pl.lit(0.0), sell_vol=pl.lit(0.0), n_trades=pl.lit(0))

    mid = (pl.col("bid_px_1") + pl.col("ask_px_1")) / 2
    df = df.with_columns(mid=mid, logmid=mid.log())
    seg = "seg"
    # 100ms mid log return in bp (null at the first row of each segment)
    df = df.with_columns(r1=((pl.col("logmid") - pl.col("logmid").shift(1)) * 1e4).over(seg))

    def lag(expr: pl.Expr, n: int) -> pl.Expr:
        return expr.shift(n).over(seg)

    def roll_sum(expr: pl.Expr, n: int) -> pl.Expr:
        return expr.rolling_sum(n).over(seg)

    cols: list[pl.Expr] = [
        ((pl.col("ask_px_1") - pl.col("bid_px_1")) / pl.col("mid") * 1e4).alias("spread_bps"),
    ]
    for k in DEPTHS:
        b = pl.sum_horizontal([f"bid_sz_{i}" for i in range(1, k + 1)])
        a = pl.sum_horizontal([f"ask_sz_{i}" for i in range(1, k + 1)])
        cols.append(((b - a) / (b + a)).alias(f"imb_{k}"))
    micro = (pl.col("bid_px_1") * pl.col("ask_sz_1") + pl.col("ask_px_1") * pl.col("bid_sz_1")) / (
        pl.col("bid_sz_1") + pl.col("ask_sz_1"))
    cols.append(((micro - pl.col("mid")) / pl.col("mid") * 1e4).alias("micro_bps"))
    for w in FLOW_WINDOWS_S:
        bv, sv = roll_sum(pl.col("buy_vol"), rows(w)), roll_sum(pl.col("sell_vol"), rows(w))
        tot = bv + sv
        cols.append(pl.when(tot > 0).then((bv - sv) / tot).otherwise(0.0).alias(f"tfi_{w}s"))
        cols.append(tot.alias(f"vol_{w}s"))
    for w in RET_WINDOWS_S:
        cols.append(((pl.col("logmid") - lag(pl.col("logmid"), rows(w))) * 1e4).alias(f"ret_{w}s"))
    for w in VOL_WINDOWS_S:
        cols.append(pl.col("r1").rolling_std(rows(w)).over(seg).alias(f"rv_{w}s"))
    for h in HORIZONS_S:
        cols.append(((lag(pl.col("logmid"), -rows(h)) - pl.col("logmid")) * 1e4).alias(f"fwd_{h}s"))

    # every window above is partitioned by `seg`, so it returns null near segment edges
    df = df.with_columns(cols)
    return df.with_columns(pl.col("ts_ns").cast(pl.Datetime("ns", "UTC")).alias("time"))


FEATURES = (["spread_bps", "micro_bps"] + [f"imb_{k}" for k in DEPTHS]
            + [f"tfi_{w}s" for w in FLOW_WINDOWS_S] + [f"vol_{w}s" for w in FLOW_WINDOWS_S]
            + [f"ret_{w}s" for w in RET_WINDOWS_S] + [f"rv_{w}s" for w in VOL_WINDOWS_S])
LABELS = [f"fwd_{h}s" for h in HORIZONS_S]


def build_product(product: str, root: Path = DATA) -> pl.DataFrame:
    book = load("book", product, root)
    try:
        trades = load("trades", product, root)
    except FileNotFoundError:
        trades = None
    return build(book, trades, load_resyncs(product, root))


if __name__ == "__main__":
    import sys
    product = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    df = build_product(product)
    print(f"{product}: {df.height:,} rows, {df['seg'].n_unique()} segments, "
          f"{df['time'].min()} -> {df['time'].max()}")
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=4):
        print(df.select(FEATURES + LABELS).describe())
