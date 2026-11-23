"""Rebuild every result from the recorded data in one go.

    uv run python run_all.py                      # all products found in data/
    uv run python run_all.py --products BTC-USD

Writes to results/:
  data_summary.csv     rows, hours covered, segments, gaps per product
  feature_ic.csv       rank correlation of each feature with each future move
  model_metrics.csv    out-of-sample IC / hit rate by model and horizon
  predictions_<product>.parquet   out-of-sample predictions (input to the cost analysis)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

import evaluate as E
import features as F

RESULTS = Path(__file__).parent / "results"


def summarize(df: pl.DataFrame, product: str) -> dict:
    t0, t1 = df["ts_ns"].min(), df["ts_ns"].max()
    covered = df.height * F.STEP_MS / 1000 / 3600
    return {
        "product": product, "rows": df.height,
        "start": str(df["time"].min()), "end": str(df["time"].max()),
        "span_hours": round((t1 - t0) / 3.6e12, 2), "covered_hours": round(covered, 2),
        "segments": df["seg"].n_unique(),
        "median_spread_bps": round(float(df["spread_bps"].median()), 4),
        "trades": int(df["n_trades"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=None)
    ap.add_argument("--data", type=Path, default=F.DATA)
    args = ap.parse_args()
    products = args.products or sorted(p.name.split("=", 1)[1] for p in (args.data / "book").glob("product=*"))
    RESULTS.mkdir(exist_ok=True)

    summaries, fics, metrics = [], [], []
    for product in products:
        t = time.time()
        df = F.build_product(product, args.data)
        summaries.append(summarize(df, product))
        print(f"[{product}] {df.height:,} rows, {summaries[-1]['segments']} segments, "
              f"{summaries[-1]['covered_hours']} h of data ({time.time() - t:.1f}s)")
        fics.append(E.feature_ics(df, product))
        m, preds = E.evaluate(df, product)
        if m.height:
            metrics.append(m)
            preds.write_parquet(RESULTS / f"predictions_{product}.parquet")
        print(f"[{product}] evaluated ({time.time() - t:.1f}s)")

    pl.DataFrame(summaries).write_csv(RESULTS / "data_summary.csv")
    pl.concat(fics, how="diagonal").write_csv(RESULTS / "feature_ic.csv")
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print(pl.DataFrame(summaries))
        print(pl.concat(fics, how="diagonal"))
        if metrics:
            m = pl.concat(metrics, how="diagonal")
            m.write_csv(RESULTS / "model_metrics.csv")
            print(m.select("product", "horizon", "model", "n_test", "ic", "fold_ic_mean",
                           "fold_ic_std", "hit", "flat", "abs_move_bps", "auc"))
        else:
            print("not enough data to evaluate yet")


if __name__ == "__main__":
    main()
