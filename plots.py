"""Draw the README charts from the latest results.

    uv run python run_all.py   # first: rebuild results/
    uv run python plots.py     # then: writes docs/*.png

Charts
------
signal.png       average price move 1s later, by order book imbalance (the signal itself)
decay.png        out-of-sample rank correlation vs. horizon (how fast the signal fades)
edge_vs_fees.png best gross edge per trade vs. round-trip taker fees (why it isn't tradeable)

All three describe the same data: signal.png is cut to the time window that
run_all.py evaluated, so rerunning plots.py while the recorder keeps writing
doesn't mix in newer data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

import costs as C  # noqa: E402
import features as F  # noqa: E402

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

COLORS = {"BTC-USD": "#2a78d6", "ETH-USD": "#eb6834"}  # categorical slots 1 and 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL = "logit"  # all-feature model reported in the README table
HORIZONS = [1, 5, 30, 60]
BUCKETS = 10


def figure(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.06, 0.885, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.1, right=0.86, top=0.78, bottom=0.15)
    return fig, ax


def window() -> tuple[dict[str, int], str]:
    """End timestamp per product, and a caption, for the data run_all.py evaluated."""
    summary = pl.read_csv(RESULTS / "data_summary.csv")
    ends = {}
    for product in summary["product"]:
        path = RESULTS / f"predictions_{product}.parquet"
        ends[product] = pl.scan_parquet(path).select(pl.col("ts_ns").max()).collect().item()
    hours = summary["covered_hours"].min()
    day = datetime.fromtimestamp(max(ends.values()) / 1e9, tz=timezone.utc).strftime("%b %-d, %Y")
    return ends, f"About {hours:.0f} hours of Coinbase data through {day} · preliminary"


def signal(ends: dict[str, int], caption: str) -> None:
    fig, ax = figure("Order book imbalance predicts the next second", caption)
    ax.axhline(0, color=AXIS, linewidth=1)
    for product, end in ends.items():
        df = (
            F.build_product(product)
            .filter(pl.col("ts_ns") <= end)
            .select("imb_1", "fwd_1s")
            .drop_nulls()
        )
        g = (
            df.with_columns(
                (pl.col("imb_1").rank("ordinal") * BUCKETS // (df.height + 1)).alias("bucket")
            )
            .group_by("bucket")
            .agg(pl.col("imb_1").mean(), pl.col("fwd_1s").mean())
            .sort("bucket")
        )
        x, y = g["imb_1"].to_list(), g["fwd_1s"].to_list()
        ax.plot(x, y, color=COLORS[product], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=product)
        ax.annotate(product, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set_xlabel("Imbalance at the best price   (−1 = all sellers  ·  +1 = all buyers)",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("Average price move 1 s later (bps)", color=INK2, fontsize=9.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
    fig.text(0.06, 0.03, "Each point is one tenth of all 100 ms snapshots, grouped by imbalance. "
             "1 bp = 0.01%.", fontsize=8, color=MUTED)
    fig.savefig(DOCS / "signal.png", facecolor=SURFACE)
    plt.close(fig)


def decay(caption: str) -> None:
    m = pl.read_csv(RESULTS / "model_metrics.csv").filter(pl.col("model") == MODEL)
    fig, ax = figure("The signal fades within seconds", caption)
    ax.axhline(0, color=AXIS, linewidth=1)
    for product, color in COLORS.items():
        rows = m.filter(pl.col("product") == product)
        ic = {int(h.removeprefix("fwd_").removesuffix("s")): v
              for h, v in zip(rows["horizon"], rows["ic"])}
        y = [ic[h] for h in HORIZONS]
        ax.plot(HORIZONS, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=product)
        ax.annotate(f"{y[0]:.2f}", (HORIZONS[0], y[0]), xytext=(-8, 0), textcoords="offset points",
                    color=INK2, fontsize=9, ha="right", va="center")
        ax.annotate(product, (HORIZONS[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS, [f"{h} s" for h in HORIZONS])
    ax.minorticks_off()
    ax.set_xlim(0.7, 75)
    ax.set_xlabel("How far ahead the model predicts", color=INK2, fontsize=9.5)
    ax.set_ylabel("Rank correlation, out of sample", color=INK2, fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)
    fig.text(0.06, 0.03, "Spearman correlation between prediction and actual move, on data the "
             "model never trained on. 0 = no skill.", fontsize=8, color=MUTED)
    fig.savefig(DOCS / "decay.png", facecolor=SURFACE)
    plt.close(fig)


def edge_vs_fees(caption: str) -> None:
    grid = pl.read_csv(RESULTS / "cost_fee_grid.csv").filter(pl.col("product") == "BTC-USD")
    best = grid.sort("gross_bps", descending=True).row(0, named=True)
    labels = [f"BTC edge per trade\n(best setting: {best['horizon'].removeprefix('fwd_')}, "
              f"top {best['top_q']:.1%} of signals)"]
    values = [best["gross_bps"]]
    colors = [COLORS["BTC-USD"]]
    for name, _maker, taker in sorted(C.FEE_TIERS, key=lambda t: t[2]):
        if taker == 0:
            continue
        tier = name.split(" (")[0]
        labels.append(f"{tier[0].upper() + tier[1:]} fee\n({taker:.2f}% per side)")
        values.append(2 * taker * 100)  # round trip, percent -> bps
        colors.append(MUTED)
    fig, ax = figure("The edge is tiny next to trading fees", caption)
    ys = list(range(len(values)))[::-1]
    ax.barh(ys, values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, v in zip(ys, values):
        ax.annotate(f"{v:.1f} bps" if v < 10 else f"{v:.0f} bps", (v, y), xytext=(6, 0),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_yticks(ys, labels, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Basis points per round trip (buy + sell)   ·   1 bp = 0.01%",
                  color=INK2, fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.15)
    fig.subplots_adjust(left=0.33, right=0.95)
    fig.text(0.06, 0.03, "Gross edge is before costs. Fees are round-trip taker fees for "
             "Coinbase fee scenarios in costs.py.", fontsize=8, color=MUTED)
    fig.savefig(DOCS / "edge_vs_fees.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    ends, caption = window()
    decay(caption)
    edge_vs_fees(caption)
    signal(ends, caption)
    print(f"wrote {', '.join(sorted(p.name for p in DOCS.glob('*.png')))} to {DOCS}")


if __name__ == "__main__":
    main()
