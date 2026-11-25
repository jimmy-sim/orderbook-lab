"""Draw the README charts from the latest results.

    uv run python run_all.py   # first: rebuild results/
    uv run python plots.py     # then: writes docs/*.svg

Charts
------
signal.svg       average price move 1s later, by order book imbalance (the signal itself)
decay.svg        out-of-sample rank correlation vs. horizon (how fast the signal fades)
edge_vs_fees.svg best gross edge per trade vs. round-trip taker fees (why it isn't tradeable)

All three describe the same data: signal.svg is cut to the time window that
run_all.py evaluated, so rerunning plots.py while the recorder keeps writing
doesn't mix in newer data.
"""

from __future__ import annotations

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

COLORS = {"BTC-USD": "#1f77b4", "ETH-USD": "#ff7f0e"}
INK, MUTED, GRID = "#222222", "#555555", "#e6e6e6"
plt.rcParams["svg.fonttype"] = "none"
MODEL = "logit"  # all-feature model reported in the README table
HORIZONS = [1, 5, 30, 60]
BUCKETS = 10


def figure():
    """Use the same quiet, white-background style as the site's other figures."""
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.18)
    return fig, ax


def window() -> dict[str, int]:
    """End timestamp per product for the data run_all.py evaluated."""
    summary = pl.read_csv(RESULTS / "data_summary.csv")
    ends = {}
    for product in summary["product"]:
        path = RESULTS / f"predictions_{product}.parquet"
        ends[product] = pl.scan_parquet(path).select(pl.col("ts_ns").max()).collect().item()
    return ends


def signal(ends: dict[str, int]) -> None:
    fig, ax = figure()
    ax.axhline(0, color=MUTED, linewidth=1)
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
        ax.plot(x, y, color=COLORS[product], linewidth=2, marker="o", markersize=4, label=product)
    ax.set_xlabel("Best-price size imbalance", fontsize=11)
    ax.set_ylabel("Mean 1-second move (bps)", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.savefig(DOCS / "signal.svg", facecolor="white")
    plt.close(fig)


def decay() -> None:
    m = pl.read_csv(RESULTS / "model_metrics.csv").filter(pl.col("model") == MODEL)
    fig, ax = figure()
    ax.axhline(0, color=MUTED, linewidth=1)
    for product, color in COLORS.items():
        rows = m.filter(pl.col("product") == product)
        ic = {int(h.removeprefix("fwd_").removesuffix("s")): v
              for h, v in zip(rows["horizon"], rows["ic"])}
        y = [ic[h] for h in HORIZONS]
        ax.plot(HORIZONS, y, color=color, linewidth=2, marker="o", markersize=4, label=product)
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS, [f"{h} s" for h in HORIZONS])
    ax.minorticks_off()
    ax.set_xlim(0.7, 75)
    ax.set_xlabel("Prediction horizon", fontsize=11)
    ax.set_ylabel("Out-of-sample rank correlation", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.savefig(DOCS / "decay.svg", facecolor="white")
    plt.close(fig)


def edge_vs_fees() -> None:
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
        colors.append("#999999")
    fig, ax = figure()
    ys = list(range(len(values)))[::-1]
    ax.barh(ys, values, color=colors, height=0.62, edgecolor="white", linewidth=1)
    for y, v in zip(ys, values):
        ax.annotate(f"{v:.1f} bps" if v < 10 else f"{v:.0f} bps", (v, y), xytext=(6, 0),
                    textcoords="offset points", color=INK, fontsize=10, va="center")
    ax.set_yticks(ys, labels, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("Basis points per round trip", fontsize=11)
    ax.set_xlim(0, max(values) * 1.15)
    fig.subplots_adjust(left=0.34, right=0.96, top=0.96, bottom=0.18)
    fig.savefig(DOCS / "edge_vs_fees.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    ends = window()
    decay()
    edge_vs_fees()
    signal(ends)
    print(f"wrote {', '.join(sorted(p.name for p in DOCS.glob('*.svg')))} to {DOCS}")


if __name__ == "__main__":
    main()
