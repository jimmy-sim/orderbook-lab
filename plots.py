"""Draw the README charts from the latest results.

    uv run python run_all.py   # first: rebuild results/
    uv run python plots.py     # then: writes docs/*.svg (+ .png copies)

Charts
------
signal           average mid move 1s later, by top-of-book imbalance (the signal itself)
decay            out-of-sample rank correlation vs. horizon (how fast the signal fades)
edge_vs_fees     best gross edge per trade vs. round-trip taker fees (why it isn't tradeable)

All three describe the same data: the signal chart is cut to the time window that
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

COLORS = {"BTC-USD": "tab:blue", "ETH-USD": "tab:orange"}
FEE_GRAY = "#b8bcc4"
MODEL = "logit"  # all-feature model reported in the README table
HORIZONS = [1, 5, 30, 60]
BUCKETS = 10
LINE = dict(linewidth=1.5, marker="o", markersize=7)


def window() -> tuple[dict[str, int], str]:
    """End timestamp per product, and a title suffix, for the data run_all.py evaluated."""
    summary = pl.read_csv(RESULTS / "data_summary.csv")
    ends = {}
    for product in summary["product"]:
        path = RESULTS / f"predictions_{product}.parquet"
        ends[product] = pl.scan_parquet(path).select(pl.col("ts_ns").max()).collect().item()
    return ends, f"{summary['covered_hours'].min():.0f} h of data, preliminary"


def save(fig, stem: str) -> None:
    """Write docs/<stem>.svg (shown in the README) and a .png copy."""
    fig.tight_layout()
    fig.savefig(DOCS / f"{stem}.svg", facecolor="white")
    fig.savefig(DOCS / f"{stem}.png", dpi=150, facecolor="white")
    plt.close(fig)


def signal(ends: dict[str, int], note: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
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
        ax.plot(g["imb_1"], g["fwd_1s"], color=COLORS[product], label=product, **LINE)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Mean 1 s mid move by top-of-book imbalance (deciles; {note})")
    ax.set_xlabel("Top-of-book imbalance (−1 = all ask size, +1 = all bid size)")
    ax.set_ylabel("Mean 1 s mid return (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    save(fig, "signal")


def decay(note: str) -> None:
    m = pl.read_csv(RESULTS / "model_metrics.csv").filter(pl.col("model") == MODEL)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for product, color in COLORS.items():
        rows = m.filter(pl.col("product") == product)
        ic = {int(h.removeprefix("fwd_").removesuffix("s")): v
              for h, v in zip(rows["horizon"], rows["ic"])}
        ax.plot(HORIZONS, [ic[h] for h in HORIZONS], color=color, label=product, **LINE)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_title(f"Out-of-sample rank IC by horizon ({MODEL}, walk-forward; {note})")
    ax.set_xlabel("Prediction horizon (seconds)")
    ax.set_ylabel("Spearman rank IC")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "decay")


def edge_vs_fees(note: str) -> None:
    grid = pl.read_csv(RESULTS / "cost_fee_grid.csv").filter(pl.col("product") == "BTC-USD")
    best = grid.sort("gross_bps", descending=True).row(0, named=True)
    labels = [f"BTC gross edge\n({best['horizon'].removeprefix('fwd_')}, top {best['top_q']:.0%} of signals)"]
    values = [best["gross_bps"]]
    colors = [COLORS["BTC-USD"]]
    for name, _maker, taker in sorted(C.FEE_TIERS, key=lambda t: t[2]):
        if taker == 0:
            continue
        labels.append(f"{name.split(' (')[0]}\n({taker:.2f}% taker)")
        values.append(2 * taker * 100)  # round trip, percent -> bps
        colors.append(FEE_GRAY)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = list(range(len(values)))[::-1]
    ax.barh(ys, values, color=colors, height=0.6)
    for y, v in zip(ys, values):
        ax.annotate(f"{v:.1f}" if v < 10 else f"{v:.0f}", (v, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_title(f"Best gross edge vs. round-trip taker fees ({note})")
    ax.set_xlabel("Basis points per round trip (1 bp = 0.01%)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save(fig, "edge_vs_fees")


def main() -> None:
    plt.style.use("default")
    DOCS.mkdir(exist_ok=True)
    ends, note = window()
    decay(note)
    edge_vs_fees(note)
    signal(ends, note)
    print(f"wrote {', '.join(sorted(p.name for p in DOCS.glob('*.svg')))} (+ .png) to {DOCS}")


if __name__ == "__main__":
    main()
