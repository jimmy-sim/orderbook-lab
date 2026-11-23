"""How well do the features predict future mid-price moves?

Splits (no shuffling, ever)
------
Rows are cut into N_BLOCKS consecutive time blocks. Fold i trains on every block
before i and tests on block i ("expanding walk-forward"), so the model is always
tested on data from *after* everything it learned from.

Embargo: the last training rows have labels that look up to 60s into the future, i.e.
into the test block. Training rows within EMBARGO_S of the test start are dropped so
no test-period price leaks into training.

Models
------
imb1_ols   one feature (top-of-book imbalance), linear regression. The textbook signal.
ols        all features, linear regression (ridge with a tiny penalty for stability).
logit      all features, logistic regression on direction (up vs down; flat rows dropped).
gbm        all features, gradient-boosted trees (sklearn HistGradientBoosting).

Metrics (on pooled out-of-sample predictions)
-------
ic         Spearman rank correlation between prediction and realised move. The standard
           quant measure: 0 = useless, 0.05 is already meaningful at high frequency.
fold_ic    mean and std of IC across test folds (stability).
hit        share of rows where sign(prediction) == sign(move), counting only rows where
           the mid actually moved. 50% = coin flip.
flat       share of rows where the mid did not move at all over the horizon.
auc        (logit only) chance the model ranks a random up-move above a random down-move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURES, HORIZONS_S, LABELS

N_BLOCKS = 5
EMBARGO_S = 2 * max(HORIZONS_S)
GBM_MAX_TRAIN = 300_000  # subsample training rows for the tree model (speed)
SEED = 0


@dataclass
class Fold:
    train: np.ndarray  # row indices
    test: np.ndarray


def walk_forward(ts_ns: np.ndarray, n_blocks: int = N_BLOCKS, embargo_s: float = EMBARGO_S) -> list[Fold]:
    n = len(ts_ns)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    folds = []
    for i in range(1, n_blocks):
        lo, hi = edges[i], edges[i + 1]
        cutoff = ts_ns[lo] - embargo_s * 1e9
        train = np.arange(0, lo)[ts_ns[:lo] < cutoff]
        if len(train) and hi > lo:
            folds.append(Fold(train, np.arange(lo, hi)))
    return folds


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pl.DataFrame({"a": a, "b": b}).select(pl.corr("a", "b", method="spearman")).item())


def hit_rate(pred: np.ndarray, y: np.ndarray) -> float:
    m = (y != 0) & (pred != 0)
    return float((np.sign(pred[m]) == np.sign(y[m])).mean()) if m.any() else float("nan")


def fit_predict(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "imb1_ols":
        i = FEATURES.index("imb_1")
        Xtr, Xte = Xtr[:, [i]], Xte[:, [i]]
    if model in ("imb1_ols", "ols"):
        sc = StandardScaler().fit(Xtr)
        return Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
    if model == "logit":
        m = ytr != 0
        if len(np.unique(ytr[m] > 0)) < 2:
            return np.zeros(len(Xte))  # training window only ever went one way: no view
        sc = StandardScaler().fit(Xtr[m])
        clf = LogisticRegression(max_iter=500).fit(sc.transform(Xtr[m]), (ytr[m] > 0).astype(int))
        return clf.predict_proba(sc.transform(Xte))[:, 1] - 0.5  # >0 means "up"
    if model == "gbm":
        rng = np.random.default_rng(SEED)
        if len(ytr) > GBM_MAX_TRAIN:
            idx = np.sort(rng.choice(len(ytr), GBM_MAX_TRAIN, replace=False))
            Xtr, ytr = Xtr[idx], ytr[idx]
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)
        return g.fit(Xtr, ytr).predict(Xte)
    raise ValueError(model)


MODELS = ("imb1_ols", "ols", "logit", "gbm")


def evaluate(df: pl.DataFrame, product: str, models=MODELS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (metrics table, out-of-sample predictions)."""
    metrics, preds = [], []
    for label in LABELS:
        exit_col = label.replace("fwd_", "spread_exit_")
        d = df.select(["ts_ns", "mid", *FEATURES, label, exit_col]).drop_nulls().drop_nans()
        d = d.filter(pl.all_horizontal(pl.col(FEATURES).is_finite()))
        if d.height < 1000:
            continue
        ts = d["ts_ns"].to_numpy()
        X = d.select(FEATURES).to_numpy()
        y = d[label].to_numpy()
        folds = walk_forward(ts)
        for model in models:
            p_all, y_all, fold_ics, idx_all, fold_id = [], [], [], [], []
            for k, f in enumerate(folds):
                fold_id.append(np.full(len(f.test), k))
                p = fit_predict(model, X[f.train], y[f.train], X[f.test])
                p_all.append(p), y_all.append(y[f.test]), idx_all.append(f.test)
                fold_ics.append(spearman(p, y[f.test]))
            p, yy, idx = np.concatenate(p_all), np.concatenate(y_all), np.concatenate(idx_all)
            row = {
                "product": product, "horizon": label, "model": model,
                "n_test": len(yy), "folds": len(folds),
                "ic": spearman(p, yy),
                "fold_ic_mean": float(np.nanmean(fold_ics)), "fold_ic_std": float(np.nanstd(fold_ics)),
                "hit": hit_rate(p, yy), "flat": float((yy == 0).mean()),
                "abs_move_bps": float(np.abs(yy).mean()),
            }
            if model == "logit":
                m = yy != 0
                row["auc"] = float(roc_auc_score(yy[m] > 0, p[m])) if m.any() else float("nan")
            metrics.append(row)
            preds.append(pl.DataFrame({
                "ts_ns": ts[idx], "horizon": label, "model": model, "pred": p, "y": yy,
                "fold": np.concatenate(fold_id),
                "spread_bps": d["spread_bps"].to_numpy()[idx],
                "spread_exit_bps": d[exit_col].to_numpy()[idx],
            }))
    return pl.DataFrame(metrics), (pl.concat(preds) if preds else pl.DataFrame())


def feature_ics(df: pl.DataFrame, product: str) -> pl.DataFrame:
    """Rank correlation of each raw feature with each future move, on all data.
    Descriptive only (no fitting), so no train/test split is needed."""
    out = []
    for label in LABELS:
        for f in FEATURES:
            d = df.select(f, label).drop_nulls().drop_nans()
            out.append({"product": product, "feature": f, "horizon": label,
                        "ic": spearman(d[f].to_numpy(), d[label].to_numpy())})
    return pl.DataFrame(out).pivot(on="horizon", index=["product", "feature"], values="ic")


if __name__ == "__main__":
    import sys

    from features import build_product
    product = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    df = build_product(product)
    m, _ = evaluate(df, product)
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=3):
        print(feature_ics(df, product))
        print(m)
