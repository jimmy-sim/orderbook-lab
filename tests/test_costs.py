import numpy as np
import polars as pl
import pytest

import costs as C

S = 1_000_000_000  # 1s in ns


def test_no_overlap_and_first_fold_never_traded():
    ts = np.arange(40) * S // 10                 # 100ms grid, 4s
    pred = np.ones(40)
    fold = np.repeat([0, 1], 20)
    idx = C.simulate(ts, pred, fold, horizon_s=1.0, top_q=1.0)
    assert idx.min() >= 20                       # fold 0 only sets the threshold
    assert np.all(np.diff(ts[idx]) >= S)         # one position at a time, held 1s
    assert list(idx) == [20, 30]


def test_threshold_uses_only_earlier_folds():
    ts = np.arange(20) * S
    pred = np.r_[np.linspace(0.1, 1.0, 10), np.full(10, 0.5)]   # fold 1 never beats fold 0's top 10%
    fold = np.repeat([0, 1], 10)
    assert len(C.simulate(ts, pred, fold, 1.0, top_q=0.10)) == 0


def test_cost_arithmetic():
    preds = pl.DataFrame({
        "ts_ns": [0, 2 * S, 4 * S, 6 * S], "horizon": "fwd_1s", "model": "m",
        "pred": [1.0, 1.0, -1.0, 1.0], "y": [0.0, 10.0, -4.0, -2.0], "fold": [0, 1, 1, 1],
        "spread_bps": [1.0, 1.0, 1.0, 1.0], "spread_exit_bps": [1.0, 3.0, 1.0, 1.0],
    })
    t = C.trade_table(preds).filter(pl.col("top_q") == 1.0).row(0, named=True)
    assert t["trades"] == 3
    assert t["gross_bps"] == pytest.approx((10 + 4 - 2) / 3)
    assert t["spread_bps"] == pytest.approx((2 + 1 + 1) / 3)       # mean of half entry + half exit
    assert t["hit"] == pytest.approx(2 / 3)
    fee = C.pct_to_bps(0.60)
    assert t["taker_net_bps"] == pytest.approx(4 - 4 / 3 - 2 * fee)
    assert t["breakeven_taker_fee_bps"] == pytest.approx((4 - 4 / 3) / 2)
