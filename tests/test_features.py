"""Checks on hand-built data where the right answer is known."""

import math

import polars as pl
import pytest

import features as F

STEP = F.STEP_MS * F.NS


def make_book(n: int, start_ns: int = 0, bid=100.0, ask=100.02, bid_sz=1.0, ask_sz=1.0) -> pl.DataFrame:
    row = {}
    for i in range(1, 11):
        row[f"bid_px_{i}"] = bid - 0.01 * (i - 1)
        row[f"ask_px_{i}"] = ask + 0.01 * (i - 1)
        row[f"bid_sz_{i}"] = bid_sz
        row[f"ask_sz_{i}"] = ask_sz
    return pl.DataFrame([{"ts_ns": start_ns + k * STEP, "exch_ns": 0, "n_updates": 1, **row}
                         for k in range(n)])


def test_imbalance_and_microprice():
    df = F.build(make_book(5, bid_sz=3.0, ask_sz=1.0))
    r = df.row(0, named=True)
    assert r["imb_1"] == pytest.approx(0.5)      # (3-1)/(3+1)
    assert r["imb_10"] == pytest.approx(0.5)
    # microprice = (100*1 + 100.02*3)/4 = 100.015; mid = 100.01 -> +0.005 above mid
    assert r["micro_bps"] == pytest.approx(0.005 / 100.01 * 1e4)
    assert r["spread_bps"] == pytest.approx(0.02 / 100.01 * 1e4)


def test_forward_return_and_no_lookahead_in_features():
    b1 = make_book(20)
    b2 = make_book(20, start_ns=20 * STEP, bid=100.10, ask=100.12)  # mid jumps +0.10 at row 20
    df = F.build(pl.concat([b1, b2]))
    # 1s = 10 rows ahead: row 10 sees the jump, row 9 does not
    assert df["fwd_1s"][9] == pytest.approx(0.0)
    assert df["fwd_1s"][10] == pytest.approx(math.log(100.11 / 100.01) * 1e4)
    # backward return must not see the jump until it has happened
    assert df["ret_1s"][19] == pytest.approx(0.0)
    assert df["ret_1s"][20] == pytest.approx(math.log(100.11 / 100.01) * 1e4)
    # last 10 rows have no 1s future
    assert df["fwd_1s"][30:].null_count() == 10


def test_labels_and_windows_do_not_cross_gaps():
    b1 = make_book(30)
    b2 = make_book(30, start_ns=30 * STEP + 5_000 * F.NS, bid=200.0, ask=200.02)  # 5s hole
    df = F.build(pl.concat([b1, b2]))
    assert df["seg"].n_unique() == 2
    # rows 20..29 would need data past the hole -> null, not a bogus 69% return
    assert df["fwd_1s"][20:30].null_count() == 10
    # first row after the hole has no 1s history
    assert df["ret_1s"][30] is None


def test_resync_breaks_segment_even_without_time_gap():
    b = make_book(40)
    resync_end = 20 * STEP - 1  # resync finished just before row 20
    df = F.build(b, resyncs=[(resync_end - 10, resync_end)])
    assert df["seg"].n_unique() == 2
    assert df["seg"][19] != df["seg"][20]


def test_trade_binning_and_flow_sign():
    b = make_book(30)
    t = pl.DataFrame({
        "ts_ns": [10 * STEP - 50 * F.NS,  # mid-bin -> credited to row 10
                  10 * STEP,              # exactly on row 10 -> row 10
                  10 * STEP + 1,          # 1ns later -> row 11
                  ],
        "trade_id": [1, 2, 3],
        "price": [100.02, 100.02, 100.0],
        "size": [1.0, 2.0, 4.0],
        "taker_side": ["buy", "buy", "sell"],
    })
    df = F.build(b, t)
    assert df["buy_vol"][10] == pytest.approx(3.0)
    assert df["sell_vol"][11] == pytest.approx(4.0)
    assert df["buy_vol"][9] == 0.0
    # 1s flow at row 11: buys 3, sells 4 -> (3-4)/7
    assert df["tfi_1s"][11] == pytest.approx(-1 / 7)
    assert df["tfi_1s"][10] == pytest.approx(1.0)
