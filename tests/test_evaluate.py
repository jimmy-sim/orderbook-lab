import numpy as np

from evaluate import hit_rate, walk_forward


def test_walk_forward_is_time_ordered_with_embargo():
    ts = (np.arange(10_000) * 100_000_000).astype(np.int64)  # 100ms grid, 1000s total
    folds = walk_forward(ts, n_blocks=5, embargo_s=120)
    assert len(folds) == 4
    for f in folds:
        assert ts[f.train].max() < ts[f.test].min()                    # train strictly before test
        assert ts[f.test].min() - ts[f.train].max() >= 120e9            # embargo respected
        assert np.all(np.diff(f.test) == 1)                             # test block is contiguous


def test_hit_rate_ignores_flat_moves():
    pred = np.array([1.0, -1.0, 1.0, 1.0])
    y = np.array([2.0, 0.0, -1.0, 0.5])
    assert hit_rate(pred, y) == 2 / 3
