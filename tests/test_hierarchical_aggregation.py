import numpy as np

from nellie.feature_extraction.hierarchical import (
    aggregate_stats_for_class,
    append_to_array,
)


class _DummyChild:
    stats_to_aggregate = ["stat"]

    def __init__(self):
        self.stat = [np.array([1.0, 2.0, np.nan, 4.0], dtype=float)]


def test_aggregate_stats_low_memory_parity():
    dummy = _DummyChild()
    list_of_idxs = [np.array([0, 1]), np.array([2, 3])]

    fast = aggregate_stats_for_class(dummy, 0, list_of_idxs, low_memory=False)
    slow = aggregate_stats_for_class(dummy, 0, list_of_idxs, low_memory=True)

    assert fast["stat"]["mean"].shape == slow["stat"]["mean"].shape

    fast_arrs, fast_headers = append_to_array(fast)
    slow_arrs, slow_headers = append_to_array(slow)

    assert fast_headers == slow_headers
    for fast_vals, slow_vals in zip(fast_arrs, slow_arrs):
        assert np.allclose(fast_vals, slow_vals, equal_nan=True)
