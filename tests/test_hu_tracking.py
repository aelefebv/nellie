import numpy as np

from nellie.tracking.hu_tracking import HuMomentTracking


class DummyImInfo:
    def __init__(self, no_z=True):
        self.no_t = False
        self.no_z = no_z
        self.axes = "TYX" if no_z else "TZYX"
        self.shape = (2, 5, 5) if no_z else (2, 3, 5, 5)
        self.dim_res = {"T": 1.0, "Z": 1.0, "Y": 1.0, "X": 1.0}


def test_log_hu_no_nan():
    im_info = DummyImInfo()
    tracker = HuMomentTracking(im_info, num_t=2, device="cpu")

    hu = np.array([[0.0, 1e-12, -1e-6]], dtype=np.float32)
    log_hu = tracker._log_hu(hu)

    assert np.all(np.isfinite(log_hu))
    assert np.isclose(log_hu[0, 0], 0.0)


def test_dense_sparse_matching_consistency():
    im_info = DummyImInfo()
    tracker = HuMomentTracking(im_info, num_t=2, max_distance_um=5.0, device="cpu", mode="dense")

    coords_pre = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=float)
    coords_post = coords_pre + np.array([[0.1, 0.0], [0.1, -0.1]], dtype=float)

    stats_pre = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    stats_post = stats_pre.copy()

    hu_pre = np.array(
        [
            [0.1, -0.2, 0.05, -0.1, 0.2, -0.3],
            [0.2, -0.1, 0.06, -0.2, 0.1, -0.4],
        ],
        dtype=np.float32,
    )
    hu_post = hu_pre.copy()

    cost_matrix = tracker._get_cost_matrix(
        coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre
    )
    row_dense, col_dense, _ = tracker._find_best_matches(cost_matrix)
    row_sparse, col_sparse, _ = tracker._match_frames_sparse(
        coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre
    )

    dense_pairs = set(zip(row_dense, col_dense))
    sparse_pairs = set(zip(row_sparse, col_sparse))
    assert dense_pairs == sparse_pairs
