"""Opt-in performance tests for ``nellie.tracking.flow_interpolation``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`,
:mod:`tests.test_labelling_perf`, :mod:`tests.test_networking_perf`, and
:mod:`tests.test_hu_tracking_perf`:

1. **Hot-path microbenchmarks** decompose the per-frame interpolation
   cost into the dominant kernels surfaced by reading the source:
   - ``_get_nearby_coords`` scaling at N coords = 100 / 500 / 1000 — the
     per-frame neighbor-lookup primitive (PRD #204). Pre-rewrite this is
     a double `cKDTree` query (`query_ball_point` for counts +
     `query(k=max_k)` for distances); post-rewrite it is a single
     `query_ball_point` + per-coord `np.linalg.norm`. The wall-clock
     here drives the choice in PRD #204.
   - ``interpolate_coord`` end-to-end at N = 1000 — the per-frame
     entry point used by ``VoxelReassigner`` and ``LabelTracks``.
   - ``interpolate_all_forward`` end-to-end on a synthetic 3-frame
     flow array, N = 500 — the per-trajectory driver. The Python
     per-coord inner loop here drives the choice in PRD #205.

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions) — Filter's three assertions remain the only ones, and only
because each pinned a decision from a real perf pass; here no
flow_interpolation perf pass has happened yet, so assertions would be
speculative. The prints scaffold the future perf pass.

There is **no end-to-end ``FlowInterpolator.run()``** equivalent to the
HuMomentTracking baseline at the top of
:mod:`tests.test_hu_tracking_perf` — `FlowInterpolator` is a per-frame
helper, not a `run()`-style stage. The closest end-to-end shape is
``interpolate_all_forward`` (covered above). FlowInterpolator's
contribution to the pipeline-level wall-clock is captured indirectly by
the VoxelReassigner perf tests in :mod:`tests.test_voxel_reassignment_perf`
(VoxelReassigner constructs two FlowInterpolator instances unconditionally
and calls ``interpolate_coord`` twice per frame).

There is also no MPS variant — `FlowInterpolator` is a CPU-only path
(`scipy.spatial.cKDTree` and pure-numpy distance / weight math). It
does not use `xp` and would not benefit from MPS dispatch.
"""

from __future__ import annotations

import time
from statistics import median

import numpy as np
import pytest

from nellie.tracking.flow_interpolation import FlowInterpolator


pytestmark = pytest.mark.benchmark


def _time_call(fn, *, iters: int) -> float:
    """Median wall-clock of ``iters`` invocations, in seconds.

    Mirror of the helper in :mod:`tests.test_hu_tracking_perf`.
    """
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return median(samples)


def _make_bare_flow(check_coords: np.ndarray, scaling: tuple, max_distance_um: float = 1.0) -> FlowInterpolator:
    """Build a FlowInterpolator instance bypassing __init__ for perf microbenchmarks.

    Mirrors ``_make_bare_flow_for_nearby_coords`` in
    :mod:`tests.test_flow_interpolation` — sets only the attributes the
    benchmarked methods read. Avoids the full constructor's ImInfo /
    ``flow_vector_array.npy`` setup.
    """
    f = FlowInterpolator.__new__(FlowInterpolator)
    f.scaling = scaling
    f.check_coords = np.asarray(check_coords, dtype=np.float64)
    f.current_t = None
    f.current_tree = None
    f.max_distance_um = float(max_distance_um)
    return f


# -------------------------------------------------------------------------
# _get_nearby_coords scaling — PRD #204 hot path
# -------------------------------------------------------------------------


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_get_nearby_coords_scaling(n, capsys) -> None:
    """Print median wall-clock for ``_get_nearby_coords`` at N coords.

    Synthetic 3D scenario: random check_coords + query coords drawn from
    the same distribution so a meaningful fraction land within radius.
    Pre-rewrite calls ``query_ball_point`` then ``query(k=max_k)``;
    post-rewrite (PRD #204 / Slice 2 #207) calls ``query_ball_point``
    once + per-coord ``np.linalg.norm``. The wall-clock difference
    drives the rewrite decision.
    """
    rng = np.random.default_rng(0)
    # Spread points over a unit volume with anisotropic scaling typical of
    # 3D microscopy (Z coarser than Y/X).
    check_coords = rng.uniform(0.0, 100.0, size=(n, 3))
    f = _make_bare_flow(check_coords, scaling=(0.5, 0.1, 0.1), max_distance_um=2.0)

    coords = rng.uniform(0.0, 100.0, size=(n, 3))

    # Warm up: first call builds the tree (one-time cost).
    f._get_nearby_coords(t=0, coords=coords)
    f.current_t = 0  # avoid per-iter tree rebuild — match production cache state

    elapsed = _time_call(lambda: f._get_nearby_coords(t=0, coords=coords), iters=5)
    with capsys.disabled():
        print(
            f"\n[perf] _get_nearby_coords N={n:4d} (cached tree) median over 5: "
            f"{elapsed * 1000:7.2f} ms"
        )


# -------------------------------------------------------------------------
# interpolate_coord end-to-end — per-frame driver entry
# -------------------------------------------------------------------------


def _make_bare_flow_for_interpolate(
    flow_vector_array: np.ndarray,
    scaling: tuple,
    no_z: bool,
    max_distance_um: float = 1.0,
) -> FlowInterpolator:
    """Bare FlowInterpolator for ``interpolate_coord`` end-to-end perf.

    Adds ``self.flow_vector_array``, ``self.forward``, and a stub
    ``self.im_info`` exposing the ``no_z`` flag (the only ImInfo
    attribute ``interpolate_coord`` reads). Mirrors the bare-class
    pattern but with one more attribute set than
    ``_make_bare_flow``.
    """
    f = FlowInterpolator.__new__(FlowInterpolator)
    f.scaling = scaling
    f.flow_vector_array = np.asarray(flow_vector_array, dtype=np.float64)
    f.forward = True
    f.current_t = None
    f.check_rows = None
    f.check_coords = None
    f.current_tree = None
    f.max_distance_um = float(max_distance_um)
    # Mirror `_allocate_memory`'s pre-bucketing — required by the
    # post-cleanup `interpolate_coord` lookup (#214).
    if f.flow_vector_array.size:
        t_col = f.flow_vector_array[:, 0]
        unique_t, inverse = np.unique(t_col, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        sorted_inverse = inverse[order]
        split_at = np.searchsorted(sorted_inverse, np.arange(1, len(unique_t)))
        grouped = np.split(order, split_at)
        f._t_to_rows = {int(t_val): rows for t_val, rows in zip(unique_t, grouped)}
    else:
        f._t_to_rows = {}

    class _StubImInfo:
        pass

    info = _StubImInfo()
    info.no_z = no_z  # type: ignore[attr-defined]
    f.im_info = info  # type: ignore[assignment]
    return f


def test_interpolate_coord_end_to_end(capsys) -> None:
    """Print median wall-clock for ``interpolate_coord`` at N=1000.

    Synthetic 3D flow_vector_array with one timepoint of marker matches.
    Pre- and post-rewrite path both run end-to-end: tree-build,
    neighbor lookup, weight compute, weighted-average. Captures the
    cumulative per-frame cost of the FlowInterpolator entry point used
    by ``VoxelReassigner`` and ``LabelTracks``.
    """
    rng = np.random.default_rng(1)
    n_markers = 1000
    # flow_vector_array 3D schema: [t, z, y, x, dz, dy, dx, cost]
    flow_vector_array = np.column_stack([
        np.zeros(n_markers),  # t
        rng.uniform(0.0, 100.0, size=n_markers),  # z (pre)
        rng.uniform(0.0, 100.0, size=n_markers),  # y (pre)
        rng.uniform(0.0, 100.0, size=n_markers),  # x (pre)
        rng.uniform(-1.0, 1.0, size=n_markers),  # dz
        rng.uniform(-1.0, 1.0, size=n_markers),  # dy
        rng.uniform(-1.0, 1.0, size=n_markers),  # dx
        rng.uniform(0.0, 0.5, size=n_markers),  # cost
    ])
    f = _make_bare_flow_for_interpolate(
        flow_vector_array, scaling=(0.5, 0.1, 0.1), no_z=False, max_distance_um=2.0,
    )

    # Query coords: same distribution so a meaningful fraction find neighbors.
    coords = rng.uniform(0.0, 100.0, size=(n_markers, 3))

    # Warm up.
    f.current_t = None
    f.interpolate_coord(coords, t=0)

    def _one_call() -> None:
        # Reset per-call cache so we measure the fresh-frame cost.
        f.current_t = None
        f.interpolate_coord(coords, t=0)

    elapsed = _time_call(_one_call, iters=3)
    with capsys.disabled():
        print(
            f"\n[perf] interpolate_coord N={n_markers} 3D (fresh cache) "
            f"median over 3: {elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# interpolate_all_forward end-to-end — per-trajectory driver — PRD #205 hot path
# -------------------------------------------------------------------------


@pytest.mark.parametrize("n_coords", [100, 500])
def test_interpolate_all_forward_end_to_end(n_coords, capsys, tmp_path) -> None:
    """Print median wall-clock for ``interpolate_all_forward`` at N coords.

    Synthetic 3D flow_vector_array spanning 3 timepoints. The driver
    iterates t in `[start_t, end_t)`, and per t iterates over each
    coord with a Python loop that computes the new coord position and
    appends to a Python list (PRD #205's per-coord vectorization
    target). The wall-clock here drives the choice in PRD #205.

    Builds a real ``FlowInterpolator`` via ``__init__`` (not
    ``__new__``) because ``interpolate_all_forward`` is a module-level
    function that reconstructs a FlowInterpolator internally. Writes
    the synthetic flow array to ``tmp_path`` and stubs an ImInfo
    pointing at it.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    rng = np.random.default_rng(2)
    n_markers = max(n_coords * 2, 200)
    flow_per_t = []
    for t in range(2):  # need t=0 and t=1 for end_t=2 driver
        flow_per_t.append(np.column_stack([
            np.full(n_markers, t, dtype=np.float64),  # t
            rng.uniform(0.0, 100.0, size=n_markers),
            rng.uniform(0.0, 100.0, size=n_markers),
            rng.uniform(0.0, 100.0, size=n_markers),
            rng.uniform(-1.0, 1.0, size=n_markers),
            rng.uniform(-1.0, 1.0, size=n_markers),
            rng.uniform(-1.0, 1.0, size=n_markers),
            rng.uniform(0.0, 0.5, size=n_markers),
        ]))
    flow_vector_array = np.concatenate(flow_per_t, axis=0)
    flow_path = tmp_path / "flow_vector_array.npy"
    np.save(flow_path, flow_vector_array)

    # Stub ImInfo: minimum surface area for FlowInterpolator __init__.
    im_path = tmp_path / "stub_im.npy"
    np.save(im_path, np.zeros((3, 50, 100, 100), dtype=np.float32))

    class _StubImInfo:
        def __init__(self):
            self.no_t = False
            self.no_z = False
            self.shape = (3, 50, 100, 100)
            self.axes = "TZYX"
            self.dim_res = {"T": 1.0, "Z": 0.5, "Y": 0.1, "X": 0.1}
            self.im_path = str(im_path)
            self.pipeline_paths = {"flow_vector_array": str(flow_path)}

        def get_memmap(self, _path):
            return np.zeros((3, 50, 100, 100), dtype=np.float32)

    info = _StubImInfo()
    coords = rng.uniform(0.0, 100.0, size=(n_coords, 3))

    # Warm up.
    interpolate_all_forward(coords.copy(), 0, 2, info)

    def _one_call() -> None:
        interpolate_all_forward(coords.copy(), 0, 2, info)

    elapsed = _time_call(_one_call, iters=3)
    with capsys.disabled():
        print(
            f"\n[perf] interpolate_all_forward N={n_coords:4d} 3D (3 frames) "
            f"median over 3: {elapsed * 1000:7.2f} ms"
        )
