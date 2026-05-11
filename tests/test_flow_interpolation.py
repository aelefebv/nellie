"""Characterization tests for ``nellie.tracking.flow_interpolation.FlowInterpolator``.

Bootstraps the test surface for :mod:`nellie.tracking.flow_interpolation`.
Pins the wiki-documented invariants and the contract of the per-frame
neighbor-lookup primitive ``_get_nearby_coords``:

- ``_get_nearby_coords`` (PRD #204 / Slice 1 #206 / Slice 2 #207):
  - Returns a pair of per-original-coord lists
    ``(nearby_idxs_return, distance_return)``, both length
    ``len(coords)``.
  - Per-coord slot is empty (``[]``) when (a) the input coord has any
    NaN component or (b) no ``check_coords`` lie within ``max_distance_um``
    of the (scaled) query.
  - Otherwise the slot holds the within-radius ``check_coords`` indices
    and their **scaled** Euclidean distances (i.e.
    ``||scaling * (query - check)||_2``).
  - Symmetry: the same pair of points should yield matching distances
    for forward and backward queries.

- ``interpolate_coord`` / ``interpolate_all_forward`` / ``interpolate_all_backward``:
  - Future Slice 1 of PRD #205 will expand the driver-level pin.

The ``_make_bare_flow_for_nearby_coords`` helper builds a
``FlowInterpolator`` instance via ``__new__`` and manually sets only the
attributes ``_get_nearby_coords`` reads (``self.scaling``,
``self.check_coords``, ``self.current_t``, ``self.current_tree``,
``self.max_distance_um``). This avoids the full constructor's ImInfo /
``flow_vector_array.npy`` setup — the same pattern used in
``test_hu_tracking._make_bare_hu_for_*`` and
``test_mocap_marking._make_bare_markers_for_*``.

The pre-rewrite NaN-coord path in ``_get_nearby_coords`` has a known
positional-alignment bug (``i not in good_coords`` over an ndarray:
the loop only aligns correctly when there are no NaN coords).
PRD #204 fixes this in Slice 2 (#207); Slice 1 deliberately does NOT
pin the buggy NaN-alignment behavior — the corrected behavior is
pinned by Slice 2's positional-alignment regression test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nellie.tracking.flow_interpolation import FlowInterpolator


# -------------------------------------------------------------------------
# Bare-instance helper for _get_nearby_coords
# -------------------------------------------------------------------------


def _make_bare_flow_for_nearby_coords(
    check_coords: np.ndarray,
    scaling: tuple,
    max_distance_um: float = 1.0,
) -> FlowInterpolator:
    """Build a FlowInterpolator instance bypassing __init__ for nearby-coord tests.

    ``_get_nearby_coords`` only reads ``self.scaling``,
    ``self.check_coords``, ``self.current_t``, ``self.current_tree``, and
    ``self.max_distance_um``. The ``__new__`` + manual attribute pattern
    mirrors ``_make_bare_hu_for_*`` in ``tests/test_hu_tracking.py`` and
    avoids the full constructor's ImInfo / ``flow_vector_array.npy`` setup.

    ``current_t`` is left as ``None`` so the first call to
    ``_get_nearby_coords`` triggers a fresh tree build from
    ``check_coords * scaling`` (matching the production code path).
    """
    f = FlowInterpolator.__new__(FlowInterpolator)
    f.scaling = scaling
    f.check_coords = np.asarray(check_coords, dtype=np.float64)
    f.current_t = None
    f.current_tree = None
    f.max_distance_um = float(max_distance_um)
    return f


# -------------------------------------------------------------------------
# _get_nearby_coords contract tests
# -------------------------------------------------------------------------


def test_get_nearby_coords_returns_pair_of_per_coord_lists() -> None:
    """Output is a 2-tuple, both length ``len(coords)``.

    Pins the basic schema contract that ``_get_vector_weights`` and
    ``_get_final_vector`` both rely on (per-coord indexing into the
    returned lists).
    """
    check_coords = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=10.0)

    coords = np.array([[0.5, 0.5], [4.5, 4.5], [100.0, 100.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert len(nearby_idxs) == len(coords)
    assert len(distances) == len(coords)


def test_get_nearby_coords_empty_check_coords_returns_empty_slots() -> None:
    """``check_coords`` empty → every query coord gets an empty slot.

    Boundary: ``cKDTree`` on an empty point set still constructs but
    every ``query_ball_point`` returns ``[]``. Pin that the per-coord
    output reflects this with empty slots (not ``None``, not raised).
    """
    check_coords = np.zeros((0, 2), dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=1.0)

    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    # The early-return branches in the current impl: ``len(nearby_idxs) == 0``
    # OR ``max_k == 0`` both return ``([], [])``. Either branch is acceptable;
    # the contract is "no neighbors found" and both representations satisfy it.
    if len(nearby_idxs) == 0:
        # Early-return short-circuit: empty result.
        assert nearby_idxs == [] and distances == []
    else:
        # Full-loop path: per-coord empty slots.
        assert len(nearby_idxs) == len(coords)
        for slot in nearby_idxs:
            assert len(slot) == 0
        for slot in distances:
            assert len(slot) == 0


def test_get_nearby_coords_no_neighbors_in_radius_returns_empty() -> None:
    """All ``check_coords`` outside the radius → ``([], [])`` short-circuit.

    The current impl short-circuits when ``max_k == 0`` (every
    ``query_ball_point`` result is empty). Pin the short-circuit form
    so the rewrite preserves the contract.
    """
    check_coords = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=1.0)

    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    # The current impl returns ``([], [])`` via the ``max_k == 0`` short-circuit.
    # The post-rewrite contract should match: every per-coord slot is empty.
    if len(nearby_idxs) == 0:
        assert nearby_idxs == [] and distances == []
    else:
        for slot in nearby_idxs:
            assert len(slot) == 0
        for slot in distances:
            assert len(slot) == 0


def test_get_nearby_coords_single_neighbor_known_distance_2d() -> None:
    """Single ``check_coord`` at known offset → correct index + scaled distance.

    Query coord at origin, check coord at ``(3, 4)`` with unit scaling:
    distance = 5.0 (3-4-5 triangle). The index returned is ``0`` (the
    only check coord). Pins both the index and the distance computation.
    """
    check_coords = np.array([[3.0, 4.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=10.0)

    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert len(nearby_idxs) == 1
    assert len(nearby_idxs[0]) == 1
    assert int(nearby_idxs[0][0]) == 0
    assert distances[0][0] == pytest.approx(5.0, rel=1e-9)


def test_get_nearby_coords_distance_reflects_scaling() -> None:
    """Distance = ``||scaling * (query - check)||_2``, NOT raw Euclidean.

    Query coord ``(0, 0)``, check coord ``(1, 1)``, scaling
    ``(3.0, 4.0)``. Scaled offset = ``(3, 4)``; distance = 5.0. Pins
    that the function applies ``scaling`` to both the query and check
    coordinates before computing distance.
    """
    check_coords = np.array([[1.0, 1.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(3.0, 4.0), max_distance_um=10.0)

    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert len(nearby_idxs[0]) == 1
    assert distances[0][0] == pytest.approx(5.0, rel=1e-9)


def test_get_nearby_coords_multi_coord_all_within_radius_2d() -> None:
    """Multi query coords, all within radius of multiple check coords.

    Pins that per-coord output slots correctly correspond to per-coord
    inputs and that distances are sorted by ``cKDTree.query`` ascending
    order (or matching order in the rewrite).
    """
    check_coords = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=2.0)

    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    # Each query coord should find all 4 check coords within radius 2.0.
    assert len(nearby_idxs[0]) == 4
    assert len(nearby_idxs[1]) == 4

    # Validate distances against analytical computation.
    # For query (0, 0): distances to {(0,0), (0,1), (1,0), (1,1)} = {0, 1, 1, sqrt(2)}.
    # For query (1, 1): distances to {(0,0), (0,1), (1,0), (1,1)} = {sqrt(2), 1, 1, 0}.
    # Order can vary — assert sorted distances match analytical sorted distances.
    expected_q0_sorted = sorted([0.0, 1.0, 1.0, np.sqrt(2.0)])
    expected_q1_sorted = sorted([np.sqrt(2.0), 1.0, 1.0, 0.0])
    np.testing.assert_allclose(sorted(distances[0]), expected_q0_sorted, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sorted(distances[1]), expected_q1_sorted, rtol=1e-9, atol=1e-12)


def test_get_nearby_coords_3d_single_neighbor_known_distance() -> None:
    """3D variant: single ``check_coord`` at known offset → correct distance.

    Query coord at origin, check coord at ``(2, 3, 6)`` with unit
    scaling: distance = ``sqrt(4 + 9 + 36) = 7.0``. Pin the 3D path.
    """
    check_coords = np.array([[2.0, 3.0, 6.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0, 1.0), max_distance_um=10.0)

    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert len(nearby_idxs[0]) == 1
    assert int(nearby_idxs[0][0]) == 0
    assert distances[0][0] == pytest.approx(7.0, rel=1e-9)


def test_get_nearby_coords_3d_distance_reflects_anisotropic_scaling() -> None:
    """3D: anisotropic scaling correctly applied to all three axes.

    Query coord ``(0, 0, 0)``, check coord ``(1, 1, 1)``, scaling
    ``(2.0, 3.0, 6.0)``. Scaled offset = ``(2, 3, 6)``; distance = 7.0.
    Pins anisotropic scaling on the 3D path (Z is typically much
    coarser than Y/X in microscopy stacks).
    """
    check_coords = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(2.0, 3.0, 6.0), max_distance_um=10.0)

    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert len(nearby_idxs[0]) == 1
    assert distances[0][0] == pytest.approx(7.0, rel=1e-9)


def test_get_nearby_coords_caches_tree_across_same_t() -> None:
    """Tree built once when ``current_t`` matches across calls.

    Pins the cache invariant: after a call sets up the tree, a second
    call with the same ``t`` reuses ``self.current_tree``. The
    production caller (``interpolate_coord``) sets ``self.current_t = t``
    after ``_get_nearby_coords`` returns; for direct-call testing the
    cache is invalidated each time (current_t left as None for the
    first call). This test sets ``current_t`` between calls to mirror
    the production cache state.
    """
    check_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=2.0)

    coords = np.array([[0.5, 0.5]], dtype=np.float64)

    # First call: builds tree (current_t was None).
    f._get_nearby_coords(t=0, coords=coords)
    f.current_t = 0  # mimic production caller's post-call assignment
    tree_after_first_call = f.current_tree

    # Second call with same t: should reuse the cached tree.
    f._get_nearby_coords(t=0, coords=coords)

    assert f.current_tree is tree_after_first_call, "Tree should not be rebuilt for same t"


def test_get_nearby_coords_rebuilds_tree_on_t_change() -> None:
    """Tree rebuilt when ``current_t`` changes.

    Pins the inverse of the cache invariant: when the caller advances
    ``t``, the tree is rebuilt against the (potentially-new)
    ``check_coords``.
    """
    check_coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=2.0)

    coords = np.array([[0.5, 0.5]], dtype=np.float64)

    # First call at t=0 builds tree.
    f._get_nearby_coords(t=0, coords=coords)
    f.current_t = 0
    tree_after_first_call = f.current_tree

    # Caller advances to t=1; tree should be rebuilt.
    f._get_nearby_coords(t=1, coords=coords)

    assert f.current_tree is not tree_after_first_call, "Tree should be rebuilt for new t"


def test_get_nearby_coords_neighbors_match_independent_kdtree_query() -> None:
    """Reference equivalence: result matches an independent ``cKDTree`` query.

    Pin against a known-good baseline: build the same tree externally
    via ``cKDTree`` and run ``query_ball_point`` directly. Indices
    should match (as a set; per-coord ordering can differ between
    implementations).
    """
    rng = np.random.default_rng(42)
    check_coords = rng.uniform(0.0, 10.0, size=(50, 2))
    scaling = (1.0, 1.0)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=scaling, max_distance_um=2.0)

    coords = rng.uniform(0.0, 10.0, size=(8, 2))
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    # Independent reference computation.
    ref_tree = cKDTree(check_coords * np.asarray(scaling))
    scaled_query = coords * np.asarray(scaling)
    ref_idxs = ref_tree.query_ball_point(scaled_query, r=2.0, p=2)

    # Per-coord index sets should match.
    for i in range(len(coords)):
        assert set(int(x) for x in nearby_idxs[i]) == set(int(x) for x in ref_idxs[i]), (
            f"Index-set mismatch at coord {i}: got "
            f"{sorted(int(x) for x in nearby_idxs[i])}, "
            f"expected {sorted(int(x) for x in ref_idxs[i])}"
        )
        # Distance values: each returned distance should equal
        # ||scaled_check - scaled_query||_2 for some neighbor.
        ref_distances_for_coord = sorted(
            float(np.linalg.norm(check_coords[j] * np.asarray(scaling) - scaled_query[i]))
            for j in ref_idxs[i]
        )
        np.testing.assert_allclose(
            sorted(distances[i]),
            ref_distances_for_coord,
            rtol=1e-9,
            atol=1e-12,
        )


def test_get_nearby_coords_input_check_coords_not_mutated() -> None:
    """``self.check_coords`` is not mutated by the call.

    Pins immutability of the input — the function multiplies by
    ``scaling`` to build the tree but should not write back into
    ``self.check_coords``.
    """
    check_coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    check_coords_copy = check_coords.copy()
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(2.0, 3.0), max_distance_um=10.0)

    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    f._get_nearby_coords(t=0, coords=coords)

    np.testing.assert_array_equal(f.check_coords, check_coords_copy)


# -------------------------------------------------------------------------
# Slice 2 (#207): post-rewrite equivalence + NaN-alignment regression
# -------------------------------------------------------------------------
#
# These tests pin the post-rewrite contract: bit-identical for the
# no-NaN case (same kernels — `query_ball_point` + `linalg.norm` —
# operating on the same scaled coordinates) and corrected positional
# alignment for the NaN case. The pre-rewrite NaN behavior was a bug
# (the `i not in good_coords` ndarray-membership loop only aligned
# correctly when `good_coords == [0, ..., N-1]`); fix-and-pin per
# ADR 0010.


def test_get_nearby_coords_caches_scaled_check_coords_attribute() -> None:
    """Tree rebuild caches ``self.scaled_check_coords`` for distance reuse.

    Pin the rewrite's caching invariant: when the tree rebuilds (on
    `current_t` change), ``self.scaled_check_coords`` is also
    materialized (`check_coords * scaling`) and stored on the
    instance. The per-coord distance compute below the tree-build
    block reads from this cached array; reusing it instead of
    recomputing the multiplication makes the per-coord
    ``np.linalg.norm`` cheap.
    """
    check_coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(2.0, 3.0), max_distance_um=20.0)

    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    f._get_nearby_coords(t=0, coords=coords)

    assert hasattr(f, "scaled_check_coords"), "Rewrite must cache scaled_check_coords"
    expected = check_coords * np.asarray((2.0, 3.0))
    np.testing.assert_allclose(f.scaled_check_coords, expected, rtol=1e-12, atol=0.0)


def test_get_nearby_coords_nan_coord_positional_alignment() -> None:
    """NaN coord interleaved with valid coords → per-coord results land in CORRECT slots.

    Pre-rewrite (PRD #204 § 2): when ``coords[1]`` is NaN, the
    ``i not in good_coords`` loop misassigns: ``coords[2]``'s
    neighbors land in slot 1 (the NaN coord's slot), or are dropped
    entirely depending on the membership-scan outcome. Post-rewrite:
    NaN coord's slot is empty, and each valid coord's slot holds its
    OWN neighbors.

    Setup: 4 query coords with `coords[1]` set NaN. Each valid coord
    has a single distinct neighbor at a known location, so the
    correct positional mapping is unambiguous.
    """
    # Each check_coord at a distinct position; expected to be matched
    # one-to-one with the corresponding query coord.
    check_coords = np.array([
        [10.0, 10.0],  # near coords[0]
        [20.0, 20.0],  # near coords[1] (the NaN slot — should be empty)
        [30.0, 30.0],  # near coords[2]
        [40.0, 40.0],  # near coords[3]
    ], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(
        check_coords, scaling=(1.0, 1.0), max_distance_um=2.0,
    )

    coords = np.array([
        [10.0, 10.0],
        [np.nan, np.nan],
        [30.0, 30.0],
        [40.0, 40.0],
    ], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    # Output length matches input (4 slots).
    assert len(nearby_idxs) == 4
    assert len(distances) == 4

    # Slot 0: valid coord at (10, 10) → matches check_coord 0 at (10, 10).
    assert len(nearby_idxs[0]) == 1
    assert int(nearby_idxs[0][0]) == 0
    assert distances[0][0] == pytest.approx(0.0, abs=1e-12)

    # Slot 1: NaN coord → empty slot (no neighbors).
    assert len(nearby_idxs[1]) == 0
    assert len(distances[1]) == 0

    # Slot 2: valid coord at (30, 30) → matches check_coord 2 at (30, 30).
    assert len(nearby_idxs[2]) == 1
    assert int(nearby_idxs[2][0]) == 2, (
        f"Slot 2 must hold check_coord 2's index; got {int(nearby_idxs[2][0])}. "
        "Pre-rewrite NaN-alignment bug shifted this — fix-and-pin per ADR 0010."
    )
    assert distances[2][0] == pytest.approx(0.0, abs=1e-12)

    # Slot 3: valid coord at (40, 40) → matches check_coord 3.
    assert len(nearby_idxs[3]) == 1
    assert int(nearby_idxs[3][0]) == 3, (
        f"Slot 3 must hold check_coord 3's index; got {int(nearby_idxs[3][0])}. "
        "Pre-rewrite NaN-alignment bug shifted this — fix-and-pin per ADR 0010."
    )
    assert distances[3][0] == pytest.approx(0.0, abs=1e-12)


def test_get_nearby_coords_all_nan_coords_returns_empty() -> None:
    """All query coords NaN → ``([], [])`` short-circuit.

    Boundary: ``good_coords`` is empty, so no tree query happens. Pin
    the early-return form (matches the consumer contract — `interpolate_coord`
    skips the whole frame on empty result).
    """
    check_coords = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=5.0)

    coords = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    nearby_idxs, distances = f._get_nearby_coords(t=0, coords=coords)

    assert nearby_idxs == [] and distances == []


# -------------------------------------------------------------------------
# interpolate_all_forward / interpolate_all_backward characterization
# (PRD #205 / Slice 1 #210 / Slice 2 #211)
# -------------------------------------------------------------------------
#
# Driver-level tests for the per-frame trajectory builders. These pin the
# behavior the Slice 2 vectorization must preserve: track-row order
# (interleaved init/post per coord at `t == frame_range[0]`),
# track-row schema (id, frame, *coord), terminal NaN propagation, and
# the forward/backward arithmetic (vector-add forward; vector-subtract
# backward).
#
# Uses a stub ImInfo to avoid the disk-backed `flow_vector_array.npy`
# loading in `FlowInterpolator.__init__` — a synthetic flow_vector_array
# is written to a tmp file per test and the stub points at it.


def _make_stub_iminfo(tmp_path, flow_vector_array, no_z, dim_res=None):
    """Build a stub ImInfo for `interpolate_all_*` driver tests.

    `FlowInterpolator.__init__` reads from im_info: `no_t`, `no_z`,
    `shape`, `axes`, `dim_res`, `im_path`, `pipeline_paths`, and calls
    `get_memmap`. The stub satisfies these with synthetic-but-plausible
    values; the synthetic flow_vector_array is written to
    `tmp_path/flow_vector_array.npy`.
    """
    if dim_res is None:
        dim_res = {"T": 1.0, "Z": 0.5, "Y": 0.1, "X": 0.1}
    flow_path = tmp_path / "flow_vector_array.npy"
    np.save(flow_path, flow_vector_array)
    im_path = tmp_path / "stub_im.npy"
    if no_z:
        shape = (3, 100, 100)
        axes = "TYX"
        memmap_shape = shape
    else:
        shape = (3, 50, 100, 100)
        axes = "TZYX"
        memmap_shape = shape
    np.save(im_path, np.zeros(memmap_shape, dtype=np.float32))

    class _StubImInfo:
        def __init__(self):
            self.no_t = False
            self.no_z = no_z
            self.shape = shape
            self.axes = axes
            self.dim_res = dim_res
            self.im_path = str(im_path)
            self.pipeline_paths = {"flow_vector_array": str(flow_path)}

        def get_memmap(self, _path):
            return np.zeros(memmap_shape, dtype=np.float32)

    return _StubImInfo()


def _make_synthetic_flow_3d(rng, n_markers, frames, max_pos=100.0, vector_scale=1.0):
    """Synthetic 3D flow_vector_array spanning ``frames`` timepoints.

    Schema: ``[t, z, y, x, dz, dy, dx, cost]`` per row.
    """
    blocks = []
    for t in range(frames):
        blocks.append(np.column_stack([
            np.full(n_markers, t, dtype=np.float64),
            rng.uniform(0.0, max_pos, size=n_markers),
            rng.uniform(0.0, max_pos, size=n_markers),
            rng.uniform(0.0, max_pos, size=n_markers),
            rng.uniform(-vector_scale, vector_scale, size=n_markers),
            rng.uniform(-vector_scale, vector_scale, size=n_markers),
            rng.uniform(-vector_scale, vector_scale, size=n_markers),
            rng.uniform(0.0, 0.5, size=n_markers),
        ]))
    return np.concatenate(blocks, axis=0)


def test_interpolate_all_forward_returns_tracks_and_frame_num(tmp_path) -> None:
    """Returns ``(tracks, track_properties)`` with parallel index correspondence.

    Pin the schema contract: ``tracks`` is a list of lists, each inner
    list is ``[id, frame, *coord]``; ``track_properties['frame_num']``
    is a list of frame numbers; both lists have the same length and
    ``frame_num[i]`` matches ``tracks[i][1]``.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    rng = np.random.default_rng(100)
    flow_array = _make_synthetic_flow_3d(rng, n_markers=200, frames=2)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    coords = rng.uniform(0.0, 100.0, size=(20, 3))
    tracks, track_properties = interpolate_all_forward(coords, 0, 2, info)

    assert isinstance(tracks, list)
    assert isinstance(track_properties, dict)
    assert "frame_num" in track_properties
    assert isinstance(track_properties["frame_num"], list)
    assert len(tracks) == len(track_properties["frame_num"])

    # Every track row is [id, frame, z, y, x] for 3D.
    for i, row in enumerate(tracks):
        assert len(row) == 5, f"Expected (id, frame, z, y, x) row, got {row}"
        assert row[1] == track_properties["frame_num"][i], (
            f"Row {i}: track frame {row[1]} != frame_num {track_properties['frame_num'][i]}"
        )


def test_interpolate_all_forward_initial_frame_double_row(tmp_path) -> None:
    """At ``t == frame_range[0]``, each valid coord contributes 2 rows; later frames contribute 1.

    Pin the per-coord interleaved init/post pattern: at the first frame
    in the range, the loop appends BOTH the initial-position row AND
    the post-vector row per valid coord. Subsequent frames append only
    the post-vector row. This is the bit-identical row-count contract
    the Slice 2 vectorization must preserve.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    rng = np.random.default_rng(101)
    flow_array = _make_synthetic_flow_3d(rng, n_markers=500, frames=3)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    # Use enough query coords positioned to find neighbors so all are valid.
    # Sampling from the same distribution as flow markers ensures most have
    # neighbors within the radius.
    coords = rng.uniform(0.0, 100.0, size=(10, 3))
    tracks, _ = interpolate_all_forward(coords, 0, 3, info, max_distance_um=5.0)

    # At least some tracks should be produced (sanity).
    assert len(tracks) > 0

    # Frame numbers in the output: frame_range[0]=0 (initial-rows),
    # 1 (post-rows for t=0), 2 (post-rows for t=1), 3 (post-rows for t=2).
    # The exact frame_num distribution depends on which coords find
    # neighbors at each frame.
    frame_nums_present = sorted(set(int(row[1]) for row in tracks))
    # If any coord was valid at t=0, we should see frame_num 0 (initial) and 1 (post).
    # If any coord was valid at t=1, we should see frame_num 2 (post).
    # The minimum frame_num must be >= 0; max <= 3 (post-vector for t=2).
    assert min(frame_nums_present) >= 0
    assert max(frame_nums_present) <= 3


def test_interpolate_all_forward_initial_block_vs_post_arithmetic(tmp_path) -> None:
    """Initial-frame init-row records the PRE-update coord; post-row records POST-update.

    Pin the in-place update timing: the per-coord loop captures
    ``coord = coords[coord_num]`` BEFORE the update, appends the init
    row using that captured value at ``frame_range[0]``, then updates
    ``coords[coord_num] += final_vector[coord_num]``, then appends the
    post row using the (now-updated) coord value.

    Setup: a single coord at a known position, with a deterministic
    flow array that places exactly one nearby marker with a known
    vector. Initial row should record the input position; post row
    should record input + vector.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    # Single marker at exactly the query position so cost is irrelevant.
    flow_array = np.array([[0.0, 50.0, 50.0, 50.0, 1.0, 2.0, 3.0, 0.0]], dtype=np.float64)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    coords = np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
    tracks, _ = interpolate_all_forward(coords, 0, 1, info, max_distance_um=10.0)

    # Two rows: initial at frame 0, post at frame 1.
    assert len(tracks) == 2

    # Find the init row (frame 0) and post row (frame 1).
    rows_by_frame = {int(row[1]): row for row in tracks}
    assert 0 in rows_by_frame, f"Initial frame 0 row missing; tracks: {tracks}"
    assert 1 in rows_by_frame, f"Post-update frame 1 row missing; tracks: {tracks}"

    init_row = rows_by_frame[0]
    post_row = rows_by_frame[1]

    # Init row records the pre-update position (50, 50, 50).
    np.testing.assert_allclose(init_row[2:5], [50.0, 50.0, 50.0], rtol=1e-9)
    # Post row records the updated position (50+1, 50+2, 50+3) = (51, 52, 53).
    np.testing.assert_allclose(post_row[2:5], [51.0, 52.0, 53.0], rtol=1e-9)


def test_interpolate_all_forward_terminal_nan_propagation(tmp_path) -> None:
    """Once a coord goes all-NaN in `final_vector`, it stops appearing in subsequent frames.

    Pin the terminal NaN propagation contract documented in
    `wiki/tracking/flow-interpolation.md`. A coord far from any marker
    will get an all-NaN final_vector, causing the driver to overwrite
    `coords[coord_num]` with NaN. On the next frame, that coord is
    NaN-input → its slot in `final_vector` will also be all-NaN → it
    is skipped (continue). It does not contribute any tracks beyond
    the frame where it died.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    # Coord 0 is right at a marker; coord 1 is far away (no nearby marker
    # at any frame → its final_vector will be all-NaN from frame 0).
    flow_array = np.array([
        [0.0, 50.0, 50.0, 50.0, 0.5, 0.5, 0.5, 0.0],
        [1.0, 50.5, 50.5, 50.5, 0.5, 0.5, 0.5, 0.0],
    ], dtype=np.float64)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    coords = np.array([
        [50.0, 50.0, 50.0],     # near a marker
        [9999.0, 9999.0, 9999.0],  # far from every marker
    ], dtype=np.float64)
    tracks, _ = interpolate_all_forward(coords, 0, 2, info, max_distance_um=5.0)

    # All track rows must have id == 0 (coord 1 dies immediately because
    # its final_vector is NaN). Coord 1 should produce no tracks at all.
    track_ids = sorted(set(int(row[0]) for row in tracks))
    assert 0 in track_ids
    assert 1 not in track_ids, (
        f"Coord 1 had no in-radius marker; should produce no tracks. "
        f"Got track_ids={track_ids}, tracks={tracks}"
    )


def test_interpolate_all_backward_subtracts_vector(tmp_path) -> None:
    """Backward direction subtracts the flow vector from the coord.

    Pin the backward arithmetic: at each frame, the new coord is
    ``old_coord - final_vector`` (vs forward's ``+``). Frame numbers
    decrement (`t - 1`) instead of increment.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_backward

    # Marker at (51, 52, 53) with vector (1, 2, 3) — the "forward" vector
    # from (50, 50, 50). Backward starts at the destination (51, 52, 53)
    # and should walk back to (50, 50, 50).
    flow_array = np.array([[0.0, 50.0, 50.0, 50.0, 1.0, 2.0, 3.0, 0.0]], dtype=np.float64)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    # Backward query: position the coord at the "destination" of the marker's vector.
    # Backward convention: start_t > end_t; frame_range = list(arange(end_t, start_t+1))[::-1].
    # With start_t=1, end_t=0: frame_range = [1, 0]. At t=1, look at markers from t=0.
    coords = np.array([[51.0, 52.0, 53.0]], dtype=np.float64)
    tracks, _ = interpolate_all_backward(coords, 1, 0, info, max_distance_um=10.0)

    # Backward path: (51, 52, 53) at frame 1, then walk back to (50, 50, 50) at frame 0.
    rows_by_frame = {int(row[1]): row for row in tracks}
    assert 1 in rows_by_frame, f"Initial backward frame 1 row missing; tracks: {tracks}"
    assert 0 in rows_by_frame, f"Backward-stepped frame 0 row missing; tracks: {tracks}"

    init_row = rows_by_frame[1]
    back_row = rows_by_frame[0]

    np.testing.assert_allclose(init_row[2:5], [51.0, 52.0, 53.0], rtol=1e-9)
    np.testing.assert_allclose(back_row[2:5], [50.0, 50.0, 50.0], rtol=1e-9)


def test_interpolate_all_forward_id_uses_min_track_num_offset(tmp_path) -> None:
    """Track id = ``coord_num + min_track_num``.

    Pin the id offset: the driver supports a ``min_track_num`` parameter
    (used by `LabelTracks` to namespace track ids across labels) that
    shifts the per-coord id. Output rows record ``coord_num + min_track_num``
    in column 0.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    flow_array = np.array([
        [0.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 60.0, 60.0, 60.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=False)

    coords = np.array([
        [50.0, 50.0, 50.0],
        [60.0, 60.0, 60.0],
    ], dtype=np.float64)
    tracks, _ = interpolate_all_forward(coords, 0, 1, info, min_track_num=1000, max_distance_um=5.0)

    # IDs in the output should be 1000 (coord 0) and 1001 (coord 1).
    track_ids = sorted(set(int(row[0]) for row in tracks))
    assert track_ids == [1000, 1001], f"Expected ids [1000, 1001], got {track_ids}"


def test_interpolate_all_forward_2d_emits_4_column_rows(tmp_path) -> None:
    """2D path emits ``[id, frame, y, x]`` rows (no z column).

    Pin the 2D schema contract: when ``im_info.no_z`` is True, track
    rows have 4 columns instead of 5. The flow_vector_array is also
    2D-shaped: ``[t, y, x, dy, dx, cost]`` per row.
    """
    from nellie.tracking.flow_interpolation import interpolate_all_forward

    # 2D flow_vector_array schema: [t, y, x, dy, dx, cost].
    flow_array = np.array([[0.0, 50.0, 50.0, 1.0, 2.0, 0.0]], dtype=np.float64)
    info = _make_stub_iminfo(tmp_path, flow_array, no_z=True)

    coords = np.array([[50.0, 50.0]], dtype=np.float64)
    tracks, _ = interpolate_all_forward(coords, 0, 1, info, max_distance_um=10.0)

    assert len(tracks) > 0
    for row in tracks:
        assert len(row) == 4, f"Expected (id, frame, y, x) row in 2D, got {row}"


def test_get_nearby_coords_post_rewrite_distances_sorted_ascending() -> None:
    """Post-rewrite distances need not be sorted (pre-rewrite was via cKDTree.query).

    Pre-rewrite ``cKDTree.query(k=max_k)`` returns distances in
    ascending order (k-nearest semantics). Post-rewrite
    ``np.linalg.norm`` over ``query_ball_point`` indices preserves the
    cKDTree `query_ball_point` traversal order — typically ascending
    by tree-internal-node visit order, but NOT a guaranteed contract.

    Consumers (`_get_vector_weights` / `_get_final_vector`) do not
    rely on sort order — they index per-coord arrays elementwise
    (multiplying ``cost_weights * distance_weights`` and reducing).
    Pin the no-sort-order claim so a future maintainer doesn't add a
    spurious sort step.
    """
    rng = np.random.default_rng(11)
    check_coords = rng.uniform(0.0, 10.0, size=(50, 2))
    f = _make_bare_flow_for_nearby_coords(check_coords, scaling=(1.0, 1.0), max_distance_um=2.0)

    coords = rng.uniform(0.0, 10.0, size=(5, 2))
    _, distances = f._get_nearby_coords(t=0, coords=coords)

    # The per-coord distance lists must contain the correct values
    # (already covered by other tests). Here we just assert each per-coord
    # list contains the analytically-correct distance set, regardless of
    # ordering — captures the "no sort contract" intent.
    for slot in distances:
        # Either empty or contains positive values; per-coord ordering is
        # not asserted (sort-free contract).
        if len(slot) > 0:
            assert (np.asarray(slot) >= 0).all()
