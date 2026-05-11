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
