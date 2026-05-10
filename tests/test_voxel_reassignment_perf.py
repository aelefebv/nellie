"""Opt-in performance tests for ``nellie.tracking.voxel_reassignment.VoxelReassigner``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`,
:mod:`tests.test_labelling_perf`, :mod:`tests.test_networking_perf`,
and :mod:`tests.test_hu_tracking_perf`:

1. **End-to-end CPU baseline** prints a VoxelReassigner wall-clock
   baseline so a future regression on the yeast fixture is visible
   relative to a prior commit.
2. **Hot-path microbenchmarks** decompose the per-frame cost into the
   three suspected hotspots surfaced by reading the code:
   - ``_assign_unique_matches`` Python-loop greedy selection (N = 100 /
     1k / 10k synthetic candidates) — strong vectorization suspect.
   - KDTree build vs query split for typical N — pin which dominates.
   - Brute-force vs KDTree at the dispatch threshold (controlled by
     ``max_bruteforce_pairs``) — pin the dispatch decision.

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no perf pass has happened yet,
so assertions would be speculative. The prints scaffold the future
perf pass — they show which inner op dominates, so an author can
target the right hot-path rather than instrumenting from scratch.

**No MPS variant.** VoxelReassigner is not MPS-onboarded (per
``wiki/now.md`` and ``wiki/gpu-runtime.md``: only filtering, labelling,
networking, and hu_tracking are). Most of VoxelReassigner's work runs
as pure-numpy regardless of the backend (KDTree is scipy-CPU; the
GPU branches are CUDA-only via ``_get_gpu_kdtree_cls``); ``device="mps"``
would set the backend identifier without changing what actually runs,
so an MPS perf number would be misleading. When VoxelReassigner is
onboarded, add an MPS variant.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nellie.tracking.voxel_reassignment import VoxelReassigner, VoxelReassignerConfig


pytestmark = pytest.mark.benchmark

_CPU = VoxelReassignerConfig(device="cpu")


def _release_voxel_reassign(v: VoxelReassigner) -> None:
    v.branch_label_memmap = None
    v.obj_label_memmap = None
    v.reassigned_branch_memmap = None
    v.reassigned_obj_memmap = None
    gc.collect()


def _time_call(fn, *, iters: int) -> float:
    """Median wall-clock of ``iters`` invocations, in seconds."""
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return median(samples)


# -------------------------------------------------------------------------
# End-to-end VoxelReassigner baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_voxel_reassign_run_baseline_prints_wall_clock(
    dim,
    make_voxel_reassign_imageinfo_2d,
    make_voxel_reassign_imageinfo_3d,
    capsys,
) -> None:
    """Print a VoxelReassigner wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates ``im_instance_label`` + ``im_skel_relabelled`` +
    ``flow_vector_array.npy`` from session caches, so this only pays
    the VoxelReassigner cost — not Filter+Label+Network+Markers+Hu on
    top).
    """
    factory = (
        make_voxel_reassign_imageinfo_2d
        if dim == "2d"
        else make_voxel_reassign_imageinfo_3d
    )

    def _one_run() -> None:
        info = factory()
        v = VoxelReassigner(info, _CPU, num_t=2)
        v.run()
        _release_voxel_reassign(v)

    # Warm up: first run pays one-time costs (memmap setup, KDTree caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] VoxelReassigner.run() {dim} median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational)
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def voxel_reassign_setup(
    request,
    make_voxel_reassign_imageinfo_2d,
    make_voxel_reassign_imageinfo_3d,
):
    """A VoxelReassigner with `_allocate_memory()` called.

    Required so ``self.spatial_shape`` is populated for
    ``_assign_unique_matches`` (which uses ``np.ravel_multi_index``).
    """
    factory = (
        make_voxel_reassign_imageinfo_2d
        if request.param == "2d"
        else make_voxel_reassign_imageinfo_3d
    )
    info = factory()
    v = VoxelReassigner(info, _CPU, num_t=2)
    v._allocate_memory()

    yield v, request.param

    _release_voxel_reassign(v)


def _synthesize_match_arrays(spatial_shape, n: int, rng: np.random.Generator):
    """Random voxel coords + matching distances of size n."""
    # Sample with replacement is fine — the greedy loop's purpose is to
    # de-duplicate. Bound coords inside spatial_shape so ravel_multi_index
    # doesn't choke.
    vox_prev = np.column_stack([
        rng.integers(0, dim, size=n) for dim in spatial_shape
    ]).astype(np.int64)
    vox_next = np.column_stack([
        rng.integers(0, dim, size=n) for dim in spatial_shape
    ]).astype(np.int64)
    distances = rng.uniform(0.0, 5.0, size=n).astype(np.float64)
    return vox_prev, vox_next, distances


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_assign_unique_matches_python_loop_scaling(
    voxel_reassign_setup, n, capsys
) -> None:
    """`_assign_unique_matches` Python-loop scaling at N = 100, 1k, 10k.

    The greedy 1-to-1 selection iterates ``range(len(order))`` in
    Python — this is the strongest vectorization suspect in the stage.
    Print wall-clock per-N to make the scaling obvious without
    instrumenting from scratch.
    """
    v, dim = voxel_reassign_setup
    rng = np.random.default_rng(42)
    vox_prev, vox_next, distances = _synthesize_match_arrays(
        v.spatial_shape, n, rng
    )

    # Warm up
    v._assign_unique_matches(vox_prev, vox_next, distances)

    elapsed = _time_call(
        lambda: v._assign_unique_matches(vox_prev, vox_next, distances),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] VoxelReassigner._assign_unique_matches {dim} "
            f"N={n} median over 5: {elapsed * 1000:.1f} ms"
        )


@pytest.mark.parametrize("n", [1000, 10000])
def test_kdtree_build_vs_query(voxel_reassign_setup, n, capsys) -> None:
    """Decompose the typical-N CPU path: cKDTree build vs query.

    For each frame VoxelReassigner builds two trees (forward + backward
    direction) and queries each. Print the build cost separately from
    the query cost so a future perf pass knows which to attack.
    """
    v, dim = voxel_reassign_setup
    rng = np.random.default_rng(7)
    coords_real = np.column_stack([
        rng.uniform(0.0, dim_size, size=n)
        for dim_size in v.spatial_shape
    ]).astype(np.float32)
    coords_query = np.column_stack([
        rng.uniform(0.0, dim_size, size=n)
        for dim_size in v.spatial_shape
    ]).astype(np.float32)

    # Warm up
    tree = cKDTree(coords_real)
    tree.query(coords_query, k=1, workers=-1)

    t_build = _time_call(lambda: cKDTree(coords_real), iters=5)

    tree = cKDTree(coords_real)
    t_query = _time_call(
        lambda: tree.query(coords_query, k=1, workers=-1),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] VoxelReassigner cKDTree {dim} N={n}: "
            f"build={t_build * 1000:.1f}ms query={t_query * 1000:.1f}ms"
        )


def test_bruteforce_vs_kdtree_dispatch_threshold(
    voxel_reassign_setup, capsys
) -> None:
    """Compare CPU brute-force vs cKDTree at a similar problem size.

    The dispatch in ``_query_tree`` picks brute-force when ``n_real * n_query <=
    max_bruteforce_pairs`` (default 1e7). At the boundary, the two paths
    have similar work; surfacing both wall-clocks lets a future perf pass
    re-tune the threshold from data rather than the current default.
    """
    v, dim = voxel_reassign_setup
    rng = np.random.default_rng(13)
    # Stay well below the 1e7 default to keep the test fast; the relative
    # comparison is what matters.
    n_real = 500
    n_query = 500
    coords_real = np.column_stack([
        rng.uniform(0.0, dim_size, size=n_real)
        for dim_size in v.spatial_shape
    ]).astype(np.float32)
    coords_query = np.column_stack([
        rng.uniform(0.0, dim_size, size=n_query)
        for dim_size in v.spatial_shape
    ]).astype(np.float32)

    # Warm up
    cKDTree(coords_real).query(coords_query, k=1, workers=-1)
    v._query_bruteforce_cpu(coords_real, coords_query)

    def _kdtree_path() -> None:
        tree = cKDTree(coords_real)
        tree.query(coords_query, k=1, workers=-1)

    t_kdtree = _time_call(_kdtree_path, iters=5)
    t_brute = _time_call(
        lambda: v._query_bruteforce_cpu(coords_real, coords_query),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] VoxelReassigner dispatch {dim} "
            f"(n_real={n_real} n_query={n_query}): "
            f"kdtree(build+query)={t_kdtree * 1000:.1f}ms "
            f"bruteforce_cpu={t_brute * 1000:.1f}ms"
        )
