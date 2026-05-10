"""Opt-in performance tests for ``nellie.feature_extraction.hierarchical.Hierarchy``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`,
:mod:`tests.test_labelling_perf`, :mod:`tests.test_networking_perf`,
and :mod:`tests.test_hu_tracking_perf`:

1. **End-to-end CPU baseline** prints a Hierarchy wall-clock baseline
   so a future regression on the yeast fixture is visible relative to
   a prior commit.
2. **Hot-path microbenchmarks** decompose the per-frame cost into the
   three suspected hotspots surfaced by reading the code:
   - ``Branches._compute_branch_lengths_and_degrees`` 2D (8-direction)
     vs 3D (26-direction) neighbor-offset loop — the largest single
     hotspot in branch-level features.
   - ``Voxels._run_frame`` whole-frame — dominant by voxel count.
   - ``Voxels._get_motility_stats`` — flow-interpolation overhead
     when temporal data is present.

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no perf pass has happened yet,
so assertions would be speculative. The prints scaffold the future
perf pass.

**No MPS variant.** Hierarchy is not MPS-onboarded (per
``wiki/now.md`` and ``wiki/gpu-runtime.md``: only filtering, labelling,
networking, and hu_tracking are). ``device="mps"`` pins the cascade
to MPS-only without CPU fallback and would fire the ``torch_ndi``
shim trap. When Hierarchy is onboarded, add an MPS variant.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest

from nellie.feature_extraction.hierarchical import (
    Branches,
    Hierarchy,
    HierarchyConfig,
    Voxels,
)
from nellie.tracking.flow_interpolation import FlowInterpolator


pytestmark = pytest.mark.benchmark

_CPU = HierarchyConfig(skip_nodes=False, device="cpu")


def _release_hierarchy(h: Hierarchy) -> None:
    """Drop Hierarchy's memmap references and force gc.

    Ten memmap handles to release. Required on Windows: input memmaps
    stay file-locked until handles are dropped.
    """
    for attr in (
        "im_raw",
        "im_struct",
        "im_distance",
        "im_skel",
        "label_components",
        "label_branches",
        "im_border_mask",
        "im_pixel_class",
        "im_obj_reassigned",
        "im_branch_reassigned",
    ):
        if hasattr(h, attr):
            setattr(h, attr, None)
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
# End-to-end Hierarchy baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_hierarchy_run_baseline_prints_wall_clock(
    dim,
    make_hierarchical_imageinfo_2d,
    make_hierarchical_imageinfo_3d,
    capsys,
) -> None:
    """Print a Hierarchy wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates 9 input memmaps from session caches: Frangi, Distance,
    Skel, PixelClass, InstanceLabel, SkelRelabelled, Border, plus the
    two reassigned memmaps and the Hu flow array — so this only pays
    the Hierarchy cost, not Filter+Label+Network+Markers+Hu+VoxelReassign
    on top).
    """
    factory = (
        make_hierarchical_imageinfo_2d if dim == "2d" else make_hierarchical_imageinfo_3d
    )

    def _one_run() -> None:
        info = factory()
        h = Hierarchy(info, _CPU)
        h.run()
        _release_hierarchy(h)

    # Warm up: first run pays one-time costs (memmap setup, scipy caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] Hierarchy.run() {dim} median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational)
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def hierarchy_setup(
    request, make_hierarchical_imageinfo_2d, make_hierarchical_imageinfo_3d
):
    """A Hierarchy with `_allocate_memory()` and flow interpolators set up.

    Required so ``Branches._compute_branch_lengths_and_degrees`` and
    ``Voxels._run_frame`` / ``_get_motility_stats`` can be driven
    directly without re-running the full ``Hierarchy.run()`` cascade.
    """
    factory = (
        make_hierarchical_imageinfo_2d if request.param == "2d" else make_hierarchical_imageinfo_3d
    )
    info = factory()
    h = Hierarchy(info, _CPU)
    h._allocate_memory()
    # Mirror the lazy init in `_run_hierarchy` so motility microbench works.
    if (
        h.enable_motility
        and not h.im_info.no_t
        and h.num_t is not None
        and h.num_t > 1
    ):
        h.flow_interpolator_fw = FlowInterpolator(h.im_info)
        h.flow_interpolator_bw = FlowInterpolator(h.im_info, forward=False)

    yield h, request.param

    _release_hierarchy(h)


def test_branches_compute_branch_lengths_and_degrees_wall_clock(
    hierarchy_setup, capsys
) -> None:
    """`Branches._compute_branch_lengths_and_degrees` per-frame wall-clock.

    The 8-direction (2D) / 26-direction (3D) neighbor-offset loop is
    the largest single hotspot in branch-level features. The 3D path
    runs ~3x as many offsets as 2D — surfacing both makes the slowdown
    on 3D obvious.
    """
    h, dim = hierarchy_setup
    branches = Branches(h)

    # Warm up
    branches._compute_branch_lengths_and_degrees(t=0)

    elapsed = _time_call(
        lambda: branches._compute_branch_lengths_and_degrees(t=0),
        iters=3,
    )

    with capsys.disabled():
        print(
            f"\n[perf] Branches._compute_branch_lengths_and_degrees {dim} "
            f"median over 3: {elapsed * 1000:.1f} ms"
        )


def test_voxels_run_frame_wall_clock(hierarchy_setup, capsys) -> None:
    """`Voxels._run_frame` per-frame wall-clock (dominant by voxel count).

    Each call mutates Voxels' lists (append-only), so we build a fresh
    Voxels each iteration — measuring the per-frame cost, not amortized
    state-rebuild.
    """
    h, dim = hierarchy_setup

    def _one_call() -> None:
        v = Voxels(h)
        v._run_frame(t=0)

    # Warm up
    _one_call()

    elapsed = _time_call(_one_call, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] Voxels._run_frame {dim} median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


def test_voxels_get_motility_stats_wall_clock(hierarchy_setup, capsys) -> None:
    """`Voxels._get_motility_stats` wall-clock — flow-interpolation overhead.

    Only fires when temporal (num_t > 1) and ``enable_motility=True``.
    The 2-frame yeast fixture has T=2, so motility runs at t=0 with
    forward flow and at t=1 with backward flow. Measure t=0 (the
    forward path) for consistency.

    Note: ``_get_motility_stats`` reads ``self.branch_labels[t]``,
    which is populated by ``_run_frame``. We pre-populate Voxels'
    state by running one full frame before timing the motility-only
    sub-call. The lists keep growing across timing iterations (each
    call appends to vec01/vec12/...), but the time spent per call
    is a stable measurement of the motility path.
    """
    h, dim = hierarchy_setup
    if h.flow_interpolator_fw is None:
        pytest.skip(
            "Flow interpolator not set up for this fixture; "
            "motility stats are no-op."
        )

    coords = np.argwhere(h.label_components[0] > 0)
    if coords.size == 0:
        pytest.skip(f"No labeled voxels in {dim} fixture at t=0; nothing to score.")

    # Pre-populate Voxels state via a full-frame run so `_get_motility_stats`
    # can be invoked in isolation (it depends on `self.branch_labels[t]`).
    v = Voxels(h)
    v._run_frame(t=0)

    # Warm up the isolated sub-call
    v._get_motility_stats(t=0, coords_1_px=coords)

    elapsed = _time_call(
        lambda: v._get_motility_stats(t=0, coords_1_px=coords),
        iters=3,
    )

    with capsys.disabled():
        print(
            f"\n[perf] Voxels._get_motility_stats {dim} ({coords.shape[0]} voxels) "
            f"median over 3: {elapsed * 1000:.1f} ms"
        )
