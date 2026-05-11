"""Opt-in performance tests for ``nellie.tracking.hu_tracking.HuMomentTracking``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`,
:mod:`tests.test_labelling_perf`, and :mod:`tests.test_networking_perf`:

1. **End-to-end CPU baseline** prints a HuMomentTracking wall-clock
   baseline so a future regression on the yeast fixture is visible
   relative to a prior commit. No assertion: there is no canonical
   baseline checked in (hardware varies wildly across dev machines).
2. **End-to-end MPS variant** prints the same wall-clock baseline but
   on torch+MPS. Doubly marked ``benchmark`` + ``mps`` so it runs with
   either marker explicitly opted in. Skips cleanly on hardware
   without torch+MPS so users on non-Mac hardware (or Macs without
   ``pip install 'nellie[mps]'``) still see green when they run
   ``pytest -m benchmark``.
3. **Hot-path microbenchmarks** decompose the per-frame matching
   cost into the dominant kernels surfaced by reading the source:
   - ``_get_cost_matrix`` end-to-end at N = 100 / 500 / 1000 markers
     (the per-frame matching bottleneck — now per-feature streaming
     after PRD #196 / Slice 2 (#198); previously the dense (N, N, F)
     broadcast tensor).
   - ``_calculate_normalized_moments`` and ``_get_hu_moments`` 3D
     scaling — the per-frame moment-math bottleneck (see PRD #191).
   - ``_find_best_matches`` decomposed: Python row/col loops vs the
     underlying ``argmin``/``min`` work — surfaces whether
     vectorizing the loops would pay off.

The interesting MPS comparison is "MPS time vs CPU time on the same
fixture"; capturing both as side-by-side ``[perf]`` lines is sufficient
— we don't gate on the absolute value because hardware varies and
hu_tracking is the lightest-on-convolutional-ops of the four onboarded
stages: only the per-frame distance ``ndi.maximum_filter`` runs on
MPS via the shim. The bulk of the per-frame cost (moment math, ROI
extraction, dense cost-matrix construction) runs as broadcast
elementwise ops on the active xp namespace, which on MPS pays the
torch dispatch overhead per op without much speedup over numpy on
small fixtures.

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no Hu perf pass has happened
yet, so assertions would be speculative. The prints scaffold the
future perf pass.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest

from nellie.tracking.hu_tracking import HuMomentTracking, HuMomentTrackingConfig
from nellie.utils import adaptive_run


pytestmark = pytest.mark.benchmark

_CPU = HuMomentTrackingConfig(device="cpu")


def _require_mps() -> None:
    """Skip the calling MPS benchmark if torch+MPS isn't actually available.

    Mirrors the helper in :mod:`tests.test_mps_smoke` so the skip
    wording is consistent across the test trio.
    """
    if not adaptive_run.mps_available():
        pytest.skip(
            "torch+MPS not available — install with `pip install 'nellie[mps]'` "
            "and run on Apple Silicon."
        )


def _release_hu(h: HuMomentTracking) -> None:
    """Drop a HuMomentTracking's memmap references and force gc.

    Mirrors the helper in :mod:`tests.test_mps_smoke` — five memmap
    handles to release. Required on Windows: input memmaps stay
    file-locked until handles are dropped.
    """
    h.label_memmap = None
    h.im_memmap = None
    h.im_frangi_memmap = None
    h.im_marker_memmap = None
    h.im_distance_memmap = None
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
# End-to-end HuMomentTracking baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_hu_tracking_run_baseline_prints_wall_clock(
    dim, make_hu_imageinfo_2d, make_hu_imageinfo_3d, capsys
) -> None:
    """Print a HuMomentTracking wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates the four upstream memmaps — ``im_preprocessed``,
    ``im_instance_label``, ``im_marker``, ``im_distance`` — from the
    session caches, so this only pays the HuMomentTracking cost — not
    Filter + Label + Markers on top).
    """
    factory = make_hu_imageinfo_2d if dim == "2d" else make_hu_imageinfo_3d

    def _one_run() -> None:
        info = factory()
        h = HuMomentTracking(info, _CPU, num_t=2)
        h.run()
        _release_hu(h)

    # Warm up: first run pays one-time costs (memmap setup, scipy caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking.run() {dim} median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# MPS variant of the end-to-end HuMomentTracking baseline
#
# Doubly-marked (``benchmark`` + ``mps``) so it runs with either marker
# explicitly opted in. Skip cleanly when MPS isn't available so users on
# non-Mac hardware still see green when they run ``pytest -m benchmark``.
# -------------------------------------------------------------------------

@pytest.mark.mps
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_hu_tracking_run_baseline_prints_wall_clock_mps(
    dim, make_hu_imageinfo_2d, make_hu_imageinfo_3d, capsys
) -> None:
    """Print a HuMomentTracking wall-clock baseline on MPS. No assertion — read the output.

    Mirror of :func:`test_hu_tracking_run_baseline_prints_wall_clock`.
    The interesting comparison is "MPS time vs CPU time on the same
    fixture"; we don't gate on the absolute value because hardware
    varies wildly across Mac models and hu_tracking is the
    lightest-on-convolutional-ops of the four onboarded stages — only
    the per-frame ``ndi.maximum_filter`` for the distance dilation
    runs on MPS via the shim. The moment / ROI / cost-matrix machinery
    is dominated by elementwise broadcast ops which on MPS pay the
    torch dispatch overhead per op without much speedup over numpy on
    small fixtures.

    Per PRD #196 / ADR 0009, the cost-matrix path now runs at explicit
    float32 throughout (the previous ``_get_difference_matrix``
    ``xp.float64`` cast that silently coerced on MPS is gone — the
    streaming refactor pinned float32 across all backends, eliminating
    the precision-divergence risk that PRD #140 § Implementation
    Decisions had flagged for hu_tracking).
    """
    _require_mps()
    factory = make_hu_imageinfo_2d if dim == "2d" else make_hu_imageinfo_3d
    config_mps = HuMomentTrackingConfig(device="mps")

    def _one_run() -> None:
        info = factory()
        h = HuMomentTracking(info, config_mps, num_t=2)
        h.run()
        _release_hu(h)

    # Warm up: first run pays one-time costs (memmap setup, MPS kernel
    # caches, torch import-time work).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking.run() {dim} (mps) median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational; CPU only)
# -------------------------------------------------------------------------

# Approximate feature dimensions used by the dense matching path. The
# absolute values matter less than scaling with N — production hu_tracking
# uses ~8 stats features per axis and 7 Hu moments (moment 7, the mirror
# moment, is intentionally omitted per the glossary).
_F_STATS = 8
_F_HU = 7
_NDIM = 3


@pytest.fixture
def hu_tracking_cpu(make_hu_imageinfo_3d):
    """A HuMomentTracking instance with CPU backend, no run.

    Used to drive the ``_get_cost_matrix`` / ``_get_difference_matrix`` /
    ``_find_best_matches`` methods with synthetic numpy inputs so the
    microbenchmarks scale with N without being bound to fixture
    marker counts.
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(info, _CPU, num_t=2)
    yield h
    _release_hu(h)


def _synthesize_match_inputs(
    n_post: int, n_pre: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random ``coords / stats / hu`` arrays at the given pre/post sizes."""
    coords_post = rng.uniform(0.0, 100.0, size=(n_post, _NDIM)).astype(np.float64)
    coords_pre = rng.uniform(0.0, 100.0, size=(n_pre, _NDIM)).astype(np.float64)
    stats_post = rng.normal(0.0, 1.0, size=(n_post, _F_STATS)).astype(np.float32)
    stats_pre = rng.normal(0.0, 1.0, size=(n_pre, _F_STATS)).astype(np.float32)
    hu_post = rng.normal(0.0, 1.0, size=(n_post, _F_HU)).astype(np.float32)
    hu_pre = rng.normal(0.0, 1.0, size=(n_pre, _F_HU)).astype(np.float32)
    return coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_get_cost_matrix_scaling(hu_tracking_cpu, n, capsys) -> None:
    """`_get_cost_matrix` end-to-end at N = 100, 500, 1000.

    The dense (N_post, N_pre, F) broadcast tensor + the three
    difference-matrix + z-score paths are the per-frame bottleneck for
    moderately-large marker counts. Print scaling per N so the
    quadratic memory + time growth is visible.
    """
    h = hu_tracking_cpu
    rng = np.random.default_rng(11)
    inputs = _synthesize_match_inputs(n_post=n, n_pre=n, rng=rng)

    # Warm up
    h._get_cost_matrix(*inputs)

    elapsed = _time_call(lambda: h._get_cost_matrix(*inputs), iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking._get_cost_matrix N={n} "
            f"median over 3: {elapsed * 1000:.1f} ms"
        )


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_calculate_normalized_moments_scaling(hu_tracking_cpu, n, capsys) -> None:
    """`_calculate_normalized_moments` scaling at N = 100/500/1000, H=W=21.

    The (N, H, W, 4, 4) broadcast tensor materialized inside the
    function is the biggest known memory blowup in hu_tracking.py per
    the 2026-05-11 audit. For typical N=1000 markers with H=W=21 (the
    yeast 3D fixture's max_radius), each broadcast tensor is ~28 MB
    and is allocated twice per call (raw + central moments).

    Backfilled here as the before/after measurement vehicle for the
    matmul rewrite in PRD #191 / Slice 2 (#193). The 2D path calls
    this function once per frame; the 3D path calls it 3× per frame
    (once per orthogonal projection — see
    :func:`test_get_hu_moments_3d_scaling` below).
    """
    h = hu_tracking_cpu
    rng = np.random.default_rng(19)
    H = W = 21  # typical max_radius for the yeast 3D fixture
    images = rng.uniform(0.0, 1.0, size=(n, H, W)).astype(np.float32)

    # Warm up
    h._calculate_normalized_moments(images)

    elapsed = _time_call(lambda: h._calculate_normalized_moments(images), iters=5)

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking._calculate_normalized_moments "
            f"N={n} H=W={H} median over 5: {elapsed * 1000:.1f} ms"
        )


@pytest.mark.parametrize("n", [100, 500, 1000])
def test_get_hu_moments_3d_scaling(hu_tracking_cpu, n, capsys) -> None:
    """`_get_hu_moments` 3D path scaling at N = 100/500/1000.

    The 3D path triggers ``_get_orthogonal_projections`` (3× ``xp.max``
    over the volume) followed by 3× ``_calculate_normalized_moments``
    (one per projection), then ``_calculate_hu_moments`` 3× and a
    final concatenate. This is the per-3D-frame moment-math hot loop.

    Backfilled as the before/after vehicle for PRD #191 / Slice 2 —
    the matmul rewrite's win compounds 3× on the 3D path vs 1× on the
    2D path, so the 3D scaling number is the more impactful
    measurement for production workloads.

    The fixture is bound to the 3D yeast image (``no_z = False``), so
    ``_get_hu_moments`` takes the 3D branch even though the synthetic
    sub_volumes are constructed directly here.
    """
    h = hu_tracking_cpu
    rng = np.random.default_rng(23)
    H = W = Z = 21
    sub_volumes = rng.uniform(0.0, 1.0, size=(n, Z, H, W)).astype(np.float32)

    # Warm up
    h._get_hu_moments(sub_volumes)

    elapsed = _time_call(lambda: h._get_hu_moments(sub_volumes), iters=5)

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking._get_hu_moments 3D "
            f"N={n} Z=H=W={Z} median over 5: {elapsed * 1000:.1f} ms"
        )


@pytest.mark.parametrize("n", [100, 1000])
def test_find_best_matches_loop_overhead(hu_tracking_cpu, n, capsys) -> None:
    """Decompose `_find_best_matches`: total + isolated argmin/min vs Python loops.

    The function does row+col argmin (vectorized) followed by two
    Python loops over N candidates each. Print total + isolated
    argmin/min cost — the difference is the Python-loop overhead, a
    candidate for vectorization.
    """
    h = hu_tracking_cpu
    rng = np.random.default_rng(17)
    cost_matrix = rng.uniform(0.0, 5.0, size=(n, n)).astype(np.float32)

    # Warm up
    h._find_best_matches(cost_matrix)

    t_total = _time_call(lambda: h._find_best_matches(cost_matrix), iters=5)

    # Isolate the underlying argmin/min that the function calls
    np.argmin(cost_matrix, axis=1)
    np.min(cost_matrix, axis=1)
    t_argmin = _time_call(
        lambda: (np.argmin(cost_matrix, axis=1), np.min(cost_matrix, axis=1),
                 np.argmin(cost_matrix, axis=0), np.min(cost_matrix, axis=0)),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] HuMomentTracking._find_best_matches N={n}: "
            f"total={t_total * 1000:.2f}ms argmin_min_only={t_argmin * 1000:.2f}ms "
            f"(loop overhead~{(t_total - t_argmin) * 1000:.2f}ms)"
        )
