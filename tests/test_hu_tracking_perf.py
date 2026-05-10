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

The interesting comparison is "MPS time vs CPU time on the same
fixture"; capturing both as side-by-side ``[perf]`` lines is sufficient
— we don't gate on the absolute value because hardware varies and
hu_tracking is the lightest-on-convolutional-ops of the four onboarded
stages: only the per-frame distance ``ndi.maximum_filter`` runs on
MPS via the shim. The bulk of the per-frame cost (moment math, ROI
extraction, dense cost-matrix construction) runs as broadcast
elementwise ops on the active xp namespace, which on MPS pays the
torch dispatch overhead per op without much speedup over numpy on
small fixtures.
"""

from __future__ import annotations

import gc
import time
from statistics import median

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

    Per PRD #140 § Implementation Decisions, the moment-distance
    matrix in ``_get_difference_matrix`` casts to ``xp.float64`` which
    silently coerces to ``float32`` on MPS — the cost-matrix path is
    where the bulk of the wall-clock time goes for fixtures with many
    markers, and the float32-instead-of-float64 reduction can shave
    cycles vs the CPU baseline (or add some — measure both ways).
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
