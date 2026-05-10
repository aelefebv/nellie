"""Opt-in performance tests for ``nellie.segmentation.networking.Network``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf` and
:mod:`tests.test_labelling_perf`:

1. **End-to-end CPU baseline** prints a Network wall-clock baseline so a
   future regression on the yeast fixture is visible relative to a
   prior commit. No assertion: there is no canonical baseline checked
   in (hardware varies wildly across dev machines).
2. **End-to-end MPS variant** prints the same wall-clock baseline but
   on torch+MPS. Doubly marked ``benchmark`` + ``mps`` so it runs with
   either marker explicitly opted in. Skips cleanly on hardware
   without torch+MPS so users on non-Mac hardware (or Macs without
   ``pip install 'nellie[mps]'``) still see green when they run
   ``pytest -m benchmark``.

The interesting comparison is "MPS time vs CPU time on the same
fixture"; capturing both as side-by-side ``[perf]`` lines is sufficient
— we don't gate on the absolute value because hardware varies and
networking is a partial-acceleration story (see PRD #140 — only the
convolutional ``ndi.convolve`` runs on MPS; the structural
``ndi.label``, ``binary_fill_holes``, skeletonization, distance
transforms, and per-frame relabel pass all force-CPU).
"""

from __future__ import annotations

import gc
import time
from statistics import median

import pytest

from nellie.segmentation.networking import Network, NetworkConfig
from nellie.utils import adaptive_run


pytestmark = pytest.mark.benchmark

_CPU = NetworkConfig(device="cpu")


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


def _release_network(net: Network) -> None:
    net.skel_memmap = None
    net.pixel_class_memmap = None
    net.skel_relabelled_memmap = None
    net.label_memmap = None
    net.im_memmap = None
    net.im_frangi_memmap = None
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
# End-to-end Network baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_network_run_baseline_prints_wall_clock(
    dim, make_network_imageinfo_2d, make_network_imageinfo_3d, capsys
) -> None:
    """Print a Network wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates ``im_preprocessed`` and ``im_instance_label`` from a
    session cache, so this only pays the Network cost — not Filter +
    Label on top).
    """
    factory = make_network_imageinfo_2d if dim == "2d" else make_network_imageinfo_3d

    def _one_run() -> None:
        info = factory()
        net = Network(info, _CPU, num_t=2)
        net.run()
        _release_network(net)

    # Warm up: first run pays one-time costs (memmap setup, scipy caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(f"\n[perf] Network.run() {dim} median over 3: {elapsed * 1000:.1f} ms")


# -------------------------------------------------------------------------
# MPS variant of the end-to-end Network baseline
#
# Doubly-marked (``benchmark`` + ``mps``) so it runs with either marker
# explicitly opted in. Skip cleanly when MPS isn't available so users on
# non-Mac hardware still see green when they run ``pytest -m benchmark``.
# -------------------------------------------------------------------------

@pytest.mark.mps
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_network_run_baseline_prints_wall_clock_mps(
    dim, make_network_imageinfo_2d, make_network_imageinfo_3d, capsys
) -> None:
    """Print a Network wall-clock baseline on MPS. No assertion — read the output.

    Mirror of :func:`test_network_run_baseline_prints_wall_clock`. The
    interesting comparison is "MPS time vs CPU time on the same fixture";
    we don't gate on the absolute value because hardware varies wildly
    across Mac models and networking's MPS speedup is bounded by the
    structural-op CPU round-trips (skeletonization, ``ndi.label``, the
    per-object distance transforms in ``_relabel_objects``, and the
    boundary-stripped 3×3(×3) min/max neighborhood filters in
    ``_remove_connected_label_pixels`` — see PRD #140).
    """
    _require_mps()
    factory = make_network_imageinfo_2d if dim == "2d" else make_network_imageinfo_3d
    config_mps = NetworkConfig(device="mps")

    def _one_run() -> None:
        info = factory()
        net = Network(info, config_mps, num_t=2)
        net.run()
        _release_network(net)

    # Warm up: first run pays one-time costs (memmap setup, MPS kernel
    # caches, torch import-time work).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] Network.run() {dim} (mps) median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )
