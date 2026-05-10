"""Opt-in performance tests for ``nellie.segmentation.mocap_marking.Markers``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`,
:mod:`tests.test_labelling_perf`, :mod:`tests.test_networking_perf`,
and :mod:`tests.test_hu_tracking_perf`:

1. **End-to-end CPU baseline** prints a Markers wall-clock baseline so
   a future regression on the yeast fixture is visible relative to a
   prior commit.
2. **Hot-path microbenchmarks** decompose the per-frame cost into the
   three dominant ops surfaced by reading the code: the multi-scale
   LoG loop in ``_local_max_peak``, the morphological NMS in
   ``_remove_close_peaks``, and the dilation+EDT in ``_distance_im``.

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no perf pass has happened yet,
so assertions would be speculative. The prints scaffold the future
perf pass — they show which inner op dominates, so an author can
target the right hot-path rather than instrumenting from scratch.

**No MPS variant.** Markers is not MPS-onboarded (per ``wiki/now.md``
and ``wiki/gpu-runtime.md``: only filtering, labelling, networking,
and hu_tracking are). ``device="mps"`` pins the cascade to MPS-only
(no CPU fallback) and would fire the ``torch_ndi`` shim trap on
``binary_dilation``. When Markers is onboarded, add an MPS variant
mirroring the four onboarded stages' perf files.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi

from nellie.segmentation.mocap_marking import Markers, MarkersConfig


pytestmark = pytest.mark.benchmark

_CPU = MarkersConfig(device="cpu")


def _release_markers(m: Markers) -> None:
    m.im_marker_memmap = None
    m.im_distance_memmap = None
    m.im_border_memmap = None
    m.label_memmap = None
    m.im_memmap = None
    m.im_frangi_memmap = None
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
# End-to-end Markers baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_markers_run_baseline_prints_wall_clock(
    dim, make_markers_imageinfo_2d, make_markers_imageinfo_3d, capsys
) -> None:
    """Print a Markers wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates ``im_preprocessed`` and ``im_instance_label`` from
    session caches, so this only pays the Markers cost — not Filter +
    Label on top).
    """
    factory = make_markers_imageinfo_2d if dim == "2d" else make_markers_imageinfo_3d

    def _one_run() -> None:
        info = factory()
        m = Markers(info, _CPU, num_t=2)
        m.run()
        _release_markers(m)

    # Warm up: first run pays one-time costs (memmap setup, scipy caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(f"\n[perf] Markers.run() {dim} median over 3: {elapsed * 1000:.1f} ms")


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational)
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def markers_setup(request, make_markers_imageinfo_2d, make_markers_imageinfo_3d):
    """Per-frame inputs for Markers microbenchmarks (CPU).

    Builds a Markers instance, allocates memmaps, sets sigmas, and
    extracts t=0 intensity + mask as numpy arrays so the microbenchmarks
    can drive the inner methods directly without re-running upstream
    stages between iterations.
    """
    factory = make_markers_imageinfo_2d if request.param == "2d" else make_markers_imageinfo_3d
    info = factory()
    m = Markers(info, _CPU, num_t=2)
    m._allocate_memory()
    m._set_default_sigmas()

    assert m.im_memmap is not None and m.label_memmap is not None
    intensity_frame = np.asarray(m.im_memmap[0])
    mask_frame = np.asarray(m.label_memmap[0] > 0).astype(bool, copy=False)

    yield m, mask_frame, intensity_frame, request.param

    _release_markers(m)


def test_distance_im_wall_clock(markers_setup, capsys) -> None:
    """`_distance_im` (binary_dilation + EDT) per-frame wall-clock.

    Usually negligible relative to the LoG loop; one print to confirm
    that assumption holds on the fixture.
    """
    m, mask, _intensity, dim = markers_setup

    # Warm up
    m._distance_im(mask)

    elapsed = _time_call(lambda: m._distance_im(mask), iters=5)

    with capsys.disabled():
        print(
            f"\n[perf] Markers._distance_im {dim} median over 5: "
            f"{elapsed * 1000:.1f} ms"
        )


def test_local_max_peak_per_sigma_breakdown(markers_setup, capsys) -> None:
    """Decompose `_local_max_peak`: total time + isolated per-sigma LoG + max_filter.

    Surfaces whether scaling on ``num_sigma`` (default 5) is what
    people expect. The per-sigma LoG (``gaussian_laplace``) and the
    in-image max_filter are the two convolutional ops; the rest is
    elementwise.
    """
    m, mask, _intensity, dim = markers_setup
    distance_im, _ = m._distance_im(mask)

    # Warm up
    m._local_max_peak(distance_im, mask, distance_im, low_memory=False)

    t_total = _time_call(
        lambda: m._local_max_peak(distance_im, mask, distance_im, low_memory=False),
        iters=3,
    )

    # Isolate one sigma's LoG and max_filter to surface the per-iter
    # cost of the dominant convolutional ops in the per-sigma loop.
    sigma_val = float(m.sigmas[0])
    sigma_vec = m._get_sigma_vec(sigma_val)
    base_im = distance_im.astype(np.float32, copy=False)

    # Warm up the isolated ops too
    scipy_ndi.gaussian_laplace(base_im, sigma_vec)
    scipy_ndi.maximum_filter(base_im, size=3, mode="nearest")

    t_log = _time_call(
        lambda: scipy_ndi.gaussian_laplace(base_im, sigma_vec),
        iters=5,
    )
    t_maxfilt = _time_call(
        lambda: scipy_ndi.maximum_filter(base_im, size=3, mode="nearest"),
        iters=5,
    )

    n_sigmas = len(m.sigmas)
    with capsys.disabled():
        print(
            f"\n[perf] Markers._local_max_peak {dim} ({n_sigmas} sigmas) "
            f"total={t_total * 1000:.1f}ms; "
            f"per-sigma LoG~{t_log * 1000:.1f}ms; "
            f"max_filter(size=3)~{t_maxfilt * 1000:.1f}ms"
        )


def test_remove_close_peaks_wall_clock(markers_setup, capsys) -> None:
    """`_remove_close_peaks` morphological NMS — single hot op, baseline.

    The maximum_filter window is ``2 * peak_min_distance + 1`` (default
    5), applied to a sparse score image of intensities at peak
    coordinates only.
    """
    m, mask, intensity, dim = markers_setup
    distance_im, _ = m._distance_im(mask)
    coords = m._local_max_peak(distance_im, mask, distance_im, low_memory=False)

    n_peaks = int(coords.shape[0])
    if n_peaks == 0:
        pytest.skip(f"No peaks found in {dim} fixture; nothing to NMS.")

    # Warm up
    m._remove_close_peaks(coords, intensity, low_memory=False)

    elapsed = _time_call(
        lambda: m._remove_close_peaks(coords, intensity, low_memory=False),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] Markers._remove_close_peaks {dim} ({n_peaks} peaks) "
            f"median over 5: {elapsed * 1000:.1f} ms"
        )
