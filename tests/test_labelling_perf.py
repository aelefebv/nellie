"""Opt-in performance tests for ``nellie.segmentation.labelling.Label``.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Mirrors
the pattern in :mod:`tests.test_filtering_perf`:

1. **End-to-end CPU baseline** prints a Label wall-clock baseline so a
   future regression on the yeast fixture is visible relative to a
   prior commit. No assertion: there is no canonical baseline checked
   in (hardware varies wildly across dev machines).
2. **End-to-end MPS variant** prints the same wall-clock baseline but
   on torch+MPS. Doubly marked ``benchmark`` + ``mps`` so it runs with
   either marker explicitly opted in. Skips cleanly on hardware
   without torch+MPS so users on non-Mac hardware (or Macs without
   ``pip install 'nellie[mps]'``) still see green when they run
   ``pytest -m benchmark``.
3. **Hot-path microbenchmarks** decompose the per-frame cost into
   the dominant sub-ops surfaced by reading ``_get_labels``:
   ``binary_fill_holes`` (3D) → first ``ndi.label`` →
   ``bincount`` + keep-mask → ``uniform_filter(size=3)`` → second
   ``ndi.label``. Plus an isolated ``_compute_frangi_threshold``
   measurement so threshold work is visible separately from
   structural work.

The interesting MPS comparison is "MPS time vs CPU time on the same
fixture"; capturing both as side-by-side ``[perf]`` lines is sufficient
— we don't gate on the absolute value because hardware varies and
labelling is a partial-acceleration story (see PRD #140 — only the
``uniform_filter`` runs on MPS; the structural ``binary_fill_holes`` /
``label`` ops round-trip to scipy on CPU).

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no Label perf pass has happened
yet, so assertions would be speculative. The prints scaffold the
future perf pass — they show which structural CPU op dominates.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi

from nellie.segmentation.labelling import Label, LabelConfig
from nellie.utils import adaptive_run


pytestmark = pytest.mark.benchmark

_CPU = LabelConfig(device="cpu")


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


def _release_label(lbl: Label) -> None:
    lbl.instance_label_memmap = None
    lbl.frangi_memmap = None
    lbl.im_memmap = None
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
# End-to-end Label baseline (CPU; informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_label_run_baseline_prints_wall_clock(
    dim, make_label_imageinfo_2d, make_label_imageinfo_3d, capsys
) -> None:
    """Print a Label wall-clock baseline. No assertion — read the output.

    Runs against the yeast fixtures via the per-test factory (which
    pre-populates ``im_preprocessed`` from a session cache, so this only
    pays the Label cost — not Filter on top).
    """
    factory = make_label_imageinfo_2d if dim == "2d" else make_label_imageinfo_3d

    def _one_run() -> None:
        info = factory()
        lbl = Label(info, _CPU, num_t=2)
        lbl.run()
        _release_label(lbl)

    # Warm up: first run pays one-time costs (memmap setup, scipy caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(f"\n[perf] Label.run() {dim} median over 3: {elapsed * 1000:.1f} ms")


# -------------------------------------------------------------------------
# MPS variant of the end-to-end Label baseline
#
# Doubly-marked (``benchmark`` + ``mps``) so it runs with either marker
# explicitly opted in. Skip cleanly when MPS isn't available so users on
# non-Mac hardware still see green when they run ``pytest -m benchmark``.
# -------------------------------------------------------------------------

@pytest.mark.mps
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_label_run_baseline_prints_wall_clock_mps(
    dim, make_label_imageinfo_2d, make_label_imageinfo_3d, capsys
) -> None:
    """Print a Label wall-clock baseline on MPS. No assertion — read the output.

    Mirror of :func:`test_label_run_baseline_prints_wall_clock`. The
    interesting comparison is "MPS time vs CPU time on the same fixture";
    we don't gate on the absolute value because hardware varies wildly
    across Mac models and labelling's MPS speedup is bounded by the
    structural-op CPU round-trips (binary_fill_holes + two label calls
    per frame — see PRD #140).
    """
    _require_mps()
    factory = make_label_imageinfo_2d if dim == "2d" else make_label_imageinfo_3d
    config_mps = LabelConfig(device="mps")

    def _one_run() -> None:
        info = factory()
        lbl = Label(info, config_mps, num_t=2)
        lbl.run()
        _release_label(lbl)

    # Warm up: first run pays one-time costs (memmap setup, MPS kernel
    # caches, torch import-time work).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] Label.run() {dim} (mps) median over 3: "
            f"{elapsed * 1000:.1f} ms"
        )


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational; CPU only)
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def label_setup(request, make_label_imageinfo_2d, make_label_imageinfo_3d):
    """Per-frame inputs for Label microbenchmarks (CPU).

    Builds a Label instance, allocates memmaps, computes thresholds for
    t=0, and extracts the per-frame Frangi view as numpy so the
    microbenchmarks can drive ``_get_labels`` and its sub-ops directly
    without re-running upstream stages between iterations.
    """
    factory = make_label_imageinfo_2d if request.param == "2d" else make_label_imageinfo_3d
    info = factory()
    lbl = Label(info, _CPU, num_t=2)
    lbl._allocate_memory()

    assert lbl.frangi_memmap is not None and lbl.im_memmap is not None
    frangi_view = np.asarray(lbl.frangi_memmap[0])
    original_view = np.asarray(lbl.im_memmap[0])
    intensity_thresh, frangi_thresh = lbl._compute_frame_thresholds(
        original_view, frangi_view
    )

    yield lbl, original_view, frangi_view, intensity_thresh, frangi_thresh, request.param

    _release_label(lbl)


def test_compute_frangi_threshold_wall_clock(label_setup, capsys) -> None:
    """`_compute_frangi_threshold` per-frame wall-clock.

    Triangle + Otsu in log10 domain, sampled. Surfaces the threshold
    cost separately from the structural ``_get_labels`` pipeline.
    """
    lbl, _orig, frangi_view, _ithresh, _fthresh, dim = label_setup

    # Warm up
    lbl._compute_frangi_threshold(frangi_view)

    elapsed = _time_call(
        lambda: lbl._compute_frangi_threshold(frangi_view),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] Label._compute_frangi_threshold {dim} median over 5: "
            f"{elapsed * 1000:.1f} ms"
        )


def test_get_labels_decomposed_wall_clock(label_setup, capsys) -> None:
    """Decompose `_get_labels`: total + sub-op timings.

    The sub-ops (mirroring the source order):
      1. ``binary_fill_holes`` (3D only — no-op in 2D path)
      2. First ``ndi.label`` on the threshold mask
      3. ``bincount`` + keep-mask construction
      4. ``uniform_filter(size=3)`` mean-filter smoothing
      5. Second ``ndi.label`` on the smoothed mask

    Surfaces which structural CPU op dominates so a future Label
    perf pass starts with the right hotspot identified.
    """
    lbl, _orig, frangi_view, _ithresh, fthresh, dim = label_setup

    # Warm up the whole pipeline
    lbl._get_labels(frangi_view, frangi_thresh=fthresh)

    t_total = _time_call(
        lambda: lbl._get_labels(frangi_view, frangi_thresh=fthresh),
        iters=3,
    )

    # Build sub-op inputs by mirroring `_get_labels` step-by-step on
    # CPU using scipy.ndimage. ``_get_labels`` mutates `mask` and
    # rebinds `labels` between steps, so we replicate that flow here
    # to time each step in isolation.
    is_3d = frangi_view.ndim == 3
    footprint = (np.ones((3, 3, 3), dtype=bool) if is_3d
                 else np.ones((3, 3), dtype=bool))
    mask = frangi_view > fthresh

    if is_3d:
        # Warm up
        scipy_ndi.binary_fill_holes(mask)
        t_fill = _time_call(
            lambda: scipy_ndi.binary_fill_holes(mask),
            iters=3,
        )
        mask = scipy_ndi.binary_fill_holes(mask)
    else:
        t_fill = 0.0

    # Step 2: first ndi.label
    scipy_ndi.label(mask, structure=footprint)
    t_label1 = _time_call(
        lambda: scipy_ndi.label(mask, structure=footprint),
        iters=3,
    )
    labels, _ = scipy_ndi.label(mask, structure=footprint)

    # Step 3: bincount + keep-mask
    def _bincount_step():
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        keep = areas >= lbl.min_area_pixels
        return keep[labels]

    _bincount_step()
    t_bincount = _time_call(_bincount_step, iters=3)
    mask = _bincount_step()

    # Step 4: uniform_filter on float mask
    mask_float = mask.astype(np.float32)
    scipy_ndi.uniform_filter(mask_float, size=3)
    t_uniform = _time_call(
        lambda: scipy_ndi.uniform_filter(mask_float, size=3),
        iters=3,
    )
    mask_smooth = scipy_ndi.uniform_filter(mask_float, size=3)
    mask = mask_smooth > 0.5

    # Step 5: second ndi.label
    scipy_ndi.label(mask, structure=footprint)
    t_label2 = _time_call(
        lambda: scipy_ndi.label(mask, structure=footprint),
        iters=3,
    )

    sum_subops = t_fill + t_label1 + t_bincount + t_uniform + t_label2

    with capsys.disabled():
        print(
            f"\n[perf] Label._get_labels {dim} total={t_total * 1000:.1f}ms; "
            f"fill={t_fill * 1000:.1f}ms label1={t_label1 * 1000:.1f}ms "
            f"bincount={t_bincount * 1000:.1f}ms uniform={t_uniform * 1000:.1f}ms "
            f"label2={t_label2 * 1000:.1f}ms (subop sum~{sum_subops * 1000:.1f}ms)"
        )
