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
3. **Hot-path microbenchmarks** decompose the per-frame cost into
   the dominant CPU-only ops surfaced by reading the source:
   - ``_skeletonize`` (whole-frame) vs ``_skeletonize_per_object``
     (low-memory variant) — pin the bulk-vs-per-object trade-off.
   - ``_relabel_objects`` per-frame: total + ``find_objects`` setup
     vs sum-of-per-object EDT — surfaces the many-small-objects
     pathology.
   - ``_remove_connected_label_pixels`` sparse coordinate-scan
     wall-clock — known CPU-only baseline.

The interesting MPS comparison is "MPS time vs CPU time on the same
fixture"; capturing both as side-by-side ``[perf]`` lines is sufficient
— we don't gate on the absolute value because hardware varies and
networking is a partial-acceleration story (see PRD #140 — only the
convolutional ``ndi.convolve`` runs on MPS; the structural
``ndi.label``, ``binary_fill_holes``, skeletonization, distance
transforms, and per-frame relabel pass all force-CPU).

The microbenchmarks are **informational** (printed `[perf]` lines, no
assertions). Filter's three assertions exist because each pinned a
decision from a real perf pass; here no Network perf pass has happened
yet, so assertions would be speculative. The prints scaffold the
future perf pass — they show which CPU-only op dominates so an author
can pick the right hot-path.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi

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


# -------------------------------------------------------------------------
# Hot-path microbenchmarks (informational; CPU only)
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def network_setup(request, make_network_imageinfo_2d, make_network_imageinfo_3d):
    """Per-frame inputs for Network microbenchmarks (CPU).

    Builds a Network instance, allocates memmaps, and extracts t=0
    label + Frangi as numpy + a fresh skeleton via the production
    ``_skeletonize`` so subsequent ops in the per-frame chain
    (``_remove_connected_label_pixels``, ``_relabel_objects``) can be
    driven directly without re-running upstream stages between
    iterations.
    """
    factory = make_network_imageinfo_2d if request.param == "2d" else make_network_imageinfo_3d
    info = factory()
    net = Network(info, _CPU, num_t=2)
    net._allocate_memory()

    assert net.label_memmap is not None and net.im_frangi_memmap is not None
    label_frame_cpu = np.asarray(net.label_memmap[0])
    frangi_frame_cpu = np.asarray(net.im_frangi_memmap[0])

    yield net, label_frame_cpu, frangi_frame_cpu, request.param

    _release_network(net)


def test_skeletonize_bulk_vs_per_object(network_setup, capsys) -> None:
    """`_skeletonize` (whole-frame) vs `_skeletonize_per_object` (low-memory).

    Pin the bulk-vs-per-object trade-off: the per-object loop pays a
    crop+skeletonize cost per label and is what runs under
    ``low_memory=True`` or after a bulk-skeletonize MemoryError.
    """
    net, label_frame, _frangi, dim = network_setup

    # Warm up
    net._skeletonize(label_frame)
    net._skeletonize_per_object(label_frame)

    t_bulk = _time_call(lambda: net._skeletonize(label_frame), iters=3)
    t_per_obj = _time_call(
        lambda: net._skeletonize_per_object(label_frame), iters=3
    )

    with capsys.disabled():
        print(
            f"\n[perf] Network skeletonize {dim}: "
            f"bulk={t_bulk * 1000:.1f}ms per_object={t_per_obj * 1000:.1f}ms "
            f"(per_object/bulk={t_per_obj / max(t_bulk, 1e-9):.2f}x)"
        )


def test_remove_connected_label_pixels_wall_clock(network_setup, capsys) -> None:
    """`_remove_connected_label_pixels` sparse coordinate-scan wall-clock.

    Sparse skeleton-coordinate scan with one OR over 8 (2D) or 26 (3D)
    neighbor offsets. CPU-only by design; ``low_memory`` is a no-op
    for this codepath.
    """
    net, label_frame, _frangi, dim = network_setup
    skel = net._skeletonize(label_frame)
    skel_pre = (skel > 0) * label_frame

    # Warm up
    net._remove_connected_label_pixels(skel_pre)

    elapsed = _time_call(
        lambda: net._remove_connected_label_pixels(skel_pre),
        iters=5,
    )

    with capsys.disabled():
        print(
            f"\n[perf] Network._remove_connected_label_pixels {dim} "
            f"median over 5: {elapsed * 1000:.1f} ms"
        )


def test_relabel_objects_decomposed_wall_clock(network_setup, capsys) -> None:
    """`_relabel_objects` per-frame: total + find_objects vs sum-of-EDT.

    The per-object EDT loop is the suspected many-small-objects
    pathology. Print total + find_objects setup cost + sum of
    per-object distance_transform_edt cost so the loop's contribution
    is visible.
    """
    net, label_frame, _frangi, dim = network_setup
    skel = net._skeletonize(label_frame)
    skel_clean = net._remove_connected_label_pixels(skel)
    skel_pre = (skel_clean > 0) * label_frame
    pixel_class = net._get_pixel_class(skel_pre, force_cpu=True)
    branch_skel_labels = net._get_branch_skel_labels(pixel_class, force_cpu=True)

    # Warm up
    net._relabel_objects(branch_skel_labels, label_frame)

    t_total = _time_call(
        lambda: net._relabel_objects(branch_skel_labels, label_frame),
        iters=3,
    )

    # Decompose: time `find_objects` + sum of per-object EDT
    labels_np = np.asarray(label_frame, dtype=np.int32)
    branch_np = np.asarray(branch_skel_labels, dtype=np.int32)

    scipy_ndi.find_objects(labels_np)
    t_find = _time_call(lambda: scipy_ndi.find_objects(labels_np), iters=3)

    slices = scipy_ndi.find_objects(labels_np)
    max_label = int(labels_np.max())

    def _sum_edts() -> None:
        for lab in range(1, max_label + 1):
            idx = lab - 1
            if idx >= len(slices):
                break
            sl = slices[idx]
            if sl is None:
                continue
            sub_labels = labels_np[sl]
            sub_branch = branch_np[sl]
            obj_mask = sub_labels == lab
            if not obj_mask.any():
                continue
            seed_mask = (sub_branch > 0) & obj_mask
            if not seed_mask.any():
                continue
            edt_input = np.logical_not(seed_mask)
            scipy_ndi.distance_transform_edt(
                edt_input,
                sampling=net.scaling,
                return_distances=False,
                return_indices=True,
            )

    _sum_edts()
    t_edt_sum = _time_call(_sum_edts, iters=3)

    with capsys.disabled():
        print(
            f"\n[perf] Network._relabel_objects {dim} ({max_label} labels) "
            f"total={t_total * 1000:.1f}ms; "
            f"find_objects={t_find * 1000:.1f}ms "
            f"sum_per_object_edt={t_edt_sum * 1000:.1f}ms"
        )
