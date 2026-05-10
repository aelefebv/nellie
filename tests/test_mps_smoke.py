"""MPS smoke tests for the nellie pipeline stages.

Each test here boots a stage end-to-end with ``device="mps"`` on a tiny
synthetic fixture and verifies it doesn't crash. Output shape/dtype
parity vs the CPU run is asserted; numerical equivalence is owned by
``tests/test_mps_equivalence.py``.

All tests are gated by the ``mps`` pytest marker (deselected by default
in ``pyproject.toml``). Run with ``pytest -m mps`` on a Mac with
``pip install 'nellie[mps]'`` and a working MPS device. Tests skip
cleanly when torch is missing or MPS is unavailable so users on
non-Mac hardware (or Macs without the extra installed) still see green
when they run ``pytest -m mps``.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label, LabelConfig
from nellie.segmentation.networking import Network, NetworkConfig
from nellie.tracking.hu_tracking import HuMomentTracking, HuMomentTrackingConfig
from nellie.utils import adaptive_run


pytestmark = pytest.mark.mps


def _require_mps() -> None:
    """Skip the test cleanly if torch+MPS isn't actually available.

    The ``mps`` marker only deselects the tests by default; it doesn't
    enforce hardware. A user who runs ``pytest -m mps`` on a machine
    without torch+MPS still needs a graceful skip rather than a hard
    failure.
    """
    if not adaptive_run.mps_available():
        pytest.skip(
            "torch+MPS not available — install with `pip install 'nellie[mps]'` "
            "and run on Apple Silicon."
        )


def _release_filter(filt: Filter) -> None:
    """Drop a Filter's memmap references and force gc.

    Mirrors the helper in ``test_filtering.py``; required on Windows
    (the ``im_preprocessed`` memmap holds an exclusive handle that
    blocks subsequent overwrites).
    """
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()


def _release_label(lbl: Label) -> None:
    """Drop a Label's memmap references and force gc.

    Mirrors :func:`_release_filter` — same Windows file-lock concern,
    extra ``instance_label_memmap`` handle to release.
    """
    lbl.instance_label_memmap = None
    lbl.frangi_memmap = None
    lbl.im_memmap = None
    gc.collect()


def _release_network(net: Network) -> None:
    """Drop a Network's memmap references and force gc.

    Mirrors :func:`_release_label` — same Windows file-lock concern, six
    memmap handles to release (three inputs that ``Network._allocate_memory``
    opens as read-only and three outputs).
    """
    net.skel_memmap = None
    net.pixel_class_memmap = None
    net.skel_relabelled_memmap = None
    net.label_memmap = None
    net.im_memmap = None
    net.im_frangi_memmap = None
    gc.collect()


def _release_hu(h: HuMomentTracking) -> None:
    """Drop a HuMomentTracking's memmap references and force gc.

    Mirrors :func:`_release_network` — same Windows file-lock concern,
    five memmap handles to release (all five inputs that
    ``HuMomentTracking._allocate_memory`` opens as read-only). The
    ``flow_vector_array`` output is a regular ``.npy`` written via
    ``np.save``, not a memmap, so it doesn't need releasing.
    """
    h.label_memmap = None
    h.im_memmap = None
    h.im_frangi_memmap = None
    h.im_marker_memmap = None
    h.im_distance_memmap = None
    gc.collect()


# ---------------------------------------------------------------------------
# Filter smoke
# ---------------------------------------------------------------------------


def _run_filter(im_info, *, device: str) -> np.ndarray:
    """Run Filter end-to-end on ``device`` and return a numpy copy of the output."""
    filt = Filter(im_info, FrangiConfig(device=device), num_t=2)
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


def test_filter_smoke_3d(make_imageinfo_3d) -> None:
    """``Filter.run()`` end-to-end on MPS — 3D path doesn't crash, shape/dtype match CPU.

    Uses the same yeast fixture as the CPU test suite (T=2, Z=17,
    Y=192, X=279). The CPU baseline is rerun in-process so the
    parity check is robust to fixture-resolution drift.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_3d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_3d(), device="mps")

    assert mps_out.shape == cpu_out.shape, (
        f"MPS output shape {mps_out.shape} != CPU shape {cpu_out.shape}"
    )
    assert mps_out.dtype == cpu_out.dtype == np.float32
    assert np.isfinite(mps_out).all(), "MPS output contains NaN/Inf"
    assert mps_out.min() >= 0.0, "Frangi output should be non-negative"


def test_filter_smoke_2d(make_imageinfo_2d) -> None:
    """``Filter.run()`` end-to-end on MPS — 2D path with LoG fusion.

    The 2D path additionally fuses a multi-scale LoG response, so this
    exercises ``gaussian_laplace`` on the MPS shim in addition to the
    Hessian/Frangi machinery.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_2d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_2d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.float32
    assert np.isfinite(mps_out).all()
    assert mps_out.min() >= 0.0


def test_filter_smoke_synthetic_3d(tmp_path) -> None:
    """``Filter.run()`` on a small purely-synthetic 3D volume — no real fixture.

    Exists so the smoke check has an independent fast path even if the
    yeast fixture is unavailable for some reason. The synthetic volume
    is shape (T=2, Z=8, Y=32, X=32) with a Gaussian-blurred bright tube —
    small enough to finish in well under a second on any MPS device.
    The OME-TIFF metadata mirrors the canonical fixture layout
    (``tests/fixtures/_generate.py``) so ImInfo's verifier accepts it
    without any per-test wrangling.
    """
    _require_mps()
    pytest.importorskip("tifffile")

    import tifffile

    from nellie.im_info import load_image

    rng = np.random.default_rng(seed=0)
    shape_zyx = (8, 32, 32)
    arr = rng.standard_normal(shape_zyx).astype(np.float32)
    # Add a bright Z-perpendicular tube so Frangi has a positive structure.
    z_idx, y_idx, x_idx = np.indices(shape_zyx)
    tube = np.exp(-(((z_idx - 4) ** 2 + (x_idx - 16) ** 2) / 4.0))
    arr = arr + tube.astype(np.float32) * 5.0

    # Wrap as (T=2, Z=8, Y=32, X=32) OME-TIFF — same metadata shape as
    # the canonical yeast fixture so ImInfo's verifier picks it up
    # without auto-detection drift.
    arr_t = np.stack([arr, arr], axis=0)
    src = tmp_path / "synth_3d.ome.tif"
    tifffile.imwrite(
        src,
        arr_t,
        photometric="minisblack",
        metadata={
            "axes": "TZYX",
            "PhysicalSizeX": 0.0655,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.0655,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 0.25,
            "PhysicalSizeZUnit": "µm",
            "TimeIncrement": 1.0,
            "TimeIncrementUnit": "s",
        },
    )

    info = load_image(src)
    out = _run_filter(info, device="mps")

    assert out.shape == arr_t.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0


# ---------------------------------------------------------------------------
# Label smoke
# ---------------------------------------------------------------------------


def _run_label(im_info, *, device: str) -> np.ndarray:
    """Run Label end-to-end on ``device`` and return a numpy copy of the labels.

    The factory fixtures (``make_label_imageinfo_*``) pre-populate the
    Frangi memmap from a session cache, so this only pays the Label
    cost — not Filter on top.
    """
    lbl = Label(im_info, LabelConfig(device=device), num_t=2)
    lbl.run()
    out = np.asarray(lbl.instance_label_memmap).copy()
    _release_label(lbl)
    return out


def test_label_smoke_3d(make_label_imageinfo_3d) -> None:
    """``Label.run()`` end-to-end on MPS — 3D path doesn't crash, shape/dtype match CPU.

    Exercises the convolutional ``uniform_filter`` call inside
    ``_get_labels`` on the MPS shim, alongside the structural
    ``binary_fill_holes`` and ``label`` round-trips back to scipy on
    CPU. Per PRD #140 § Implementation Decisions, labelling is a
    *partial-acceleration* story: only ``uniform_filter`` runs on MPS.

    The CPU baseline is rerun in-process so the parity check is robust
    to fixture-resolution drift across machines / torch versions.
    """
    _require_mps()

    cpu_out = _run_label(make_label_imageinfo_3d(), device="cpu")
    mps_out = _run_label(make_label_imageinfo_3d(), device="mps")

    assert mps_out.shape == cpu_out.shape, (
        f"MPS labels shape {mps_out.shape} != CPU shape {cpu_out.shape}"
    )
    assert mps_out.dtype == cpu_out.dtype == np.int32
    # Labels are non-negative integer IDs — background is 0, foreground IDs > 0.
    assert mps_out.min() == 0
    assert (mps_out == 0).any(), "Expected at least some background voxels"


def test_label_smoke_2d(make_label_imageinfo_2d) -> None:
    """``Label.run()`` end-to-end on MPS — 2D path.

    The 2D path skips ``binary_fill_holes`` (only 3D fills holes —
    see ``Label._get_labels``) so it exercises a slightly smaller op
    surface than the 3D smoke. Same shape/dtype contract.
    """
    _require_mps()

    cpu_out = _run_label(make_label_imageinfo_2d(), device="cpu")
    mps_out = _run_label(make_label_imageinfo_2d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.int32
    assert mps_out.min() == 0
    assert (mps_out == 0).any()


# ---------------------------------------------------------------------------
# Network smoke
# ---------------------------------------------------------------------------


def _run_network(im_info, *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Network end-to-end on ``device`` and return numpy copies of the three outputs.

    Returns ``(skel, pixel_class, skel_relabelled)``. The factory
    fixtures (``make_network_imageinfo_*``) pre-populate the Frangi and
    Label memmaps from a session cache, so this only pays the Network
    cost — not Filter + Label on top.
    """
    net = Network(im_info, NetworkConfig(device=device), num_t=2)
    net.run()
    skel = np.asarray(net.skel_memmap).copy()
    pc = np.asarray(net.pixel_class_memmap).copy()
    relab = np.asarray(net.skel_relabelled_memmap).copy()
    _release_network(net)
    return skel, pc, relab


def test_network_smoke_3d(make_network_imageinfo_3d) -> None:
    """``Network.run()`` end-to-end on MPS — 3D path doesn't crash, shape/dtype match CPU.

    Exercises the convolutional ``ndi.convolve`` call inside
    ``_get_pixel_class_impl`` on the MPS shim, alongside the structural
    ``ndi.label`` call inside ``_get_branch_skel_labels`` (which
    round-trips to scipy on CPU via the shim). Per PRD #140 §
    Implementation Decisions, networking is a *partial-acceleration*
    story: only the convolutional ops run on MPS; the structural
    ``label``, ``binary_fill_holes``, and skeletonization round-trip to
    scipy on CPU.

    The CPU baseline is rerun in-process so the parity check is robust
    to fixture-resolution drift across machines / torch versions.
    """
    _require_mps()

    skel_cpu, pc_cpu, relab_cpu = _run_network(
        make_network_imageinfo_3d(), device="cpu"
    )
    skel_mps, pc_mps, relab_mps = _run_network(
        make_network_imageinfo_3d(), device="mps"
    )

    # Skeleton image: int32 labels, background = 0.
    assert skel_mps.shape == skel_cpu.shape, (
        f"MPS skel shape {skel_mps.shape} != CPU shape {skel_cpu.shape}"
    )
    assert skel_mps.dtype == skel_cpu.dtype == np.int32

    # Pixel-class image: uint8 with values in {0, 1, 2, 3, 4}
    # (background, isolated, tip, edge, junction-clipped).
    assert pc_mps.shape == pc_cpu.shape
    assert pc_mps.dtype == pc_cpu.dtype == np.uint8
    assert pc_mps.min() == 0
    assert pc_mps.max() <= 4

    # Skeleton relabelled image: uint32, background = 0.
    assert relab_mps.shape == relab_cpu.shape
    assert relab_mps.dtype == relab_cpu.dtype == np.uint32
    assert relab_mps.min() == 0


def test_network_smoke_2d(make_network_imageinfo_2d) -> None:
    """``Network.run()`` end-to-end on MPS — 2D path.

    The 2D path uses a (3, 3) convolution kernel instead of (3, 3, 3),
    so it exercises a slightly different shim dispatch (``conv2d`` vs
    ``conv3d``). Same shape/dtype contract.
    """
    _require_mps()

    skel_cpu, pc_cpu, relab_cpu = _run_network(
        make_network_imageinfo_2d(), device="cpu"
    )
    skel_mps, pc_mps, relab_mps = _run_network(
        make_network_imageinfo_2d(), device="mps"
    )

    assert skel_mps.shape == skel_cpu.shape
    assert skel_mps.dtype == skel_cpu.dtype == np.int32

    assert pc_mps.shape == pc_cpu.shape
    assert pc_mps.dtype == pc_cpu.dtype == np.uint8
    assert pc_mps.min() == 0
    assert pc_mps.max() <= 4

    assert relab_mps.shape == relab_cpu.shape
    assert relab_mps.dtype == relab_cpu.dtype == np.uint32
    assert relab_mps.min() == 0


# ---------------------------------------------------------------------------
# HuMomentTracking smoke
# ---------------------------------------------------------------------------


def _run_hu(im_info, *, device: str) -> np.ndarray:
    """Run HuMomentTracking end-to-end on ``device`` and return the saved flow array.

    The factory fixtures (``make_hu_imageinfo_*``) pre-populate Frangi /
    Label / Marker / Distance memmaps from a session cache, so this only
    pays the HuMomentTracking cost — not the four upstream stages.

    Returns the contents of the flow vector ``.npy`` written by
    ``HuMomentTracking._run_hu_tracking``. Shape is ``(N, 6)`` for 2D
    fixtures and ``(N, 8)`` for 3D fixtures (see
    :mod:`tests.test_hu_tracking` for the schema characterization).
    """
    h = HuMomentTracking(im_info, HuMomentTrackingConfig(device=device), num_t=2)
    h.run()
    flow = np.load(h.flow_vector_array_path)
    _release_hu(h)
    return flow


def test_hu_tracking_smoke_3d(make_hu_imageinfo_3d) -> None:
    """``HuMomentTracking.run()`` end-to-end on MPS — 3D path doesn't crash, schema matches CPU.

    Exercises the convolutional ``ndi.maximum_filter`` call inside
    ``_get_frame_features`` (distance dilation) on the MPS shim,
    alongside the per-frame moment / Hu / cost-matrix math which runs
    on the active xp namespace. Per PRD #140 § Implementation
    Decisions, hu_tracking is the lightest of the four onboarded stages
    on the convolutional axis but exercises the most ``xp`` ops in the
    matching pipeline (moment math, distance broadcasting, z-score
    normalization, dense cost-matrix construction).

    Schema parity (column count + integer-vs-float column dtype) is the
    contract this smoke pins; numerical equivalence is owned by
    :mod:`tests.test_mps_equivalence`.

    Note on the float64 → float32 → float16 precision cascade specific to
    this stage: the moment-distance matrix in ``_get_difference_matrix``
    casts to ``xp.float64`` (which silently coerces to ``float32`` on
    MPS), then the cost matrix accumulates in ``xp.float16``. This is a
    documented determinism risk for hu_tracking on MPS — see
    :mod:`tests.test_mps_equivalence` for the calibrated tolerances.
    """
    _require_mps()

    cpu_flow = _run_hu(make_hu_imageinfo_3d(), device="cpu")
    mps_flow = _run_hu(make_hu_imageinfo_3d(), device="mps")

    # Shape contract: 3D = (N, 8) — columns ``[t, z, y, x, dz, dy, dx, cost]``.
    # ``N`` may differ between CPU and MPS by a few rows on borderline
    # cost-cutoff matches (the float64→float32→float16 stack drifts
    # marginal scores), so assert the column count, not the row count.
    assert mps_flow.ndim == cpu_flow.ndim == 2
    assert mps_flow.shape[1] == cpu_flow.shape[1] == 8
    # Empty-vs-populated dtype contract is part of the wiki schema for
    # this module — populated arrays come out float64 (np.column_stack
    # upcast from int64 + float32 mix); empty arrays come out float32
    # (the fallback path in ``_run_hu_tracking``).
    if mps_flow.shape[0] > 0:
        assert mps_flow.dtype == np.float64
    else:
        assert mps_flow.dtype == np.float32


def test_hu_tracking_smoke_2d(make_hu_imageinfo_2d) -> None:
    """``HuMomentTracking.run()`` end-to-end on MPS — 2D path doesn't crash, schema matches CPU.

    The 2D path uses a (3, 3) ``maximum_filter`` footprint instead of
    (3, 3, 3) and computes 6 Hu moments per marker instead of 18 (no
    orthogonal projections needed). Same shape / dtype contract as the
    3D path with one column fewer (``(N, 6)``: ``[t, y, x, dy, dx, cost]``).
    """
    _require_mps()

    cpu_flow = _run_hu(make_hu_imageinfo_2d(), device="cpu")
    mps_flow = _run_hu(make_hu_imageinfo_2d(), device="mps")

    assert mps_flow.ndim == cpu_flow.ndim == 2
    assert mps_flow.shape[1] == cpu_flow.shape[1] == 6
    if mps_flow.shape[0] > 0:
        assert mps_flow.dtype == np.float64
    else:
        assert mps_flow.dtype == np.float32
