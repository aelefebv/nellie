"""Characterization tests for ``nellie.segmentation.mocap_marking.Markers``.

Pins the wiki-documented invariants on both the 3D and 2D paths:
- Output dtypes and value sets:
  - ``im_marker``: uint8, values ⊂ {0, 1}
  - ``im_distance``: float32, values ≥ 0, **clamped at 2 × max_radius_px**
  - ``im_border``: uint8, values ⊂ {0, 1}
- Spatial invariants:
  - ``border ∩ mask == ∅`` (border is the outside shell from
    ``dilation(mask) XOR mask``, line 440 in ``mocap_marking.py``)
  - ``marker ⊆ mask`` (peaks are gated on
    ``valid_mask = mask & (distance_im > 0)``, line 482)
- Empty-mask short-circuit: zero label volume → all-zero
  ``marker`` / ``distance`` / ``border`` outputs and no error raised
- ``use_im`` modes:
  - ``'distance'`` produces non-empty markers on the 3D fixture
  - ``'frangi'`` produces non-empty markers AND the resulting marker
    locations differ from ``'distance'`` (sanity that the two paths do
    different work)
  - ``'frangi'`` raises with a clear error when ``im_preprocessed`` is
    absent on disk (Markers should not silently no-op)
- ``low_memory`` chunked vs unchunked equivalence on BOTH 2D and 3D:
  ``_log_halo`` and ``_nms_halo`` keep chunked output byte-identical to
  the full-volume path
- Input memmaps (raw + Frangi + Label) are not mutated by ``Markers.run()``
- ``viewer.status`` is left untouched when ``viewer=None`` and is set
  exactly once per frame when ``viewer`` is a stub
- ``_run_mocap_marking`` write-path branch (line 764-775): the
  ``im_marker_memmap[:] = ...`` path fires when
  ``im_marker_memmap.shape != self.shape and im_info.no_t``; the
  ``im_marker_memmap[t] = ...`` path fires otherwise

The cross-frame state mutation in ``_run_frame``'s inner OOM cascade is
**deliberately not pinned** here — Slice 3 of PRD #84 deletes that
cascade entirely (resolved decision #2 in the dechaos report), so a test
that pinned the mutation would force its own deletion in Slice 3.

The Frangi and Label memmaps that ``Markers`` consumes are precomputed
once per session by ``conftest.frangi_*_path`` / ``conftest.label_*_path``;
per-test ImInfos copy both memmaps into a fresh working directory so
each test gets isolated ``im_marker`` / ``im_distance`` / ``im_border``
targets.
"""

from __future__ import annotations

import gc
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers


def _release_markers(m: Markers) -> None:
    """Drop a Markers' memmap references and force gc.

    Mirrors ``test_networking._release_network``. Required on Windows:
    the output memmaps can stay file-locked until the handles are
    dropped, blocking any later overwrite of the same paths.
    """
    m.im_marker_memmap = None
    m.im_distance_memmap = None
    m.im_border_memmap = None
    m.label_memmap = None
    m.im_memmap = None
    m.im_frangi_memmap = None
    gc.collect()


def _run_markers(info: ImInfo, **kwargs) -> dict[str, np.ndarray]:
    """Run ``Markers`` on ``info`` and return copies of the on-disk outputs."""
    kwargs.setdefault("device", "cpu")
    m = Markers(info, num_t=2, **kwargs)
    m.run()
    out = {
        "marker": np.asarray(m.im_marker_memmap).copy(),
        "distance": np.asarray(m.im_distance_memmap).copy(),
        "border": np.asarray(m.im_border_memmap).copy(),
    }
    _release_markers(m)
    return out


# Module-scoped: run Markers once on each fixture and share outputs
# across the read-only invariant tests (dtype, value set, contracts).

@pytest.fixture(scope="module")
def markers_outputs_3d(make_markers_imageinfo_3d_module) -> dict[str, np.ndarray]:
    info = make_markers_imageinfo_3d_module()
    return _run_markers(info)


@pytest.fixture(scope="module")
def markers_outputs_2d(make_markers_imageinfo_2d_module) -> dict[str, np.ndarray]:
    info = make_markers_imageinfo_2d_module()
    return _run_markers(info)


@pytest.fixture(scope="module")
def label_volume_3d(make_markers_imageinfo_3d_module) -> np.ndarray:
    """Module-scoped: copy of the 3D ``im_instance_label`` memmap.

    Used by the spatial-invariant tests (``border ∩ mask == ∅`` and
    ``marker ⊆ mask``) that need the upstream label volume to compute
    the foreground mask. Reading from a fresh ImInfo so the file handle
    is independent of any Markers instance's memmap.
    """
    info = make_markers_imageinfo_3d_module()
    return np.asarray(
        info.get_memmap(info.pipeline_paths["im_instance_label"])
    ).copy()


@pytest.fixture(scope="module")
def label_volume_2d(make_markers_imageinfo_2d_module) -> np.ndarray:
    """Module-scoped: copy of the 2D ``im_instance_label`` memmap (see :func:`label_volume_3d`)."""
    info = make_markers_imageinfo_2d_module()
    return np.asarray(
        info.get_memmap(info.pipeline_paths["im_instance_label"])
    ).copy()


# -------------------------------------------------------------------------
# 3D path — output dtype / contract assertions
# -------------------------------------------------------------------------


def test_markers_runs_end_to_end_3d(markers_outputs_3d, imageinfo_3d) -> None:
    assert markers_outputs_3d["marker"].shape == imageinfo_3d.shape
    assert markers_outputs_3d["distance"].shape == imageinfo_3d.shape
    assert markers_outputs_3d["border"].shape == imageinfo_3d.shape


def test_im_marker_dtype_and_value_set_3d(markers_outputs_3d) -> None:
    """``im_marker`` is uint8 with values strictly in {0, 1}."""
    marker = markers_outputs_3d["marker"]
    assert marker.dtype == np.uint8
    unique = np.unique(marker)
    assert set(unique.tolist()).issubset({0, 1}), (
        f"Unexpected im_marker values: {unique.tolist()}"
    )
    # Sanity: the fixture exercises actual peak detection.
    assert marker.sum() > 0, "3D fixture produced no markers"


def test_im_distance_dtype_and_clamp_3d(markers_outputs_3d, imageinfo_3d) -> None:
    """``im_distance`` is float32 with values ≥ 0 and ≤ ``2 × max_radius_px``.

    Pins the wiki-documented clamp at line 448 of ``mocap_marking.py``,
    which mimics the legacy KD-tree's behavior for infinities by
    truncating distances at twice the maximum object radius.
    """
    distance = markers_outputs_3d["distance"]
    assert distance.dtype == np.float32
    assert distance.min() >= 0.0
    # The default max_radius_um is 1 µm; convert to pixels using the
    # fixture's X resolution. Markers' constructor uses the same
    # division (``self.max_radius_px = self.max_radius_um / x_res``).
    x_res = float(imageinfo_3d.dim_res["X"])
    max_radius_px = 1.0 / x_res
    upper_bound = 2.0 * max_radius_px
    assert distance.max() <= upper_bound + 1e-5, (
        f"im_distance max {distance.max()} exceeds 2 * max_radius_px = "
        f"{upper_bound}"
    )


def test_im_border_dtype_and_value_set_3d(markers_outputs_3d) -> None:
    """``im_border`` is uint8 with values strictly in {0, 1}."""
    border = markers_outputs_3d["border"]
    assert border.dtype == np.uint8
    unique = np.unique(border)
    assert set(unique.tolist()).issubset({0, 1}), (
        f"Unexpected im_border values: {unique.tolist()}"
    )
    # Sanity: the fixture has objects so border is non-empty.
    assert border.sum() > 0, "3D fixture produced no border voxels"


def test_border_disjoint_from_mask_3d(markers_outputs_3d, label_volume_3d) -> None:
    """``border ∩ mask == ∅`` (border is the outside shell of the mask).

    ``_distance_im`` computes ``border = dilation(mask) XOR mask`` so by
    construction the border voxels lie strictly outside the foreground
    mask. Pin the contract on the on-disk outputs.
    """
    border = markers_outputs_3d["border"]
    mask = label_volume_3d > 0
    overlap = (border > 0) & mask
    assert not overlap.any(), (
        f"border ∩ mask is non-empty ({int(overlap.sum())} voxels); "
        "border should be the strict outside shell of the mask"
    )


def test_markers_inside_objects_3d(markers_outputs_3d, label_volume_3d) -> None:
    """``marker ⊆ mask`` — peaks must be inside segmented objects.

    ``_local_max_peak`` gates peaks on
    ``valid_mask = mask & (distance_im > 0)`` (line 482), so every
    detected marker voxel must lie inside an object label.
    """
    marker = markers_outputs_3d["marker"]
    mask = label_volume_3d > 0
    outside = (marker > 0) & ~mask
    assert not outside.any(), (
        f"{int(outside.sum())} marker voxels lie outside any object label"
    )


def test_input_memmaps_not_mutated_3d(make_markers_imageinfo_3d) -> None:
    """Hash raw, Frangi, and Label inputs before/after ``Markers.run()``."""
    info = make_markers_imageinfo_3d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])
    label_path = Path(info.pipeline_paths["im_instance_label"])

    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()
    label_before = hashlib.sha256(label_path.read_bytes()).hexdigest()

    _run_markers(info)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before, (
        "Markers mutated the raw input memmap"
    )
    assert hashlib.sha256(frangi_path.read_bytes()).hexdigest() == frangi_before, (
        "Markers mutated the Frangi input memmap"
    )
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == label_before, (
        "Markers mutated the Label input memmap"
    )


# -------------------------------------------------------------------------
# Behavior tests: short-circuits, mode switches, error paths
# -------------------------------------------------------------------------


def test_empty_mask_short_circuits_to_zero_outputs(make_markers_imageinfo_3d) -> None:
    """All-zero label volume → all-zero ``marker`` / ``distance`` / ``border`` (no error).

    ``_run_frame_impl`` (line 663) checks ``not xp_mod.any(mask_frame)``
    and returns zero arrays without invoking the distance transform or
    LoG pipeline. Pins the wiki-documented short-circuit.
    """
    info = make_markers_imageinfo_3d()
    label_path = info.pipeline_paths["im_instance_label"]

    # Zero out the label memmap on disk so every frame triggers the
    # short-circuit. Using r+ avoids re-allocating the file (which would
    # change its dtype/shape metadata).
    label_mm = tifffile.memmap(label_path, mode="r+")
    label_mm[:] = 0
    label_mm.flush()
    del label_mm
    gc.collect()

    out = _run_markers(info)
    assert out["marker"].sum() == 0, "Empty mask should produce no markers"
    assert out["distance"].max() == 0.0, "Empty mask should produce zero distance"
    assert out["border"].sum() == 0, "Empty mask should produce no border voxels"
    # Dtypes are still pinned in the empty path.
    assert out["marker"].dtype == np.uint8
    assert out["distance"].dtype == np.float32
    assert out["border"].dtype == np.uint8


def test_use_im_distance_produces_markers_3d(make_markers_imageinfo_3d) -> None:
    """``use_im='distance'`` (default) produces non-empty markers on the 3D fixture."""
    out = _run_markers(make_markers_imageinfo_3d(), use_im="distance")
    assert out["marker"].sum() > 0, (
        "use_im='distance' produced no markers on the 3D fixture"
    )


def test_use_im_frangi_differs_from_distance_3d(make_markers_imageinfo_3d) -> None:
    """``use_im='frangi'`` produces non-empty markers AND differs from ``'distance'``.

    Sanity check that the two paths actually do different work — if
    they produced byte-identical outputs, one branch would be dead.
    """
    out_distance = _run_markers(make_markers_imageinfo_3d(), use_im="distance")
    out_frangi = _run_markers(make_markers_imageinfo_3d(), use_im="frangi")
    assert out_frangi["marker"].sum() > 0, (
        "use_im='frangi' produced no markers on the 3D fixture"
    )
    assert not np.array_equal(out_distance["marker"], out_frangi["marker"]), (
        "use_im='distance' and use_im='frangi' produced byte-identical "
        "marker outputs — one of the branches is not actually doing "
        "different work"
    )


def test_use_im_frangi_raises_when_frangi_absent(make_imageinfo_3d, label_3d_path) -> None:
    """``use_im='frangi'`` raises (not silent no-op) when ``im_preprocessed`` is missing.

    Constructs a Markers ImInfo with the Label memmap copied in but no
    Frangi memmap on disk. ``_allocate_memory`` calls
    ``self.im_info.get_memmap(im_preprocessed)`` which fails when the
    file does not exist; the error must propagate (the wiki invariant
    is that Markers should not silently no-op).
    """
    # Use the bare 3D ImInfo factory and copy in only the Label memmap;
    # ``im_preprocessed`` is intentionally absent so the Frangi branch
    # in ``_allocate_memory`` (line 391) cannot read it.
    info = make_imageinfo_3d()
    label_dst = Path(info.pipeline_paths["im_instance_label"])
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(label_3d_path, label_dst)
    assert not Path(info.pipeline_paths["im_preprocessed"]).exists(), (
        "Test setup error: im_preprocessed should not exist for this test"
    )

    # Markers' run() catches OOM/GPU-unavailable errors and continues
    # the cascade, but a missing-file error should propagate cleanly.
    with pytest.raises(Exception) as exc_info:
        Markers(info, num_t=2, device="cpu", use_im="frangi").run()

    # The actual error type is FileNotFoundError from tifffile.memmap
    # (or RuntimeError if the inner check at line 678 is reached on a
    # different code path). Either way the message must reference the
    # missing Frangi file or the Frangi flag itself.
    msg = str(exc_info.value).lower()
    assert (
        "frangi" in msg
        or "im_preprocessed" in msg
        or "no such file" in msg
        or "not found" in msg
    ), (
        f"Expected error mentioning Frangi/im_preprocessed/missing file; "
        f"got {type(exc_info.value).__name__}: {exc_info.value!r}"
    )


def test_low_memory_matches_full_2d(make_markers_imageinfo_2d) -> None:
    """Full-volume and chunked-low-memory runs produce identical outputs (2D).

    Pins the wiki-documented ``_log_halo`` + ``_nms_halo`` correctness:
    the chunked LoG and chunked NMS paths use just enough halo voxels
    to keep results byte-identical to the unchunked path. Pinning all
    three outputs (``marker``, ``distance``, ``border``) ensures both
    the LoG and NMS halos are exercised — distance is unchanged by
    chunking (it's a global EDT) but marker positions and border
    voxels depend on the per-chunk halo math.
    """
    full = _run_markers(make_markers_imageinfo_2d(), low_memory=False)
    # Force multi-chunk processing on the small yeast-2d fixture.
    chunked = _run_markers(
        make_markers_imageinfo_2d(), low_memory=True, max_chunk_voxels=20_000
    )
    assert np.array_equal(full["marker"], chunked["marker"]), (
        "im_marker differs between full and chunked 2D runs"
    )
    assert np.array_equal(full["distance"], chunked["distance"]), (
        "im_distance differs between full and chunked 2D runs"
    )
    assert np.array_equal(full["border"], chunked["border"]), (
        "im_border differs between full and chunked 2D runs"
    )


def test_low_memory_matches_full_3d(make_markers_imageinfo_3d) -> None:
    """Full-volume and chunked-low-memory runs produce identical outputs (3D).

    Same contract as :func:`test_low_memory_matches_full_2d`, but on
    the 3D fixture so ``_log_halo`` and ``_nms_halo`` are exercised
    along all three spatial axes.
    """
    full = _run_markers(make_markers_imageinfo_3d(), low_memory=False)
    chunked = _run_markers(
        make_markers_imageinfo_3d(), low_memory=True, max_chunk_voxels=50_000
    )
    assert np.array_equal(full["marker"], chunked["marker"]), (
        "im_marker differs between full and chunked 3D runs"
    )
    assert np.array_equal(full["distance"], chunked["distance"]), (
        "im_distance differs between full and chunked 3D runs"
    )
    assert np.array_equal(full["border"], chunked["border"]), (
        "im_border differs between full and chunked 3D runs"
    )


# -------------------------------------------------------------------------
# Viewer / write-path branches
# -------------------------------------------------------------------------


class _StubViewer:
    """Minimal viewer stub: records every write to ``status``."""

    def __init__(self) -> None:
        self.status_writes: list[str] = []

    @property
    def status(self) -> str:
        return self.status_writes[-1] if self.status_writes else ""

    @status.setter
    def status(self, value: str) -> None:
        self.status_writes.append(value)


def test_viewer_status_callback(make_markers_imageinfo_2d) -> None:
    """``viewer.status`` is set once per frame; ``viewer=None`` is a no-op.

    The viewer-update branch in ``_run_mocap_marking`` (line 759-760)
    only runs when ``self.viewer is not None``. Pin both arms in one
    test:
      - Build a Markers with ``viewer=None`` and assert the run
        completes cleanly (no AttributeError).
      - Build a Markers with a stub viewer and assert
        ``status`` is written exactly ``num_t`` times.
    """
    # No-op arm: viewer=None.
    info_none = make_markers_imageinfo_2d()
    m_none = Markers(info_none, num_t=2, device="cpu", viewer=None)
    m_none.run()  # must not raise
    _release_markers(m_none)

    # Stub-viewer arm: assert per-frame writes.
    info_stub = make_markers_imageinfo_2d()
    stub = _StubViewer()
    m_stub = Markers(info_stub, num_t=2, device="cpu", viewer=stub)
    m_stub.run()
    assert len(stub.status_writes) == 2, (
        f"Expected viewer.status set once per frame (2 writes); "
        f"got {len(stub.status_writes)}: {stub.status_writes}"
    )
    # Sanity: each message identifies the frame.
    assert "Frame: 1 of 2" in stub.status_writes[0]
    assert "Frame: 2 of 2" in stub.status_writes[1]
    _release_markers(m_stub)


# -------------------------------------------------------------------------
# Single-frame shape-branch test for _run_mocap_marking
# -------------------------------------------------------------------------


class _StubMemmap:
    """Numpy-array-backed stand-in for a tifffile memmap.

    Supports ``__setitem__`` (so ``[:]`` and ``[t]`` writes from
    ``_run_mocap_marking`` work) and ``flush()`` (called after each
    write). Used to wedge a shape-mismatched marker memmap into a
    Markers instance so the ``[:]`` write branch fires.
    """

    def __init__(self, shape, dtype) -> None:
        self.arr = np.zeros(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def __setitem__(self, key, value) -> None:
        self.arr[key] = value

    def __getitem__(self, key):
        return self.arr[key]

    def flush(self) -> None:
        pass


def _build_single_frame_iminfo(workdir: Path) -> ImInfo:
    """Build a single-frame (no T) 2D ImInfo with Filter+Label outputs on disk.

    The raw image is YX (no T axis), so ``info.no_t == True`` and
    ``info.shape == (1, Y, X)`` (after canonical normalization).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "single_frame.ome.tif"
    raw = np.zeros((96, 96), dtype=np.uint16)
    raw[30:60, 30:60] = 1500
    tifffile.imwrite(
        raw_path,
        raw,
        photometric="minisblack",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.0655,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.0655,
            "PhysicalSizeYUnit": "µm",
        },
    )
    file_info = FileInfo(str(raw_path))
    file_info.find_metadata()
    file_info.load_metadata()
    info = ImInfo(file_info)
    Filter(info, FrangiConfig(device="cpu"), num_t=1).run()
    Label(info, num_t=1, device="cpu").run()
    gc.collect()
    return info


def test_single_frame_shape_branch_both_paths(tmp_path: Path) -> None:
    """Exercise both write paths in ``_run_mocap_marking`` (line 764-775).

    The branch is:
      ``if self.im_marker_memmap.shape != self.shape and self.im_info.no_t:``
        ``[:]`` write
      ``else:``
        ``[t]`` write

    To trigger the ``[:]`` arm we need both:
      1. ``no_t == True`` (single-frame fixture)
      2. ``im_marker_memmap.shape != self.shape``

    Under normal usage ``_normalize_memmap`` re-pads any squeezed memmap
    so the shape always matches, so the branch is exercised by
    swapping in a stub memmap with a deliberately-squeezed shape after
    ``_allocate_memory``.

    The ``[t]`` arm fires for every Markers run on the multi-frame
    yeast fixtures (already covered by ``_run_markers``); this test
    pins it explicitly on a single-frame ImInfo where the shapes match.
    """
    # ----- [:] arm -----
    info = _build_single_frame_iminfo(tmp_path / "splat_branch")
    m_splat = Markers(info, num_t=1, device="cpu")
    m_splat._set_backend("cpu")
    m_splat._set_low_memory(False)
    m_splat._allocate_memory()
    m_splat._set_default_sigmas()

    # Wedge in shape-mismatched (Y, X) stubs so the [:] branch fires.
    squeezed_shape = m_splat.shape[1:]  # drop the leading T=1
    m_splat.im_marker_memmap = _StubMemmap(squeezed_shape, np.uint8)
    m_splat.im_distance_memmap = _StubMemmap(squeezed_shape, np.float32)
    m_splat.im_border_memmap = _StubMemmap(squeezed_shape, np.uint8)

    assert m_splat.im_marker_memmap.shape != m_splat.shape, (
        "Test setup error: stub memmap shape should differ from m.shape"
    )
    assert m_splat.im_info.no_t, "Test setup error: ImInfo should report no_t"

    m_splat._run_mocap_marking()

    # If the [:] arm fired, the squeezed stubs received the squeezed
    # frame data. A non-empty bright square should produce a non-empty
    # border (the dilation-XOR-mask outside shell).
    assert m_splat.im_border_memmap.arr.shape == squeezed_shape
    assert m_splat.im_border_memmap.arr.sum() > 0, (
        "[:] write path produced an empty border on a non-empty mask"
    )

    # ----- [t] arm -----
    info_t = _build_single_frame_iminfo(tmp_path / "frame_branch")
    m_t = Markers(info_t, num_t=1, device="cpu")
    m_t._set_backend("cpu")
    m_t._set_low_memory(False)
    m_t._allocate_memory()
    m_t._set_default_sigmas()

    # Default _allocate_memory yields shape-matched memmaps (after
    # _normalize_memmap re-pads), so the else branch fires.
    assert m_t.im_marker_memmap is not None
    assert m_t.im_marker_memmap.shape == m_t.shape, (
        "Test setup error: default memmap shape should match m.shape"
    )

    m_t._run_mocap_marking()
    border_t = np.asarray(m_t.im_border_memmap)
    assert border_t.shape == m_t.shape, (
        "[t] write path produced unexpected output shape"
    )
    assert border_t.sum() > 0, (
        "[t] write path produced an empty border on a non-empty mask"
    )
    _release_markers(m_t)
    # Drop the wedged in-memory stubs explicitly so gc cleans the file
    # handles ImInfo still owns.
    m_splat.im_marker_memmap = None
    m_splat.im_distance_memmap = None
    m_splat.im_border_memmap = None
    m_splat.label_memmap = None
    m_splat.im_memmap = None
    m_splat.im_frangi_memmap = None
    gc.collect()


# -------------------------------------------------------------------------
# 2D path
# -------------------------------------------------------------------------


def test_markers_runs_end_to_end_2d(markers_outputs_2d, imageinfo_2d) -> None:
    assert markers_outputs_2d["marker"].shape == imageinfo_2d.shape
    assert markers_outputs_2d["distance"].shape == imageinfo_2d.shape
    assert markers_outputs_2d["border"].shape == imageinfo_2d.shape


def test_2d_output_invariants(
    markers_outputs_2d, label_volume_2d, imageinfo_2d
) -> None:
    """2D mirror of the dtype / value-set / contract / spatial suite."""
    marker = markers_outputs_2d["marker"]
    distance = markers_outputs_2d["distance"]
    border = markers_outputs_2d["border"]

    # dtypes
    assert marker.dtype == np.uint8
    assert distance.dtype == np.float32
    assert border.dtype == np.uint8

    # marker / border value sets
    assert set(np.unique(marker).tolist()).issubset({0, 1})
    assert set(np.unique(border).tolist()).issubset({0, 1})

    # distance ≥ 0 and ≤ 2 * max_radius_px
    assert distance.min() >= 0.0
    x_res = float(imageinfo_2d.dim_res["X"])
    upper_bound = 2.0 * (1.0 / x_res)
    assert distance.max() <= upper_bound + 1e-5

    # spatial invariants
    mask = label_volume_2d > 0
    overlap = (border > 0) & mask
    assert not overlap.any(), (
        f"2D border ∩ mask is non-empty ({int(overlap.sum())} voxels)"
    )
    outside_marker = (marker > 0) & ~mask
    assert not outside_marker.any(), (
        f"2D marker has {int(outside_marker.sum())} voxels outside any object"
    )
