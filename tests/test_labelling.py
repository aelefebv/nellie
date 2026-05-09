"""Characterization tests for ``nellie.segmentation.labelling.Label``.

Pins the wiki-documented invariants on both the 3D and 2D paths:
- Output dtype / background / dense IDs
- Min-area pruning honors anisotropic ``min_radius_um``
- ``min_radius_um`` is floored at ``x_res`` in the constructor
- Input memmaps are not mutated
- Full-volume vs chunked-Z produces the same component count per frame
- Cross-chunk stitching: an object straddling a chunk boundary stays one label
- Threshold paths: ``otsu_thresh_intensity`` and explicit ``threshold`` are honored

The Frangi memmap that ``Label`` consumes is precomputed once per session
by ``conftest.frangi_3d_path`` / ``frangi_2d_path``; per-test ImInfos
copy the memmap into a fresh working directory so each test gets an
isolated ``im_instance_label`` target.
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nellie.im_info import ImInfo, load_image
from nellie.segmentation.labelling import Label, LabelConfig


# Bar-shaped 3D fixture: a 5×5 cross at (Y=6..10, X=6..10) extruded along
# Z=3..5. With ``chunk_z=3`` the boundaries fall at Z=3 and Z=6, so the
# bar straddles both. With ``chunk_z=4`` the boundary falls at Z=4, so
# the bar straddles that single boundary. Either way, correct
# union-find stitching collapses the chunked outputs to one foreground
# label per frame.
_SYNTHETIC_SHAPE = (2, 8, 16, 16)
_SYNTHETIC_BAR_Z = slice(3, 6)
_SYNTHETIC_BAR_YX = (slice(6, 11), slice(6, 11))


def _release_label(lbl: Label) -> None:
    """Drop a Label's memmap references and force gc.

    Mirrors ``test_filtering._release_filter``. Required on Windows: the
    instance-label memmap can stay file-locked until the handles are
    dropped, blocking any later overwrite of the same path.
    """
    lbl.instance_label_memmap = None
    lbl.frangi_memmap = None
    lbl.im_memmap = None
    gc.collect()


def _run_label(info: ImInfo, **kwargs) -> np.ndarray:
    """Run ``Label`` on ``info`` and return a copy of the on-disk labels."""
    lbl = Label(info, LabelConfig(device="cpu", **kwargs), num_t=2)
    lbl.run()
    out = np.asarray(lbl.instance_label_memmap).copy()
    _release_label(lbl)
    return out


def _label_count_per_frame(labels: np.ndarray) -> list[int]:
    """Return the number of distinct foreground labels in each timepoint."""
    return [int((np.unique(labels[t]) != 0).sum()) for t in range(labels.shape[0])]


def _build_synthetic_label_iminfo(workdir: Path) -> ImInfo:
    """Create a 3D ImInfo with a hand-built raw + Frangi memmap.

    The synthetic Frangi memmap has a noisy background (so the
    log10-domain triangle/Otsu threshold is well-defined) plus a bright
    bar at Z=3..5, Y=6..10, X=6..10. The raw image mirrors the bar so
    the optional intensity-mask path also has signal.

    Returns an ImInfo ready for ``Label(info)`` with the Frangi memmap
    already on disk under ``info.pipeline_paths['im_preprocessed']``.
    """
    raw = np.zeros(_SYNTHETIC_SHAPE, dtype=np.uint16)
    raw[:, _SYNTHETIC_BAR_Z, _SYNTHETIC_BAR_YX[0], _SYNTHETIC_BAR_YX[1]] = 4000

    raw_path = workdir / "synthetic_bar.ome.tif"
    tifffile.imwrite(
        raw_path,
        raw,
        photometric="minisblack",
        metadata={
            "axes": "TZYX",
            "PhysicalSizeX": 0.1,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.1,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 0.1,
            "PhysicalSizeZUnit": "µm",
            "TimeIncrement": 1.0,
            "TimeIncrementUnit": "s",
        },
    )

    info = load_image(raw_path)

    rng = np.random.default_rng(42)
    frangi = rng.uniform(0.005, 0.05, size=_SYNTHETIC_SHAPE).astype(np.float32)
    frangi[:, _SYNTHETIC_BAR_Z, _SYNTHETIC_BAR_YX[0], _SYNTHETIC_BAR_YX[1]] = (
        rng.uniform(0.5, 1.0, size=(2, 3, 5, 5)).astype(np.float32)
    )

    out_path = info.pipeline_paths["im_preprocessed"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(
        out_path, frangi, photometric="minisblack", metadata={"axes": "TZYX"}
    )
    return info


# Module-scoped: run Label once on the 3D fixture and share the labels
# across the read-only invariant tests (dtype, background, dense IDs).

@pytest.fixture(scope="module")
def labels_3d(make_label_imageinfo_3d_module) -> np.ndarray:
    info = make_label_imageinfo_3d_module()
    return _run_label(info)


@pytest.fixture(scope="module")
def labels_2d(make_label_imageinfo_2d_module) -> np.ndarray:
    info = make_label_imageinfo_2d_module()
    return _run_label(info)


# -------------------------------------------------------------------------
# 3D path
# -------------------------------------------------------------------------


def test_label_runs_end_to_end_3d(labels_3d, imageinfo_3d) -> None:
    assert labels_3d.shape == imageinfo_3d.shape


def test_output_dtype_int32_3d(labels_3d) -> None:
    assert labels_3d.dtype == np.int32


def test_background_is_zero_3d(labels_3d) -> None:
    """Background must be exactly 0 (not e.g. -1 or any other sentinel)."""
    # At least some background voxels exist and they're all zero.
    assert (labels_3d == 0).any()
    # Every label id 0 corresponds to background; no negative values.
    assert labels_3d.min() == 0


def test_label_ids_dense_per_frame_3d(labels_3d) -> None:
    """Unique IDs in each frame form a contiguous ``[0..N]`` range."""
    for t in range(labels_3d.shape[0]):
        ids = np.unique(labels_3d[t])
        assert ids.min() == 0
        # IDs must be 0, 1, 2, ..., N with no gaps.
        assert np.array_equal(ids, np.arange(ids.size, dtype=ids.dtype)), (
            f"t={t}: label IDs not dense ({ids.tolist()})"
        )


def test_min_area_pruning_drops_small_objects(make_label_imageinfo_3d) -> None:
    """A larger ``min_radius_um`` removes objects below the sphere-volume threshold."""
    small = _run_label(make_label_imageinfo_3d(), min_radius_um=0.25)
    large = _run_label(make_label_imageinfo_3d(), min_radius_um=1.0)

    small_counts = _label_count_per_frame(small)
    large_counts = _label_count_per_frame(large)

    assert any(lg < sm for lg, sm in zip(large_counts, small_counts)), (
        f"Larger min_radius_um did not prune any objects "
        f"(small={small_counts}, large={large_counts})"
    )


def test_anisotropic_min_area_pixel_volume(make_label_imageinfo_3d) -> None:
    """min_area_pixels uses anisotropic dim_res when computing the sphere volume.

    The yeast-3d fixture has X=Y=0.0655 µm, Z=0.25 µm. For min_radius_um=0.25,
    the anisotropic sphere volume is (4/3)π(0.25)^3 µm³ divided by
    (0.0655 × 0.0655 × 0.25) µm³/voxel ≈ 62 voxels. If Z were treated
    as isotropic to X (0.0655), the same formula would give ~233 voxels.
    """
    info = make_label_imageinfo_3d()
    lbl = Label(info, LabelConfig(device="cpu", min_radius_um=0.25), num_t=2)
    x_res = info.dim_res["X"]
    y_res = info.dim_res["Y"]
    z_res = info.dim_res["Z"]
    expected_volume_um3 = (4.0 / 3.0) * np.pi * (lbl.min_radius_um ** 3)
    expected_voxels = int(np.ceil(expected_volume_um3 / (x_res * y_res * z_res)))
    isotropic_voxels = int(np.ceil(expected_volume_um3 / (x_res ** 3)))
    assert lbl.min_area_pixels == expected_voxels
    # Sanity: anisotropic and isotropic give different answers for this fixture.
    assert lbl.min_area_pixels != isotropic_voxels


def test_min_radius_um_floored_at_x_res(imageinfo_3d) -> None:
    """``min_radius_um`` smaller than ``x_res`` is silently floored at ``x_res``."""
    x_res = float(imageinfo_3d.dim_res["X"])
    sub_pixel = x_res / 1000.0
    lbl = Label(imageinfo_3d, LabelConfig(device="cpu", min_radius_um=sub_pixel), num_t=2)
    assert lbl.min_radius_um == pytest.approx(x_res)


def test_input_memmaps_not_mutated_3d(make_label_imageinfo_3d) -> None:
    """Hash both raw OME-TIFF and Frangi memmap before/after Label.run()."""
    info = make_label_imageinfo_3d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])

    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()

    _run_label(info)

    raw_after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_after = hashlib.sha256(frangi_path.read_bytes()).hexdigest()

    assert raw_before == raw_after, "Label mutated the raw input memmap"
    assert frangi_before == frangi_after, "Label mutated the Frangi input memmap"


def test_full_vs_chunked_z_label_count_equivalence(make_label_imageinfo_3d) -> None:
    """Full-volume and chunked-Z runs produce the same component count per frame.

    Mask byte-equivalence is not stable across chunked/full: the
    ``uniform_filter`` smoothing pass in ``_get_labels`` runs on different
    neighborhoods at chunk boundaries, so individual mask voxels can
    flicker. The behavior contract the wiki names is that chunking does
    not fragment objects via ID assignment — i.e. the *count* of
    components per frame matches.
    """
    full = _run_label(make_label_imageinfo_3d())
    chunked = _run_label(make_label_imageinfo_3d(), chunk_z=4)

    full_counts = _label_count_per_frame(full)
    chunked_counts = _label_count_per_frame(chunked)
    assert full_counts == chunked_counts, (
        f"Chunked-Z fragmented or merged objects "
        f"(full={full_counts}, chunked={chunked_counts})"
    )


def test_cross_chunk_label_stitching(tmp_path: Path) -> None:
    """An object straddling a chunk boundary stays a single label, not two.

    Synthetic 3D fixture: a bright bar through Z=3..5. With ``chunk_z=3``
    the chunk boundaries fall at Z=3 and Z=6 (so the bar sits across
    two chunk seams). Without union-find stitching, this would label
    the bar as two or three separate components.

    Note: the smoothing pass in ``_get_labels`` (``uniform_filter`` size 3
    then ``> 0.5``) erodes the bar's corner voxels, so we don't assert
    the bar region is fully non-zero — only that all surviving
    foreground voxels in the bar region carry the same label id.
    """
    info = _build_synthetic_label_iminfo(tmp_path)
    labels = _run_label(info, chunk_z=3, min_radius_um=0.1)

    # Per-frame: exactly one foreground label.
    counts = _label_count_per_frame(labels)
    assert counts == [1, 1], f"Cross-chunk stitching failed (counts={counts})"

    # All foreground voxels in the bar region carry the same id (i.e. the
    # bar wasn't fragmented across chunks). Background corners eroded by
    # the smoothing pass are tolerated.
    for t in range(labels.shape[0]):
        bar = labels[t, _SYNTHETIC_BAR_Z, _SYNTHETIC_BAR_YX[0], _SYNTHETIC_BAR_YX[1]]
        bar_fg = bar[bar > 0]
        assert bar_fg.size > 0, f"t={t}: bar entirely missing from labels"
        assert np.unique(bar_fg).size == 1, (
            f"t={t}: bar fragmented across chunks "
            f"(found ids {np.unique(bar_fg).tolist()})"
        )


def test_otsu_thresh_intensity_changes_output(make_label_imageinfo_3d) -> None:
    """``otsu_thresh_intensity=True`` engages the intensity-mask path:
    the raw intensity Otsu threshold both gates ``_compute_frangi_threshold``
    sampling and multiplies into the per-frame Frangi during labeling
    (see ``Label._run_frame_full_volume``). The output therefore differs
    from the pure-Frangi baseline.

    Asserts difference (in count or coverage), not direction. On the
    yeast fixture the count happens to match between the two paths but
    the foreground voxel total shifts substantially, which proves the
    mask is being applied.
    """
    no_mask = _run_label(make_label_imageinfo_3d())
    with_mask = _run_label(make_label_imageinfo_3d(), otsu_thresh_intensity=True)

    no_mask_counts = _label_count_per_frame(no_mask)
    with_mask_counts = _label_count_per_frame(with_mask)
    no_mask_voxels = int((no_mask != 0).sum())
    with_mask_voxels = int((with_mask != 0).sum())

    differs = (
        no_mask_counts != with_mask_counts
        or no_mask_voxels != with_mask_voxels
    )
    assert differs, (
        f"otsu_thresh_intensity did not change Label output "
        f"(counts {no_mask_counts}=={with_mask_counts}, "
        f"voxels {no_mask_voxels}=={with_mask_voxels})"
    )


def test_explicit_threshold_parameter_honored(make_label_imageinfo_3d) -> None:
    """An explicit ``threshold`` engages the intensity-mask path with a
    fixed value (instead of computing it via Otsu). Output therefore
    differs from the no-threshold baseline.

    Direction of the change is fixture-dependent: the threshold is a
    raw-intensity gate that both restricts the Frangi-threshold sample
    and multiplies into the labeled Frangi, so a stricter raw threshold
    can either reduce or *increase* component count depending on how
    the resulting Frangi cutoff falls. The contract is that the
    parameter is honored — i.e. that turning it on changes the output.
    """
    info = make_label_imageinfo_3d()
    raw = np.asarray(info.im)
    median_thresh = float(np.median(raw[raw > 0]))

    no_thresh = _run_label(make_label_imageinfo_3d())
    with_thresh = _run_label(make_label_imageinfo_3d(), threshold=median_thresh)

    no_thresh_counts = _label_count_per_frame(no_thresh)
    with_thresh_counts = _label_count_per_frame(with_thresh)
    no_thresh_voxels = int((no_thresh != 0).sum())
    with_thresh_voxels = int((with_thresh != 0).sum())

    differs = (
        no_thresh_counts != with_thresh_counts
        or no_thresh_voxels != with_thresh_voxels
    )
    assert differs, (
        f"Explicit threshold={median_thresh:.2f} did not change Label output "
        f"(counts {no_thresh_counts}=={with_thresh_counts}, "
        f"voxels {no_thresh_voxels}=={with_thresh_voxels})"
    )


# -------------------------------------------------------------------------
# 2D path
# -------------------------------------------------------------------------


def test_label_runs_end_to_end_2d(labels_2d, imageinfo_2d) -> None:
    assert labels_2d.shape == imageinfo_2d.shape


def test_2d_output_invariants(labels_2d, make_label_imageinfo_2d) -> None:
    """2D mirror of the dtype / background / dense / unmutated suite."""
    # dtype
    assert labels_2d.dtype == np.int32
    # background
    assert (labels_2d == 0).any()
    assert labels_2d.min() == 0
    # dense IDs
    for t in range(labels_2d.shape[0]):
        ids = np.unique(labels_2d[t])
        assert np.array_equal(ids, np.arange(ids.size, dtype=ids.dtype))

    # input memmaps unmutated
    info = make_label_imageinfo_2d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()
    _run_label(info)
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before
    assert hashlib.sha256(frangi_path.read_bytes()).hexdigest() == frangi_before
