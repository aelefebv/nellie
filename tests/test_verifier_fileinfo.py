"""Characterization tests for ``nellie.im_info.verifier.FileInfo``.

Slice 1 of the verifier dechaos refactor (see
``wiki/outputs/dechaos-verifier.md``). Pins the existing FileInfo
contract end-to-end via real fixture files so the implementation can
be safely refactored in subsequent slices.

Tests are organized by concern:
  A. Per-format ``dim_res`` extraction (5 ``metadata_type`` branches)
  B. Axes normalization (``_normalize_time_axis``)
  C. Validation (the 7 methods)
  D. Mutators (``change_*`` / ``select_temporal_range``)
  E. ``save_ome_tiff`` + provenance round-trip
  F. ``read_file`` 3-fallback chain
  G. ``_get_output_path`` naming strategies

Three tests carry design-quirk comments — pinning behavior we do NOT
want to silently change in later slices.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import ome_types
import pytest
import tifffile

from nellie.im_info.verifier import FileInfo

# Fixture paths are derived once at import time from the same on-disk
# locations the conftest constants point at. (The ``tests/`` directory
# isn't a package — no relative imports — so we mirror the conftest's
# layout instead of importing its constants.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"
FIXTURE_3D_PATH = _FIXTURES_DIR / "yeast_3d_t0_to_1.ome.tif"
FIXTURE_2D_PATH = _FIXTURES_DIR / "yeast_2d_t0_to_1.ome.tif"
FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH = _FIXTURES_DIR / "imagej_with_physicalsize.tif"
FIXTURE_IMAGEJ_NO_PHYSICALSIZE_PATH = _FIXTURES_DIR / "imagej_no_physicalsize.tif"
FIXTURE_RAW_TIFF_NO_RESUNIT_PATH = _FIXTURES_DIR / "raw_tiff_no_resunit.tif"
FIXTURE_RAW_TIFF_CENTIMETER_PATH = _FIXTURES_DIR / "raw_tiff_centimeter.tif"
FIXTURE_RAW_TIFF_INCH_PATH = _FIXTURES_DIR / "raw_tiff_inch.tif"


def _copy_fixture_to_tmp(source: Path, workdir: Path) -> Path:
    """Local mirror of conftest's ``copy_fixture_to_tmp`` (no relative-import option)."""
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / source.name
    shutil.copy(source, dst)
    return dst


def _loaded_file_info(source, workdir) -> FileInfo:
    """Copy fixture, construct FileInfo, run find/load — return the loaded object."""
    dst = _copy_fixture_to_tmp(source, workdir)
    fi = FileInfo(str(dst))
    fi.find_metadata()
    fi.load_metadata()
    return fi


# ============================================================
# A0. Constructor purity + prepare_output_dirs (Slice 3 boundary)
# ============================================================


def test_thin_constructor_does_no_io(tmp_path) -> None:
    """``FileInfo(filepath)`` is a thin constructor — no filesystem I/O.

    Slice 3 moved ``os.makedirs`` out of ``__init__`` into
    ``prepare_output_dirs``, called automatically from ``find_metadata``.
    The bare constructor only stores path-derived strings.

    Pinned so tests can construct ``FileInfo`` to inspect path
    computations without creating output directories on disk.
    """
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = workdir / 'tiny.tif'
    src.write_bytes(b'')  # empty placeholder; constructor doesn't read it
    output_dir = workdir / 'custom_out'
    fi = FileInfo(str(src), output_dir=str(output_dir))
    # Constructor stored the path but did NOT create the directory.
    assert fi.output_dir == str(output_dir)
    assert not output_dir.exists()
    assert not (output_dir / 'nellie_necessities').exists()


def test_prepare_output_dirs_creates_both_dirs(tmp_path) -> None:
    """Explicit ``prepare_output_dirs`` creates both output directories."""
    src = tmp_path / 'tiny.tif'
    src.write_bytes(b'')
    output_dir = tmp_path / 'out'
    fi = FileInfo(str(src), output_dir=str(output_dir))
    fi.prepare_output_dirs()
    assert output_dir.exists()
    assert (output_dir / 'nellie_necessities').exists()


def test_prepare_output_dirs_is_idempotent(tmp_path) -> None:
    """Re-calling ``prepare_output_dirs`` is safe (uses ``exist_ok=True``)."""
    src = tmp_path / 'tiny.tif'
    src.write_bytes(b'')
    fi = FileInfo(str(src), output_dir=str(tmp_path / 'out'))
    fi.prepare_output_dirs()
    fi.prepare_output_dirs()  # would raise without exist_ok=True
    assert (tmp_path / 'out').exists()


def test_find_metadata_creates_output_dirs(tmp_path) -> None:
    """``find_metadata`` calls ``prepare_output_dirs`` before extraction.

    Pins the auto-call wiring so production callers (run.py, napari
    fileselect) don't need to call prepare_output_dirs explicitly.
    """
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path / 'fixture_workdir')
    # _loaded_file_info calls find_metadata; output dirs must exist now.
    assert os.path.isdir(fi.output_dir)
    assert os.path.isdir(fi.nellie_necessities_dir)


# ============================================================
# A. Per-format dim_res extraction (metadata_type branches)
# ============================================================

# ------ metadata_type detection ------

def test_metadata_type_ome_3d(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.metadata_type == "ome"


def test_metadata_type_ome_2d(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_2D_PATH, tmp_path)
    assert fi.metadata_type == "ome"


def test_metadata_type_imagej_with_physicalsize(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH, tmp_path)
    assert fi.metadata_type == "imagej"


def test_metadata_type_imagej_no_physicalsize_falls_back_to_tif_tags(tmp_path) -> None:
    """ImageJ fixture lacking ``physicalsizex`` triggers the tif-tags fallback path."""
    fi = _loaded_file_info(FIXTURE_IMAGEJ_NO_PHYSICALSIZE_PATH, tmp_path)
    assert fi.metadata_type == "imagej_tif_tags"


def test_metadata_type_raw_tiff_is_none(tmp_path) -> None:
    """Raw TIFF (no OME, no ImageJ) → ``metadata_type is None``."""
    fi = _loaded_file_info(FIXTURE_RAW_TIFF_NO_RESUNIT_PATH, tmp_path)
    assert fi.metadata_type is None


# ------ OME branch ------

def test_ome_3d_dim_res_all_four_keys(tmp_path) -> None:
    """3D OME fixture: dim_res has X/Y/Z/T all populated."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res == pytest.approx(
        {"X": 0.0655, "Y": 0.0655, "Z": 0.25, "T": 4.535566806793213}
    )


def test_ome_2d_dim_res_z_is_none(tmp_path) -> None:
    """2D OME fixture has no Z axis; dim_res['Z'] stays None (X/Y/T populated)."""
    fi = _loaded_file_info(FIXTURE_2D_PATH, tmp_path)
    assert fi.dim_res is not None
    assert fi.dim_res["Z"] is None
    assert fi.dim_res["X"] == pytest.approx(0.0655)
    assert fi.dim_res["Y"] == pytest.approx(0.0655)
    assert fi.dim_res["T"] == pytest.approx(4.535566806793213)


# ------ ImageJ branches ------

def test_imagej_with_physicalsize_dim_res(tmp_path) -> None:
    """ImageJ with ``physicalsizex/y``, ``spacing``, ``finterval`` → all four keys filled."""
    fi = _loaded_file_info(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH, tmp_path)
    assert fi.dim_res == pytest.approx(
        {"X": 0.108, "Y": 0.108, "Z": 0.5, "T": 2.0}
    )


def test_imagej_no_physicalsize_falls_back_to_tif_tags_dim_res(tmp_path) -> None:
    """ImageJ without ``physicalsizex`` → fallback fills X/Y from tif tags; Z/T stay None."""
    fi = _loaded_file_info(FIXTURE_IMAGEJ_NO_PHYSICALSIZE_PATH, tmp_path)
    assert fi.dim_res == {"X": 1.0, "Y": 1.0, "Z": None, "T": None}


# ------ Raw TIFF (tif-tags) branch with each ResolutionUnit ------

def test_raw_tiff_no_resunit_no_scaling(tmp_path) -> None:
    """RESUNIT.NONE (or absent) → no scaling: 1/10000 = 0.0001."""
    fi = _loaded_file_info(FIXTURE_RAW_TIFF_NO_RESUNIT_PATH, tmp_path)
    assert fi.dim_res is not None
    assert fi.dim_res["X"] == pytest.approx(0.0001)
    assert fi.dim_res["Y"] == pytest.approx(0.0001)


def test_raw_tiff_centimeter_applies_1e4_scaling(tmp_path) -> None:
    """RESUNIT.CENTIMETER → multiply by 1e4: (1/10000) * 1e4 = 1.0."""
    fi = _loaded_file_info(FIXTURE_RAW_TIFF_CENTIMETER_PATH, tmp_path)
    assert fi.dim_res is not None
    assert fi.dim_res["X"] == pytest.approx(1.0)
    assert fi.dim_res["Y"] == pytest.approx(1.0)


def test_raw_tiff_inch_applies_25400_scaling(tmp_path) -> None:
    """RESUNIT.INCH → multiply by 25400: (1/10000) * 25400 = 2.54."""
    fi = _loaded_file_info(FIXTURE_RAW_TIFF_INCH_PATH, tmp_path)
    assert fi.dim_res is not None
    assert fi.dim_res["X"] == pytest.approx(2.54)
    assert fi.dim_res["Y"] == pytest.approx(2.54)


# ------ Unsupported extension ------

def test_unsupported_extension_raises(tmp_path) -> None:
    """``find_metadata`` on a non-tif/tiff/nd2 extension raises ValueError."""
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny PNG header
    fi = FileInfo(str(bogus))
    with pytest.raises(ValueError, match="File type not supported"):
        fi.find_metadata()


# ============================================================
# B. Axes normalization (_normalize_time_axis)
# ============================================================

def test_normalize_time_axis_imagej_with_physicalsize_stays_yx(tmp_path) -> None:
    """ImageJ-with-physicalsize fixture: shape (32,32) and axes 'YX' — no T prepend."""
    fi = _loaded_file_info(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH, tmp_path)
    assert fi.axes == "YX"


def test_normalize_time_axis_prepends_t_when_leading_singleton(tmp_path) -> None:
    """``_normalize_time_axis``: axes='ZYX' + shape=(1,16,512,512) → axes='TZYX'.

    Driven by direct state manipulation: tifffile won't let us write a
    file whose axes string disagrees with the data shape, so the
    helper is exercised in isolation. The contract under test is the
    pure transform on ``self.axes``/``self.shape``.
    """
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    fi.axes = "ZYX"
    fi.shape = (1, 16, 512, 512)
    fi._normalize_time_axis()
    assert fi.axes == "TZYX"


def test_normalize_time_axis_noop_when_t_already_present(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    fi.axes = "TZYX"
    fi.shape = (2, 17, 192, 279)
    fi._normalize_time_axis()
    assert fi.axes == "TZYX"


def test_normalize_time_axis_noop_when_axes_none(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    fi.axes = None
    fi.shape = (16, 512, 512)
    fi._normalize_time_axis()
    assert fi.axes is None


def test_normalize_time_axis_noop_when_lengths_match(tmp_path) -> None:
    """Lengths match (no leading singleton): leave axes alone."""
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    fi.axes = "ZYX"
    fi.shape = (16, 512, 512)
    fi._normalize_time_axis()
    assert fi.axes == "ZYX"


# ============================================================
# C. Validation (the 7 methods)
# ============================================================

# ---- _axis_errors: 5 distinct error paths ----

def test_axis_errors_when_axes_none(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    # axes/shape are None at construction time
    assert fi._axis_errors() == ["Axes or shape metadata not loaded"]


def test_axis_errors_when_length_mismatch(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "XY"  # length 2 vs shape length 4
    errs = fi._axis_errors()
    assert "Axes length does not match data shape" in errs


def test_axis_errors_when_invalid_letter(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "TZQX"  # Q is not in {T,Z,Y,X,C}
    errs = fi._axis_errors()
    assert "Axes must only use T, Z, C, Y, X" in errs


def test_axis_errors_when_duplicates(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "TTYX"  # two T's
    errs = fi._axis_errors()
    assert "Axes must not contain duplicates" in errs


def test_axis_errors_when_x_or_y_missing(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "TZCX"  # no Y
    errs = fi._axis_errors()
    assert "Axes must include both X and Y" in errs


def test_axis_errors_accepts_c_in_axes(tmp_path) -> None:
    """Pins current intentional behavior: FileInfo accepts multichannel input.

    ``_axis_errors``'s ``allowed_axes`` set includes 'C'
    (verifier.py:373). ``ImInfo._normalize_axes`` excludes C — see
    ``test_verifier_iminfo.py`` and dechaos report Pass 5. The two
    contracts diverge intentionally: FileInfo validates user input
    (pre-channel-collapse); ImInfo loads the canonical post-collapse
    OME-TIFF.
    """
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "TCYX"
    fi.shape = (2, 3, 16, 16)  # matches axes length
    assert fi._axis_errors() == []


# ---- _dim_errors: missing-resolution paths ----

def test_dim_errors_returns_empty_when_axes_none(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    # axes/dim_res both None at construction
    assert fi._dim_errors() == []


def test_dim_errors_returns_empty_when_dim_res_none(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    fi.find_metadata()  # sets axes but not dim_res
    assert fi.dim_res is None
    assert fi._dim_errors() == []


def test_dim_errors_flags_missing_x(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    fi.dim_res["X"] = None
    assert "Missing X resolution" in fi._dim_errors()


def test_dim_errors_flags_missing_y(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    fi.dim_res["Y"] = None
    assert "Missing Y resolution" in fi._dim_errors()


def test_dim_errors_flags_missing_z(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    fi.dim_res["Z"] = None
    assert "Missing Z resolution" in fi._dim_errors()


def test_dim_errors_flags_missing_t(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    fi.dim_res["T"] = None
    assert "Missing T resolution" in fi._dim_errors()


def test_dim_errors_silent_for_axis_not_in_dim_res(tmp_path) -> None:
    """C is not a key in dim_res; presence in axes does not produce an error."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "TCYX"
    fi.shape = (2, 3, 16, 16)
    # X/Y/T present in dim_res, Z is in axes → only error is missing Z
    # (C is silently skipped because it's not a dim_res key)
    errs = fi._dim_errors()
    assert "Missing C resolution" not in errs


# ---- _time_range_errors: 4 raise paths + silent no-ops ----

def test_time_range_errors_silent_when_axes_none(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    assert fi._time_range_errors() == []


def test_time_range_errors_silent_when_no_t_axis(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH, tmp_path)
    # axes='YX' has no T
    assert fi._time_range_errors() == []


def test_time_range_errors_silent_when_t_start_none(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    # FileInfo.__init__ sets t_start = 0 (typed int), but _time_range_errors
    # has a guard for the None case. The test pins that silent-no-op contract;
    # narrowing-via-assert isn't possible because the value is intentionally
    # None at the assertion point. Suppress the single-line assignment error.
    fi.t_start = None  # type: ignore[assignment]
    assert fi._time_range_errors() == []


def test_time_range_errors_negative_start_or_end(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.t_start = -1
    assert "Temporal range must be >= 0" in fi._time_range_errors()


def test_time_range_errors_start_greater_than_end(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.t_start = 1
    fi.t_end = 0
    assert "Start frame must be <= end frame" in fi._time_range_errors()


def test_time_range_errors_out_of_bounds(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    # 3D fixture has T=2 → max_t = 1; t_end=99 is out of bounds
    fi.t_end = 99
    assert "Temporal range out of bounds" in fi._time_range_errors()


# ---- _validate: asymmetric raise (THE design quirk) ----

def test_validate_implicit_after_load_metadata_3d(tmp_path) -> None:
    """``load_metadata`` runs ``_validate`` implicitly; on a clean fixture, no errors."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.validation_errors == []
    assert fi.good_axes is True
    assert fi.good_dims is True


def test_validate_raises_only_on_time_errors(tmp_path) -> None:
    """``_validate`` raises ValueError only when ``_time_range_errors`` is non-empty."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.t_end = 99  # past max_t
    with pytest.raises(ValueError, match="Temporal range out of bounds"):
        fi._validate()


def test_validate_does_not_raise_on_axis_or_dim_errors(tmp_path) -> None:
    """Pins current intentional behavior: ``_validate`` raises only on time errors.

    Axis and dim errors are flagged via ``good_axes``/``good_dims``
    booleans for the napari widget to display, not raised as
    exceptions. Time errors are raised because they represent
    programmer-error in the temporal slicing pipeline (the napari
    widget already validates the range before calling). See dechaos
    report Pass 5 and Slice 5.
    """
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "XY"  # length mismatch, axis-error path
    # Should NOT raise even though _check_axes will flag the error
    fi._validate()
    assert fi.good_axes is False
    assert "Axes length does not match data shape" in fi.validation_errors


# ---- get_validation_errors: concatenated report, no mutation ----

def test_get_validation_errors_returns_concatenated(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    fi.axes = "TZQX"  # invalid letter
    fi.dim_res["X"] = None
    fi.t_end = 99
    expected = fi._axis_errors() + fi._dim_errors() + fi._time_range_errors()
    assert fi.get_validation_errors() == expected


def test_get_validation_errors_does_not_mutate_flags(tmp_path) -> None:
    """Unlike ``_check_axes``/``_check_dim_res``, ``get_validation_errors`` is read-only."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    assert fi.dim_res is not None
    # Force good flags True, then introduce a hidden error
    fi.good_axes = True
    fi.good_dims = True
    fi.dim_res["X"] = None
    _ = fi.get_validation_errors()
    # Flags should be unchanged because get_validation_errors does not mutate
    assert fi.good_axes is True
    assert fi.good_dims is True


# ============================================================
# D. Mutators (5 methods)
# ============================================================

# ---- change_axes: half-commented gate ----

def test_change_axes_to_valid_string(tmp_path) -> None:
    """``change_axes('TZYX')`` on a (2,17,192,279) fixture leaves ``good_axes=True``."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.change_axes("TZYX")
    assert fi.axes == "TZYX"
    assert fi.good_axes is True


def test_change_axes_bad_length_raises(tmp_path) -> None:
    """``change_axes`` raises ``ValueError`` on length mismatch.

    Slice 2 restored the length gate (originally added in commit
    bb2b0b7, disabled by commit 492edfb in Aug 2024). The napari
    fileselect widget at nellie_fileselect.py:868-879 already
    pre-validates length before calling change_axes (red error +
    short-circuit), so the verifier-level gate is redundant for
    napari but defensive for programmatic callers (run.py, tests,
    scripted use). On length mismatch, ``self.axes`` is NOT mutated.
    """
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    original_axes = fi.axes
    with pytest.raises(
        ValueError, match="New axes must have the same length as the shape of the data"
    ):
        fi.change_axes("XY")
    # axes is preserved on failure; not mutated to bad value
    assert fi.axes == original_axes


def test_change_axes_missing_x_or_y_flags_invalid(tmp_path) -> None:
    """``change_axes('TZYC')`` on (2,17,192,279): length matches but missing X."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.change_axes("TZYC")
    assert fi.axes == "TZYC"
    assert fi.good_axes is False
    assert "Axes must include both X and Y" in fi.validation_errors


# ---- change_dim_res ----

def test_change_dim_res_updates_value(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.change_dim_res("X", 0.5)
    assert fi.dim_res is not None
    assert fi.dim_res["X"] == 0.5


def test_change_dim_res_invalid_dim_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(ValueError, match="Invalid dimension 'foo'"):
        fi.change_dim_res("foo", 0.5)


def test_change_dim_res_before_load_metadata_raises(tmp_path) -> None:
    """If ``dim_res`` is None (not yet loaded), ``change_dim_res`` raises."""
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    assert fi.dim_res is None
    with pytest.raises(ValueError, match="Dimension resolutions are not initialized"):
        fi.change_dim_res("X", 0.5)


# ---- change_selected_channel ----

def test_change_selected_channel_requires_good_dims(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.good_dims = False
    with pytest.raises(ValueError, match="valid axes and dimensions"):
        fi.change_selected_channel(0)


def test_change_selected_channel_requires_good_axes(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.good_axes = False
    with pytest.raises(ValueError, match="valid axes and dimensions"):
        fi.change_selected_channel(0)


def test_change_selected_channel_requires_c_in_axes(tmp_path) -> None:
    """3D OME fixture has axes='TZYX' — no C, raises KeyError."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(KeyError, match="No channel dimension"):
        fi.change_selected_channel(0)


def _make_synthetic_tczyx_tiff(path, t=1, c=3, y=16, x=16) -> None:
    """Write a small TCYX TIFF (raw, not OME) with axes='TCYX'."""
    arr = np.zeros((t, c, y, x), dtype=np.uint16)
    for ch in range(c):
        arr[:, ch, :, :] = ch + 1
    tifffile.imwrite(
        str(path), arr, metadata={"axes": "TCYX"}, photometric="minisblack"
    )


def _loaded_synthetic_tcyx_file_info(tmp_path) -> FileInfo:
    """Build a FileInfo loaded from a raw synthetic TCYX TIFF, with dim_res patched."""
    p = tmp_path / "multich.tif"
    _make_synthetic_tczyx_tiff(p)
    fi = FileInfo(str(p))
    fi.find_metadata()
    fi.load_metadata()
    # Raw TIFF tags don't have T resolution → good_dims=False; patch so
    # change_selected_channel's gate passes.
    fi.change_dim_res("T", 1.0)
    return fi


def test_change_selected_channel_invalid_index_raises(tmp_path) -> None:
    fi = _loaded_synthetic_tcyx_file_info(tmp_path)
    with pytest.raises(IndexError, match="Invalid channel index"):
        fi.change_selected_channel(99)


def test_change_selected_channel_negative_index_raises(tmp_path) -> None:
    fi = _loaded_synthetic_tcyx_file_info(tmp_path)
    with pytest.raises(IndexError, match="Invalid channel index"):
        fi.change_selected_channel(-1)


def test_change_selected_channel_valid_updates_ch(tmp_path) -> None:
    fi = _loaded_synthetic_tcyx_file_info(tmp_path)
    fi.change_selected_channel(2)
    assert fi.ch == 2


# ---- select_temporal_range: 7 validation paths ----

def test_select_temporal_range_axes_none_raises(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst))
    with pytest.raises(ValueError, match="Axes or shape metadata not loaded"):
        fi.select_temporal_range(0, 1)


def test_select_temporal_range_axes_shape_mismatch_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.axes = "XY"  # length 2 vs shape length 4
    with pytest.raises(ValueError, match="Axes and shape length mismatch"):
        fi.select_temporal_range(0, 1)


def test_select_temporal_range_no_t_axis_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE_PATH, tmp_path)
    # axes='YX'
    with pytest.raises(KeyError, match="No time dimension"):
        fi.select_temporal_range(0, 1)


def test_select_temporal_range_negative_start_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(IndexError, match="Start frame must be >= 0"):
        fi.select_temporal_range(-1, 1)


def test_select_temporal_range_negative_end_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(IndexError, match="End frame must be >= 0"):
        fi.select_temporal_range(0, -1)


def test_select_temporal_range_start_after_end_raises(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(ValueError, match="Start frame must be <= end frame"):
        fi.select_temporal_range(1, 0)


def test_select_temporal_range_out_of_bounds_raises(tmp_path) -> None:
    """3D fixture has T=2 → max_t=1; end=2 raises."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    with pytest.raises(IndexError, match="Temporal range out of bounds"):
        fi.select_temporal_range(0, 2)


def test_select_temporal_range_happy_path(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.select_temporal_range(0, 1)
    assert (fi.t_start, fi.t_end) == (0, 1)


# ============================================================
# E. save_ome_tiff
# ============================================================

def test_save_ome_tiff_requires_good_axes_and_dims(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.good_axes = False
    with pytest.raises(ValueError, match="Cannot save file with invalid axes or dimensions"):
        fi.save_ome_tiff()


def test_save_ome_tiff_creates_output_file(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    assert os.path.exists(fi.ome_output_path)


def test_save_ome_tiff_collapses_channel(tmp_path) -> None:
    """Synthetic TCYX (T=2,C=3) with ch=1 → C dropped from output axes; data is C=1 plane.

    Uses T=2 (not T=1) so the synthetic TIFF doesn't have a singleton T
    that some tifffile versions strip on readback.

    The provenance JSON's ``output_axes`` records the post-collapse
    axes string ('TYX'). tifffile's ``series[0].axes`` may strip
    singleton T/C from the physical shape on readback, so the
    provenance JSON is the canonical source of truth for the saved
    axes contract.
    """
    p = tmp_path / "multich.tif"
    _make_synthetic_tczyx_tiff(p, t=2, c=3, y=16, x=16)
    fi = FileInfo(str(p))
    fi.find_metadata()
    fi.load_metadata()
    fi.change_dim_res("T", 1.0)  # raw TIFF lacks T resolution
    fi.change_selected_channel(1)
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    comment = tifffile.tiffcomment(fi.ome_output_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    description = ome.images[0].description
    assert description is not None
    prov = json.loads(description)
    out_data = tifffile.imread(fi.ome_output_path)
    assert prov["output_axes"] == "TYX"
    # OME pixel metadata reflects channel collapse: SizeC == 1
    assert ome.images[0].pixels.size_c == 1
    # Second channel is filled with value 2 (ch index 1 → ch+1)
    assert int(out_data.max()) == 2
    assert int(out_data.min()) == 2


def test_save_ome_tiff_t_prepend_for_zyx(tmp_path) -> None:
    """ZYX source (no T) is prepended with a T axis on save.

    Verified via the provenance JSON's ``output_axes`` (canonical
    contract). tifffile may report a different ``series[0].axes``
    string after stripping singletons; the OME description holds the
    intent.
    """
    p = tmp_path / "zyx.tif"
    arr = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    tifffile.imwrite(str(p), arr, metadata={"axes": "ZYX"}, photometric="minisblack")
    fi = FileInfo(str(p))
    fi.find_metadata()
    fi.load_metadata()
    # Raw TIFF: only X/Y picked up (1.0/1.0); patch Z so good_dims=True.
    fi.change_dim_res("Z", 0.25)
    assert fi.good_dims is True
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    comment = tifffile.tiffcomment(fi.ome_output_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    description = ome.images[0].description
    assert description is not None
    prov = json.loads(description)
    assert prov["output_axes"].startswith("T")
    # OME pixel metadata reflects the prepended T: SizeT >= 1
    assert ome.images[0].pixels.size_t >= 1


def test_save_ome_tiff_slices_t(tmp_path) -> None:
    """select_temporal_range(0, 0) on T=2 fixture → output has T=1.

    Verified via OME pixel metadata's ``size_t`` (canonical contract).
    tifffile may strip the singleton T from ``series[0].axes`` on
    readback; OME's ``SizeT`` is the source of truth.
    """
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.select_temporal_range(0, 0)
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    comment = tifffile.tiffcomment(fi.ome_output_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    assert ome.images[0].pixels.size_t == 1


def test_save_ome_tiff_provenance_json_keys(tmp_path) -> None:
    """Provenance JSON in OME image description has the 6 documented keys."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    comment = tifffile.tiffcomment(fi.ome_output_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    description = ome.images[0].description
    assert description is not None
    prov = json.loads(description)
    assert set(prov.keys()) == {
        "source_axes",
        "output_axes",
        "dim_res",
        "channel",
        "t_start",
        "t_end",
    }


def test_save_ome_tiff_dim_res_round_trip(tmp_path) -> None:
    """physical_size_x/y/z + time_increment in the saved OME-TIFF match the input dim_res."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.save_ome_tiff()
    assert fi.ome_output_path is not None
    assert fi.dim_res is not None
    comment = tifffile.tiffcomment(fi.ome_output_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    pixels = ome.images[0].pixels
    assert pixels.physical_size_x == pytest.approx(fi.dim_res["X"])
    assert pixels.physical_size_y == pytest.approx(fi.dim_res["Y"])
    assert pixels.physical_size_z == pytest.approx(fi.dim_res["Z"])
    assert pixels.time_increment == pytest.approx(fi.dim_res["T"])


# ============================================================
# F. read_file 3-fallback chain
# ============================================================

def test_read_file_returns_memmap_for_tiff(tmp_path) -> None:
    """``read_file`` on a memmap-friendly TIFF returns a memmap."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    data = fi.read_file()
    # tifffile.memmap returns either np.memmap or has a .filename attribute
    assert isinstance(data, np.memmap) or hasattr(data, "filename")


def test_read_file_sets_dtype(tmp_path) -> None:
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    fi.read_file()
    assert fi.dtype is not None


def test_read_file_falls_back_to_imread_for_compressed(tmp_path) -> None:
    """Compressed TIFFs are not memmap-able; ``read_file`` should fall back to ``imread``."""
    p = tmp_path / "compressed.tif"
    arr = np.random.randint(0, 100, (16, 16), dtype=np.uint16)
    tifffile.imwrite(str(p), arr, compression="zlib")
    fi = FileInfo(str(p))
    fi.find_metadata()
    fi.load_metadata()
    data = fi.read_file()
    # Fallback returns a true ndarray (not a memmap)
    assert not isinstance(data, np.memmap)
    assert data.shape == (16, 16)


def test_read_file_unsupported_extension_raises(tmp_path) -> None:
    """``read_file`` on a non-tif/tiff/nd2 extension raises ValueError."""
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"\x89PNG\r\n\x1a\n")
    fi = FileInfo(str(bogus))
    # Skip find_metadata (would also raise); set extension and call read_file directly
    with pytest.raises(ValueError, match="not supported"):
        fi.read_file()


# ============================================================
# G. _get_output_path (naming strategies)
# ============================================================

def test_get_output_path_detailed_includes_axes_and_dim_res(tmp_path) -> None:
    """Detailed naming: filename contains axes letters with rounded dim_res values."""
    fi = _loaded_file_info(FIXTURE_3D_PATH, tmp_path)
    # axes='TZYX'; dim_res has X=0.0655, Y=0.0655, Z=0.25, T=4.535566806793213
    assert fi.ome_output_path is not None
    name = os.path.basename(fi.ome_output_path)
    assert "T4p5356" in name  # T rounded to 4 decimals, '.'→'p'
    assert "X0p0655" in name


def test_get_output_path_stable_drops_axes_and_dim_res(tmp_path) -> None:
    """Stable naming: filename equals ``filename_no_ext`` (no axes/dim_res/ch/t suffix)."""
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst), output_naming="stable")
    fi.find_metadata()
    fi.load_metadata()
    assert fi.ome_output_path is not None
    name = os.path.basename(fi.ome_output_path)
    # stable form: filename_no_ext + '.ome.tif'
    expected = fi.filename_no_ext + ".ome.tif"
    assert name == expected


def test_get_output_path_unsupported_naming_raises(tmp_path) -> None:
    dst = _copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path)
    fi = FileInfo(str(dst), output_naming="bogus")
    fi.find_metadata()
    # find_metadata sets axes/shape; load_metadata calls _validate → _get_output_path
    with pytest.raises(ValueError, match="Unsupported output naming"):
        fi.load_metadata()
