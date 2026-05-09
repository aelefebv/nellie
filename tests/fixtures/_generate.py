"""Regenerate the test fixtures and the 2D demo from the 3D source.

Outputs:
- ``sample_data/yeast_2d_mitochondria.ome.tif`` — Z-max projection of the
  3D source, all timepoints. Parallel demo dataset for the 2D code path.
- ``tests/fixtures/yeast_3d_t0_to_1.ome.tif`` — first 2 timepoints of the
  3D source. Drives 3D characterization tests.
- ``tests/fixtures/yeast_2d_t0_to_1.ome.tif`` — first 2 timepoints of the
  Z-max projection. Drives 2D characterization tests.
- ``tests/fixtures/imagej_with_physicalsize.tif`` — ImageJ TIFF with
  ``physicalsizex`` set in imagej_metadata. Drives FileInfo's
  ``'imagej'`` metadata branch.
- ``tests/fixtures/imagej_no_physicalsize.tif`` — ImageJ TIFF without
  ``physicalsizex``; drives the ``'imagej_tif_tags'`` fallback branch
  that combines ImageJ metadata with raw TIFF tags.
- ``tests/fixtures/raw_tiff_no_resunit.tif`` — Plain TIFF with
  XResolution/YResolution tags but no ResolutionUnit. Drives the
  ``None`` branch with no scaling (assumes microns).
- ``tests/fixtures/raw_tiff_centimeter.tif`` — Plain TIFF with
  ResolutionUnit=CENTIMETER; drives the ×1e4 scaling path.
- ``tests/fixtures/raw_tiff_inch.tif`` — Plain TIFF with
  ResolutionUnit=INCH; drives the ×25400 scaling path.

Run from the repo root:

    python tests/fixtures/_generate.py

The outputs are committed; this script only needs to be re-run if the
3D source changes, the desired subsample changes, or the verifier
characterization tests need updated fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "sample_data" / "yeast_3d_mitochondria.ome.tif"
DEMO_2D = REPO_ROOT / "sample_data" / "yeast_2d_mitochondria.ome.tif"
FIXTURE_3D = REPO_ROOT / "tests" / "fixtures" / "yeast_3d_t0_to_1.ome.tif"
FIXTURE_2D = REPO_ROOT / "tests" / "fixtures" / "yeast_2d_t0_to_1.ome.tif"

# Verifier characterization fixtures (one per FileInfo metadata_type branch).
FIXTURE_IMAGEJ_WITH_PHYSICALSIZE = REPO_ROOT / "tests" / "fixtures" / "imagej_with_physicalsize.tif"
FIXTURE_IMAGEJ_NO_PHYSICALSIZE = REPO_ROOT / "tests" / "fixtures" / "imagej_no_physicalsize.tif"
FIXTURE_RAW_TIFF_NO_RESUNIT = REPO_ROOT / "tests" / "fixtures" / "raw_tiff_no_resunit.tif"
FIXTURE_RAW_TIFF_CENTIMETER = REPO_ROOT / "tests" / "fixtures" / "raw_tiff_centimeter.tif"
FIXTURE_RAW_TIFF_INCH = REPO_ROOT / "tests" / "fixtures" / "raw_tiff_inch.tif"

NUM_TIMEPOINTS = 2

# Physical scale carried over from the source so ImInfo's verifier picks up
# the same dim_res. Keep these in sync if the source file ever changes.
PHYSICAL_SIZE_X = 0.0655
PHYSICAL_SIZE_Y = 0.0655
PHYSICAL_SIZE_Z = 0.25
PHYSICAL_SIZE_UNIT = "µm"
TIME_INCREMENT = 4.535566806793213
TIME_INCREMENT_UNIT = "s"

# Synthetic verifier-fixture parameters. Small Y×X so each fixture is a
# few KB. Hardcoded so test assertions can compare against literal values.
VERIFIER_FIXTURE_SHAPE_2D = (32, 32)  # (Y, X)
VERIFIER_FIXTURE_DTYPE = np.uint16
VERIFIER_IMAGEJ_PHYSICAL_X = 0.108
VERIFIER_IMAGEJ_PHYSICAL_Y = 0.108
VERIFIER_IMAGEJ_PHYSICAL_Z = 0.5
VERIFIER_IMAGEJ_FRAME_INTERVAL = 2.0
# Raw TIFF resolution: tifffile encodes as (numerator, denominator) and
# verifier reads ``denominator / numerator`` (line 258-259). For
# ``XResolution=(10000, 1)`` (10000 px per unit) verifier returns 1/10000.
# CENTIMETER then multiplies by 1e4 to convert to micrometers; INCH by
# 25400. NO_RESUNIT case leaves it as-is (assumed micrometers already).
VERIFIER_RAW_RESOLUTION_NUMERATOR = 10000
VERIFIER_RAW_RESOLUTION_DENOMINATOR = 1


def _read_source() -> np.ndarray:
    with tifffile.TiffFile(SOURCE) as tf:
        data = tf.asarray()
        axes = tf.series[0].axes
    if axes != "TZYX":
        raise RuntimeError(
            f"Expected source axes TZYX, got {axes!r}. The generator "
            "assumes the canonical Bio-Formats layout."
        )
    return data


def _write_3d(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack",
        metadata={
            "axes": "TZYX",
            "PhysicalSizeX": PHYSICAL_SIZE_X,
            "PhysicalSizeXUnit": PHYSICAL_SIZE_UNIT,
            "PhysicalSizeY": PHYSICAL_SIZE_Y,
            "PhysicalSizeYUnit": PHYSICAL_SIZE_UNIT,
            "PhysicalSizeZ": PHYSICAL_SIZE_Z,
            "PhysicalSizeZUnit": PHYSICAL_SIZE_UNIT,
            "TimeIncrement": TIME_INCREMENT,
            "TimeIncrementUnit": TIME_INCREMENT_UNIT,
        },
    )


def _write_2d(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack",
        metadata={
            "axes": "TYX",
            "PhysicalSizeX": PHYSICAL_SIZE_X,
            "PhysicalSizeXUnit": PHYSICAL_SIZE_UNIT,
            "PhysicalSizeY": PHYSICAL_SIZE_Y,
            "PhysicalSizeYUnit": PHYSICAL_SIZE_UNIT,
            "TimeIncrement": TIME_INCREMENT,
            "TimeIncrementUnit": TIME_INCREMENT_UNIT,
        },
    )


def _make_verifier_2d_data() -> np.ndarray:
    """Deterministic small 2D image for verifier characterization fixtures."""
    rng = np.random.default_rng(seed=20260509)
    return rng.integers(
        0, np.iinfo(VERIFIER_FIXTURE_DTYPE).max,
        size=VERIFIER_FIXTURE_SHAPE_2D, dtype=VERIFIER_FIXTURE_DTYPE,
    )


def _write_imagej_with_physicalsize(path: Path, data: np.ndarray) -> None:
    """ImageJ TIFF with ``physicalsizex`` set in imagej_metadata.

    Drives FileInfo's ``'imagej'`` metadata branch
    (verifier.py:161-163, 218-230).
    """
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        metadata={
            "physicalsizex": VERIFIER_IMAGEJ_PHYSICAL_X,
            "physicalsizey": VERIFIER_IMAGEJ_PHYSICAL_Y,
            "spacing": VERIFIER_IMAGEJ_PHYSICAL_Z,
            "finterval": VERIFIER_IMAGEJ_FRAME_INTERVAL,
            "unit": "um",
        },
    )


def _write_imagej_no_physicalsize(path: Path, data: np.ndarray) -> None:
    """ImageJ TIFF without ``physicalsizex``; falls back to TIFF tags.

    Drives the ``'imagej_tif_tags'`` branch (verifier.py:161-166)
    that combines ImageJ metadata with raw TIFF tags. Resolution tags
    are written so ``_get_tif_tags_metadata`` has data to extract.
    """
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        resolution=(
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
        ),
        resolutionunit=tifffile.RESUNIT.CENTIMETER,
        metadata={"unit": "um"},
    )


def _write_raw_tiff_no_resunit(path: Path, data: np.ndarray) -> None:
    """Plain TIFF with XResolution/YResolution but no ResolutionUnit.

    Drives the ``None`` metadata_type branch (verifier.py:167-169, 346)
    with the no-scaling path (verifier.py:263-269 — neither CENTIMETER
    nor INCH branch fires; X/Y assumed to already be in micrometers).
    """
    tifffile.imwrite(
        path,
        data,
        resolution=(
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
        ),
        resolutionunit=tifffile.RESUNIT.NONE,
    )


def _write_raw_tiff_centimeter(path: Path, data: np.ndarray) -> None:
    """Plain TIFF with ResolutionUnit=CENTIMETER.

    Drives the ``None`` metadata_type branch with the ×1e4 scaling path
    (verifier.py:264-266).
    """
    tifffile.imwrite(
        path,
        data,
        resolution=(
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
        ),
        resolutionunit=tifffile.RESUNIT.CENTIMETER,
    )


def _write_raw_tiff_inch(path: Path, data: np.ndarray) -> None:
    """Plain TIFF with ResolutionUnit=INCH.

    Drives the ``None`` metadata_type branch with the ×25400 scaling
    path (verifier.py:267-269).
    """
    tifffile.imwrite(
        path,
        data,
        resolution=(
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
            (VERIFIER_RAW_RESOLUTION_NUMERATOR, VERIFIER_RAW_RESOLUTION_DENOMINATOR),
        ),
        resolutionunit=tifffile.RESUNIT.INCH,
    )


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Source file missing: {SOURCE}")

    data = _read_source()  # (T, Z, Y, X)
    print(f"Source: {data.shape} {data.dtype}")

    # 2D demo: Z-max projection across all timepoints
    data_2d_full = data.max(axis=1)  # (T, Y, X)
    print(f"Writing {DEMO_2D.relative_to(REPO_ROOT)}: {data_2d_full.shape} {data_2d_full.dtype}")
    _write_2d(DEMO_2D, data_2d_full)

    # 3D fixture: first NUM_TIMEPOINTS timepoints
    data_3d_sub = data[:NUM_TIMEPOINTS]
    print(f"Writing {FIXTURE_3D.relative_to(REPO_ROOT)}: {data_3d_sub.shape} {data_3d_sub.dtype}")
    _write_3d(FIXTURE_3D, data_3d_sub)

    # 2D fixture: first NUM_TIMEPOINTS timepoints of the Z-max projection
    data_2d_sub = data_2d_full[:NUM_TIMEPOINTS]
    print(f"Writing {FIXTURE_2D.relative_to(REPO_ROOT)}: {data_2d_sub.shape} {data_2d_sub.dtype}")
    _write_2d(FIXTURE_2D, data_2d_sub)

    # Verifier characterization fixtures (one per FileInfo metadata_type branch).
    verifier_data = _make_verifier_2d_data()
    print(
        f"Writing {FIXTURE_IMAGEJ_WITH_PHYSICALSIZE.relative_to(REPO_ROOT)}: "
        f"{verifier_data.shape} {verifier_data.dtype}"
    )
    _write_imagej_with_physicalsize(FIXTURE_IMAGEJ_WITH_PHYSICALSIZE, verifier_data)

    print(
        f"Writing {FIXTURE_IMAGEJ_NO_PHYSICALSIZE.relative_to(REPO_ROOT)}: "
        f"{verifier_data.shape} {verifier_data.dtype}"
    )
    _write_imagej_no_physicalsize(FIXTURE_IMAGEJ_NO_PHYSICALSIZE, verifier_data)

    print(
        f"Writing {FIXTURE_RAW_TIFF_NO_RESUNIT.relative_to(REPO_ROOT)}: "
        f"{verifier_data.shape} {verifier_data.dtype}"
    )
    _write_raw_tiff_no_resunit(FIXTURE_RAW_TIFF_NO_RESUNIT, verifier_data)

    print(
        f"Writing {FIXTURE_RAW_TIFF_CENTIMETER.relative_to(REPO_ROOT)}: "
        f"{verifier_data.shape} {verifier_data.dtype}"
    )
    _write_raw_tiff_centimeter(FIXTURE_RAW_TIFF_CENTIMETER, verifier_data)

    print(
        f"Writing {FIXTURE_RAW_TIFF_INCH.relative_to(REPO_ROOT)}: "
        f"{verifier_data.shape} {verifier_data.dtype}"
    )
    _write_raw_tiff_inch(FIXTURE_RAW_TIFF_INCH, verifier_data)

    print("Done.")


if __name__ == "__main__":
    main()
