"""Regenerate the test fixtures and the 2D demo from the 3D source.

Outputs:
- ``sample_data/yeast_2d_mitochondria.ome.tif`` — Z-max projection of the
  3D source, all timepoints. Parallel demo dataset for the 2D code path.
- ``tests/fixtures/yeast_3d_t0_to_1.ome.tif`` — first 2 timepoints of the
  3D source. Drives 3D characterization tests.
- ``tests/fixtures/yeast_2d_t0_to_1.ome.tif`` — first 2 timepoints of the
  Z-max projection. Drives 2D characterization tests.

Run from the repo root:

    python tests/fixtures/_generate.py

The outputs are committed; this script only needs to be re-run if the
3D source changes or the desired subsample changes.
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

NUM_TIMEPOINTS = 2

# Physical scale carried over from the source so ImInfo's verifier picks up
# the same dim_res. Keep these in sync if the source file ever changes.
PHYSICAL_SIZE_X = 0.0655
PHYSICAL_SIZE_Y = 0.0655
PHYSICAL_SIZE_Z = 0.25
PHYSICAL_SIZE_UNIT = "µm"
TIME_INCREMENT = 4.535566806793213
TIME_INCREMENT_UNIT = "s"


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

    print("Done.")


if __name__ == "__main__":
    main()
