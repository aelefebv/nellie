"""One-shot capture of `Filter.run()` output SHA-256 on the standard fixtures.

Produces the hardcoded SHAs used in
``test_run_filter_3d_snapshot_pre_rewrite`` and
``test_run_filter_2d_snapshot_pre_rewrite`` (in ``tests/test_filtering.py``).

PRD #233 Slice 1 captures these SHAs against the current implementation;
Slice 2 must preserve them bit-identically for finite-only fixture data.
Rerun if upstream Frangi math intentionally changes (rare — `frangi_math`
hasn't moved since the closed-form 3D eigvalsh swap).

Usage::

    python tests/_capture_filter_run_sha.py
"""

from __future__ import annotations

import gc
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

from nellie.im_info import load_image
from nellie.segmentation.filtering import Filter, FrangiConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_3D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_3d_t0_to_1.ome.tif"
FIXTURE_2D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_2d_t0_to_1.ome.tif"

CFG = FrangiConfig(device="cpu")


def _sha_of_run(fixture_path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="capture_filter_sha_") as tmp:
        workdir = Path(tmp)
        local = workdir / fixture_path.name
        shutil.copy(fixture_path, local)
        info = load_image(local)
        filt = Filter(info, CFG, num_t=2)
        filt.run()
        out = np.array(filt.frangi_memmap)
        sha = hashlib.sha256(out.tobytes()).hexdigest()
        filt.frangi_memmap = None
        filt.im_memmap = None
        gc.collect()
    return sha


def main() -> None:
    sha_3d = _sha_of_run(FIXTURE_3D_PATH)
    sha_2d = _sha_of_run(FIXTURE_2D_PATH)
    print(f"3D SHA: {sha_3d}")
    print(f"2D SHA: {sha_2d}")


if __name__ == "__main__":
    main()
