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
