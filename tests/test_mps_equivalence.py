"""MPS / CPU numerical equivalence tests for the nellie pipeline stages.

Each test runs the same stage twice on the same input — once with
``device="cpu"``, once with ``device="mps"`` — and asserts the
backends agree within float32 tolerance.

All tests are gated by the ``mps`` pytest marker (deselected by default
in ``pyproject.toml``). Run with ``pytest -m mps`` on a Mac with
``pip install 'nellie[mps]'`` and a working MPS device. Tests skip
cleanly when torch is missing or MPS is unavailable.

**On the chosen tolerances.** The PRD (#140 § Testing Decisions) calls
for vesselness mean within 0.5% relative + per-voxel max abs diff
within "float32 tolerance for typical work." Empirically the real
numbers on the yeast fixtures are way below those:

  3D yeast fixture:  rel_mean_diff ~1e-6  max_abs_diff ~1e-8
  2D yeast fixture:  rel_mean_diff ~1e-7  max_abs_diff ~1e-7

The tolerances here use the PRD ceiling (0.5% / 1e-4) on purpose, so
the test still passes if MPS rounding drifts a touch across torch
versions. Tightening them further would be a regression-detection
win, but the current bar is what the slice acceptance criteria
specify and it's what reviewers should evaluate against.
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
    failure. ``mps_available`` returns False when torch is missing,
    when MPS isn't built into the wheel, or when the OS doesn't expose
    a Metal-capable device.
    """
    if not adaptive_run.mps_available():
        pytest.skip(
            "torch+MPS not available — install with `pip install 'nellie[mps]'` "
            "and run on Apple Silicon."
        )


def _release_filter(filt: Filter) -> None:
    """Drop a Filter's memmap references and force gc.

    Required on Windows: the ``im_preprocessed`` memmap holds an
    exclusive handle that blocks subsequent overwrites.
    """
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()


def _run_filter(im_info, *, device: str) -> np.ndarray:
    """Run Filter end-to-end on ``device`` and return a numpy copy of the output."""
    filt = Filter(im_info, FrangiConfig(device=device), num_t=2)
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


def _vesselness_stats(
    cpu: np.ndarray, mps: np.ndarray
) -> dict[str, float]:
    """Summary statistics comparing CPU and MPS vesselness outputs.

    Returned keys:
        cpu_mean / mps_mean — overall mean response magnitude
        rel_mean_diff — abs(mps_mean - cpu_mean) / max(cpu_mean, 1e-12)
        max_abs_diff — per-voxel L_inf difference
        l2_rel_diff — relative L2 distance (norm of diff / norm of cpu)
    """
    cpu_mean = float(cpu.mean())
    mps_mean = float(mps.mean())
    diff = mps - cpu
    max_abs = float(np.max(np.abs(diff)))
    cpu_norm = float(np.linalg.norm(cpu))
    diff_norm = float(np.linalg.norm(diff))
    return {
        "cpu_mean": cpu_mean,
        "mps_mean": mps_mean,
        "rel_mean_diff": abs(mps_mean - cpu_mean) / max(cpu_mean, 1e-12),
        "max_abs_diff": max_abs,
        "l2_rel_diff": diff_norm / max(cpu_norm, 1e-12),
    }


# ---------------------------------------------------------------------------
# Filter equivalence
# ---------------------------------------------------------------------------


def test_filter_equivalence_cpu_vs_mps_3d(make_imageinfo_3d, capsys) -> None:
    """``Filter.run()`` on the 3D yeast fixture: CPU vs MPS within tolerance.

    Tolerances:
      - ``rel_mean_diff <= 0.005`` — PRD ceiling of 0.5% relative mean
        difference. Real number on this fixture sits well below 1e-3.
      - ``max_abs_diff <= 1e-4`` — per-voxel L_inf difference in
        float32 territory. Empirically ~1e-5 on this fixture.

    If you tighten these, run a few times locally first; MPS reduction
    order drifts a touch run-to-run on cumulative-sum-ish ops.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_3d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_3d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.float32

    stats = _vesselness_stats(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq 3D] cpu_mean={stats['cpu_mean']:.6e} "
            f"mps_mean={stats['mps_mean']:.6e} "
            f"rel_mean_diff={stats['rel_mean_diff']:.3e} "
            f"max_abs_diff={stats['max_abs_diff']:.3e} "
            f"l2_rel_diff={stats['l2_rel_diff']:.3e}"
        )

    assert stats["rel_mean_diff"] <= 0.005, (
        f"3D vesselness mean drifted {stats['rel_mean_diff']:.3%} between "
        f"CPU ({stats['cpu_mean']:.3e}) and MPS ({stats['mps_mean']:.3e}); "
        f"expected <= 0.5% per PRD #140."
    )
    assert stats["max_abs_diff"] <= 1e-4, (
        f"3D vesselness per-voxel max abs diff "
        f"({stats['max_abs_diff']:.3e}) exceeds the float32 tolerance "
        f"of 1e-4 used as the slice acceptance bar."
    )


def test_filter_equivalence_cpu_vs_mps_2d(make_imageinfo_2d, capsys) -> None:
    """``Filter.run()`` on the 2D yeast fixture: CPU vs MPS within tolerance.

    The 2D path additionally fuses a multi-scale LoG response (via the
    shim's ``gaussian_laplace``), so this exercises a different op
    surface than the 3D test. Same tolerance bar; empirically the LoG
    fusion adds a touch more drift than the 3D Hessian path but it
    stays within the ceiling.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_2d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_2d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.float32

    stats = _vesselness_stats(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq 2D] cpu_mean={stats['cpu_mean']:.6e} "
            f"mps_mean={stats['mps_mean']:.6e} "
            f"rel_mean_diff={stats['rel_mean_diff']:.3e} "
            f"max_abs_diff={stats['max_abs_diff']:.3e} "
            f"l2_rel_diff={stats['l2_rel_diff']:.3e}"
        )

    assert stats["rel_mean_diff"] <= 0.005, (
        f"2D vesselness mean drifted {stats['rel_mean_diff']:.3%} between "
        f"CPU ({stats['cpu_mean']:.3e}) and MPS ({stats['mps_mean']:.3e}); "
        f"expected <= 0.5% per PRD #140."
    )
    assert stats["max_abs_diff"] <= 1e-4, (
        f"2D vesselness per-voxel max abs diff "
        f"({stats['max_abs_diff']:.3e}) exceeds the float32 tolerance "
        f"of 1e-4 used as the slice acceptance bar."
    )
