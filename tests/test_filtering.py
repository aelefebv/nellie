"""Characterization tests for ``nellie.segmentation.filtering.Filter``.

Tests cover the wiki-documented invariants on both the 3D and 2D paths
plus a regression test for the spacing-geomean γ rescale and an xfail
test for the latent OOM-fallback ``NameError`` (slice 3 turns it green).
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pytest

from nellie.segmentation.filtering import Filter


def _release_filter(filt: Filter) -> None:
    """Drop a Filter's memmap references and force gc.

    Required on Windows: the ``im_preprocessed`` memmap is shared across
    Filter instances on the same ImInfo, and Windows refuses to overwrite
    a file that's still open via mmap. POSIX is happy to overwrite without
    this dance.
    """
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()


# Module-scoped fixtures: run Filter once per fixture, share the output
# across all read-only invariant tests to keep total runtime bounded.

@pytest.fixture(scope="module")
def frangi_3d_output(imageinfo_3d) -> np.ndarray:
    filt = Filter(imageinfo_3d, num_t=2, device="cpu")
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


@pytest.fixture(scope="module")
def frangi_2d_output(imageinfo_2d) -> np.ndarray:
    filt = Filter(imageinfo_2d, num_t=2, device="cpu")
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


# -------------------------------------------------------------------------
# 3D path
# -------------------------------------------------------------------------

def test_filter_runs_end_to_end(frangi_3d_output, imageinfo_3d) -> None:
    assert frangi_3d_output.shape == imageinfo_3d.shape


def test_output_dtype_is_float32(frangi_3d_output) -> None:
    assert frangi_3d_output.dtype == np.float32


def test_output_is_nonnegative(frangi_3d_output) -> None:
    assert frangi_3d_output.min() >= 0


def test_output_finite(frangi_3d_output) -> None:
    assert np.isfinite(frangi_3d_output).all()


def test_input_memmap_unchanged(imageinfo_3d) -> None:
    src = Path(imageinfo_3d.im_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    filt = Filter(imageinfo_3d, num_t=2, device="cpu")
    filt.run()
    after = hashlib.sha256(src.read_bytes()).hexdigest()
    _release_filter(filt)
    assert before == after


def test_response_peak_in_signal_region(frangi_3d_output, imageinfo_3d) -> None:
    """Top Frangi voxels should sit in above-mean intensity regions of the input."""
    raw = np.array(imageinfo_3d.im)
    raw_mean = raw.mean()

    flat_resp = frangi_3d_output.ravel()
    top_idx = np.argpartition(flat_resp, -100)[-100:]
    raw_at_top = raw.ravel()[top_idx]

    fraction_bright = (raw_at_top > raw_mean).mean()
    assert fraction_bright > 0.9, (
        f"Only {fraction_bright:.0%} of top-100 Frangi voxels are above mean intensity"
    )


def test_spacing_geomean_gamma_regression(frangi_3d_output) -> None:
    """Without the γ rescale, ``(1 - exp(-S²/γ²))`` saturates and max → ~1.0.

    On the anisotropic-spacing yeast fixture (Z=0.25, X=Y=0.0655 µm) the
    geomean is ~0.1, so the rescale is meaningfully active. With it
    intact the max stays well below 0.01; without it the response
    saturates orders of magnitude higher.
    """
    out_max = float(frangi_3d_output.max())
    assert 1e-6 < out_max < 0.01, (
        f"3D Frangi max {out_max} outside expected band — γ rescale may be broken"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Slice 3 (#54) fixes the gammas NameError in _run_frame OOM fallback",
)
def test_oom_fallback_does_not_nameerror(imageinfo_3d, monkeypatch) -> None:
    """Inject a synthetic OOM into the per-frame path; fallback should run, not raise NameError."""
    filt = Filter(imageinfo_3d, num_t=2, device="cpu")

    original = Filter._compute_vesselness
    state = {"raised": False}

    def flaky_compute(self, frame, mask=True):
        if not state["raised"]:
            state["raised"] = True
            raise MemoryError("simulated OOM for fallback test")
        return original(self, frame, mask=mask)

    monkeypatch.setattr(Filter, "_compute_vesselness", flaky_compute)
    filt.run()
    assert filt.frangi_memmap is not None
    _release_filter(filt)


def test_remove_edges_zeroes_border(imageinfo_3d) -> None:
    filt_keep = Filter(imageinfo_3d, num_t=2, device="cpu", remove_edges=False)
    filt_keep.run()
    nonzero_keep = int(np.count_nonzero(np.asarray(filt_keep.frangi_memmap)))
    _release_filter(filt_keep)

    filt_strip = Filter(imageinfo_3d, num_t=2, device="cpu", remove_edges=True)
    filt_strip.run()
    nonzero_strip = int(np.count_nonzero(np.asarray(filt_strip.frangi_memmap)))
    _release_filter(filt_strip)

    assert nonzero_strip < nonzero_keep, (
        f"remove_edges=True did not strip any voxels "
        f"(keep={nonzero_keep}, strip={nonzero_strip})"
    )


# -------------------------------------------------------------------------
# 2D path
# -------------------------------------------------------------------------

def test_2d_path_runs_end_to_end(frangi_2d_output, imageinfo_2d) -> None:
    assert frangi_2d_output.shape == imageinfo_2d.shape


def test_2d_log_blobness_fusion(imageinfo_2d, monkeypatch) -> None:
    """LoG fusion should add non-zero voxels beyond what Frangi alone produces."""
    filt_with = Filter(imageinfo_2d, num_t=2, device="cpu")
    filt_with.run()
    nonzero_with = int(np.count_nonzero(np.asarray(filt_with.frangi_memmap)))
    _release_filter(filt_with)

    def zero_log(self, frame, **_kwargs):
        return self.xp.zeros_like(frame)

    monkeypatch.setattr(Filter, "_filter_log", zero_log)
    filt_without = Filter(imageinfo_2d, num_t=2, device="cpu")
    filt_without.run()
    nonzero_without = int(np.count_nonzero(np.asarray(filt_without.frangi_memmap)))
    _release_filter(filt_without)

    assert nonzero_with > nonzero_without, (
        f"LoG fusion added no extra signal "
        f"(with={nonzero_with}, without={nonzero_without})"
    )


def test_2d_output_invariants(frangi_2d_output, imageinfo_2d) -> None:
    assert frangi_2d_output.dtype == np.float32
    assert frangi_2d_output.min() >= 0
    assert np.isfinite(frangi_2d_output).all()

    src = Path(imageinfo_2d.im_path)
    digest_now = hashlib.sha256(src.read_bytes()).hexdigest()
    filt = Filter(imageinfo_2d, num_t=2, device="cpu")
    filt.run()
    digest_after = hashlib.sha256(src.read_bytes()).hexdigest()
    _release_filter(filt)
    assert digest_now == digest_after
