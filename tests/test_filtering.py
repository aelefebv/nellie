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

from nellie.segmentation.filtering import Filter, FrangiConfig


_CPU = FrangiConfig(device="cpu")
_CPU_KEEP_EDGES = FrangiConfig(device="cpu", remove_edges=False)
_CPU_STRIP_EDGES = FrangiConfig(device="cpu", remove_edges=True)


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
    filt = Filter(imageinfo_3d, _CPU, num_t=2)
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


@pytest.fixture(scope="module")
def frangi_2d_output(imageinfo_2d) -> np.ndarray:
    filt = Filter(imageinfo_2d, _CPU, num_t=2)
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


def test_input_memmap_unchanged(make_imageinfo_3d) -> None:
    info = make_imageinfo_3d()
    src = Path(info.im_path)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    filt = Filter(info, _CPU, num_t=2)
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


def test_oom_fallback_does_not_nameerror(imageinfo_3d, monkeypatch) -> None:
    """Inject a synthetic OOM into the per-frame path; fallback should run, not raise NameError."""
    filt = Filter(imageinfo_3d, _CPU, num_t=2)

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


def test_remove_edges_zeroes_border(make_imageinfo_3d) -> None:
    filt_keep = Filter(make_imageinfo_3d(), _CPU_KEEP_EDGES, num_t=2)
    filt_keep.run()
    nonzero_keep = int(np.count_nonzero(np.asarray(filt_keep.frangi_memmap)))
    _release_filter(filt_keep)

    filt_strip = Filter(make_imageinfo_3d(), _CPU_STRIP_EDGES, num_t=2)
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


def test_2d_log_blobness_fusion(make_imageinfo_2d, monkeypatch) -> None:
    """LoG fusion should add non-zero voxels beyond what Frangi alone produces."""
    from nellie.segmentation import frangi_math

    filt_with = Filter(make_imageinfo_2d(), _CPU, num_t=2)
    filt_with.run()
    nonzero_with = int(np.count_nonzero(np.asarray(filt_with.frangi_memmap)))
    _release_filter(filt_with)

    def zero_log(image, sigmas, sigma_vec_fn, mask, xp, ndi, work_dtype="float32"):
        return xp.zeros_like(image)

    monkeypatch.setattr(frangi_math, "log_blobness", zero_log)
    filt_without = Filter(make_imageinfo_2d(), _CPU, num_t=2)
    filt_without.run()
    nonzero_without = int(np.count_nonzero(np.asarray(filt_without.frangi_memmap)))
    _release_filter(filt_without)

    assert nonzero_with > nonzero_without, (
        f"LoG fusion added no extra signal "
        f"(with={nonzero_with}, without={nonzero_without})"
    )


def test_2d_log_blobness_fusion_low_memory(make_imageinfo_2d, monkeypatch) -> None:
    """LoG fusion must apply in the low-memory chunked path too.

    Pins the fix for a silent divergence: prior to the fix, ``low_memory=True``
    on a 2D image silently dropped the LoG-blobness backfill that the
    non-chunked path adds. ``max_chunk_voxels=10_000`` forces the fixture
    (~53k voxels) to actually chunk, so the per-chunk mask assembly is
    exercised, not just the single-chunk degenerate case.
    """
    from nellie.segmentation import frangi_math

    config_low_mem = FrangiConfig(
        device="cpu", low_memory=True, max_chunk_voxels=10_000
    )

    filt_with = Filter(make_imageinfo_2d(), config_low_mem, num_t=2)
    filt_with.run()
    nonzero_with = int(np.count_nonzero(np.asarray(filt_with.frangi_memmap)))
    _release_filter(filt_with)

    def zero_log(image, sigmas, sigma_vec_fn, mask, xp, ndi, work_dtype="float32"):
        return xp.zeros_like(image)

    monkeypatch.setattr(frangi_math, "log_blobness", zero_log)
    filt_without = Filter(make_imageinfo_2d(), config_low_mem, num_t=2)
    filt_without.run()
    nonzero_without = int(np.count_nonzero(np.asarray(filt_without.frangi_memmap)))
    _release_filter(filt_without)

    assert nonzero_with > nonzero_without, (
        f"LoG fusion silently dropped in low-memory chunked path "
        f"(with={nonzero_with}, without={nonzero_without})"
    )


def test_2d_output_invariants(frangi_2d_output, make_imageinfo_2d) -> None:
    assert frangi_2d_output.dtype == np.float32
    assert frangi_2d_output.min() >= 0
    assert np.isfinite(frangi_2d_output).all()

    info = make_imageinfo_2d()
    src = Path(info.im_path)
    digest_now = hashlib.sha256(src.read_bytes()).hexdigest()
    filt = Filter(info, _CPU, num_t=2)
    filt.run()
    digest_after = hashlib.sha256(src.read_bytes()).hexdigest()
    _release_filter(filt)
    assert digest_now == digest_after


# -------------------------------------------------------------------------
# FrangiConfig validation (__post_init__)
# -------------------------------------------------------------------------

def test_frangi_config_default_constructs() -> None:
    FrangiConfig()


def test_frangi_config_rejects_bad_device() -> None:
    with pytest.raises(ValueError, match="device"):
        FrangiConfig(device="bogus")


def test_frangi_config_accepts_cuda_alias() -> None:
    FrangiConfig(device="cuda")


@pytest.mark.parametrize("field", [
    "min_radius_um", "max_radius_um", "alpha_sq", "beta_sq",
    "frob_thresh_division", "max_chunk_voxels", "max_threshold_samples",
])
def test_frangi_config_rejects_nonpositive_numeric(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        FrangiConfig(**{field: 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=field):
        FrangiConfig(**{field: -1})  # type: ignore[arg-type]


def test_frangi_config_frob_thresh_optional_none_ok() -> None:
    FrangiConfig(frob_thresh=None)


def test_frangi_config_frob_thresh_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="frob_thresh"):
        FrangiConfig(frob_thresh=0)
    with pytest.raises(ValueError, match="frob_thresh"):
        FrangiConfig(frob_thresh=-0.5)


def test_frangi_config_rejects_inverted_radius_range() -> None:
    with pytest.raises(ValueError, match="min_radius_um"):
        FrangiConfig(min_radius_um=2.0, max_radius_um=1.0)


def test_frangi_config_equal_radii_ok() -> None:
    FrangiConfig(min_radius_um=0.5, max_radius_um=0.5)
