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
import scipy.ndimage as scipy_ndi

from nellie.segmentation import frangi_math
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


# -------------------------------------------------------------------------
# `_get_frob_mask` inf handling
#
# `frobenius_norm = sqrt(...) / max_abs` from `compute_hessian` cannot
# produce infs in normal pipelines (max_abs is guarded against zero), so
# the inf branch is unreachable from real fixtures. Synthetic arrays
# pin the contract directly: inf voxels stay in the mask, the input is
# never mutated, and the pathological all-inf case yields no signal.
# -------------------------------------------------------------------------

def test_get_frob_mask_with_infs_keeps_them_and_preserves_input(
    make_imageinfo_2d,
) -> None:
    info = make_imageinfo_2d()
    filt = Filter(info, _CPU, num_t=2)

    arr = np.array(
        [[0.0, 0.5, 1.0, 2.0],
         [3.0, np.inf, 5.0, np.inf],
         [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float32,
    )
    snapshot = arr.copy()

    mask = filt._get_frob_mask(arr)

    assert mask.dtype == np.bool_
    assert mask.shape == arr.shape
    assert mask[1, 1] and mask[1, 3], "inf voxels must end up in the mask"
    np.testing.assert_array_equal(arr, snapshot)

    _release_filter(filt)


def test_get_frob_mask_all_infs_yields_no_signal(make_imageinfo_2d) -> None:
    info = make_imageinfo_2d()
    filt = Filter(info, _CPU, num_t=2)

    arr = np.full((4, 4), np.inf, dtype=np.float32)
    mask = filt._get_frob_mask(arr)

    assert mask.dtype == np.bool_
    assert not mask.any(), (
        "all-inf input has no finite signal to threshold against; "
        "mask must be all-False to match prior behavior"
    )

    _release_filter(filt)


# -------------------------------------------------------------------------
# `_backend_for_array` dispatch
# -------------------------------------------------------------------------

def test_backend_for_array_dispatches_numpy(make_imageinfo_2d) -> None:
    """A NumPy array must map to the (numpy, scipy.ndimage) backend pair."""
    info = make_imageinfo_2d()
    filt = Filter(info, _CPU, num_t=2)

    xp, ndi = filt._backend_for_array(np.array([1.0, 2.0], dtype=np.float32))

    assert xp is np
    assert ndi is scipy_ndi

    _release_filter(filt)


def test_mask_volume_returns_input_object_in_place(make_imageinfo_2d) -> None:
    """`_mask_volume` mutates and returns its input — pin against accidental revert.

    Pre-rewrite computed `frangi_frame * frangi_mask` (allocating a new
    full-volume array per frame); the in-place rewrite multiplies in
    place and returns the same object. Bit-identical for downstream
    callers — `_run_filter` discards its caller-side reference and
    immediately overwrites with the masked result — but the
    same-object identity is the cheapest pin against an accidental
    revert in a future cleanup.
    """
    info = make_imageinfo_2d()
    filt = Filter(info, _CPU, num_t=2)

    frame = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)
    out = filt._mask_volume(frame)

    assert out is frame, (
        "_mask_volume must operate in place — switch back to `frangi_frame "
        "*= frangi_mask` if a refactor reintroduced the allocating `*` form"
    )

    _release_filter(filt)


# -------------------------------------------------------------------------
# Dense vs sparse vesselness paths
#
# The dense fast path (active when h_mask covers every voxel) must be
# numerically identical to the existing sparse path, which iterates over
# the masked voxels via xp.where + scatter. Parametrized over 2D and 3D
# so the closed-form 2D and 3D eigenvalue paths both get a focused
# equivalence check.
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def synthetic_h_components(request, make_imageinfo_2d, make_imageinfo_3d):
    """Realistic Hessian components from the first frame of the chosen fixture."""
    factory = make_imageinfo_2d if request.param == "2d" else make_imageinfo_3d
    info = factory()
    filt = Filter(info, _CPU, num_t=2)
    filt._get_t()
    filt._set_default_sigmas()

    raw = np.asarray(filt.im_info.get_memmap(filt.im_info.im_path)[0], dtype=np.float32)
    sigma_vec = filt._get_sigma_vec(filt.sigmas[0])
    smoothed = scipy_ndi.gaussian_filter(
        raw, sigma=sigma_vec, mode="reflect", truncate=filt.truncate
    )
    spacing = filt._get_spacing(smoothed.ndim)
    h_components, _frob = frangi_math.compute_hessian(
        smoothed, spacing, low_memory=False, xp=np, work_dtype="float32"
    )
    gamma_sq = 2.0 * (1e-3 ** 2)

    yield filt, h_components, gamma_sq

    _release_filter(filt)


def test_dense_and_sparse_vesselness_paths_agree(synthetic_h_components) -> None:
    filt, h_components, gamma_sq = synthetic_h_components
    template = next(iter(h_components.values()))
    h_mask_all = np.ones(template.shape, dtype=bool)

    dense_out = filt._compute_vesselness_dense(h_components, gamma_sq)
    sparse_out = filt._compute_vesselness_sparse(h_components, h_mask_all, gamma_sq)

    np.testing.assert_allclose(dense_out, sparse_out, atol=1e-6)


def test_dispatcher_picks_dense_when_mask_full(synthetic_h_components) -> None:
    """`_compute_vesselness_chunkwise` must route an all-True mask to the dense path."""
    filt, h_components, gamma_sq = synthetic_h_components
    template = next(iter(h_components.values()))
    h_mask_all = np.ones(template.shape, dtype=bool)

    calls = {"dense": 0, "sparse": 0}
    real_dense = Filter._compute_vesselness_dense
    real_sparse = Filter._compute_vesselness_sparse

    def spy_dense(self, h_components, gamma_sq):
        calls["dense"] += 1
        return real_dense(self, h_components, gamma_sq)

    def spy_sparse(self, h_components, h_mask, gamma_sq):
        calls["sparse"] += 1
        return real_sparse(self, h_components, h_mask, gamma_sq)

    try:
        Filter._compute_vesselness_dense = spy_dense
        Filter._compute_vesselness_sparse = spy_sparse
        filt._compute_vesselness_chunkwise(h_components, h_mask_all, gamma_sq)
    finally:
        Filter._compute_vesselness_dense = real_dense
        Filter._compute_vesselness_sparse = real_sparse

    assert calls == {"dense": 1, "sparse": 0}


def test_dispatcher_picks_sparse_when_mask_partial(synthetic_h_components) -> None:
    filt, h_components, gamma_sq = synthetic_h_components
    template = next(iter(h_components.values()))
    h_mask = np.ones(template.shape, dtype=bool)
    # One False voxel — forces the sparse path. Index is shape-agnostic so
    # the same line works for both 2D and 3D fixtures.
    h_mask.flat[0] = False

    calls = {"dense": 0, "sparse": 0}
    real_dense = Filter._compute_vesselness_dense
    real_sparse = Filter._compute_vesselness_sparse

    def spy_dense(self, h_components, gamma_sq):
        calls["dense"] += 1
        return real_dense(self, h_components, gamma_sq)

    def spy_sparse(self, h_components, h_mask, gamma_sq):
        calls["sparse"] += 1
        return real_sparse(self, h_components, h_mask, gamma_sq)

    try:
        Filter._compute_vesselness_dense = spy_dense
        Filter._compute_vesselness_sparse = spy_sparse
        filt._compute_vesselness_chunkwise(h_components, h_mask, gamma_sq)
    finally:
        Filter._compute_vesselness_dense = real_dense
        Filter._compute_vesselness_sparse = real_sparse

    assert calls == {"dense": 0, "sparse": 1}


# -------------------------------------------------------------------------
# PRD #233 Slice 1 — pin _compute_vesselness reduction pattern + isinf scope
#
# In-band determinism snapshots and call-count tests pin the current
# implementation so Slice 2's rewrite (subsample-first inf + fused
# any/all into sum) can be verified bit-identical for finite-only
# fixture data (locally, before merge) and structurally correct via
# reduction-count flips (in CI). Frangi math is SIMD-sensitive across
# platforms — hardcoded SHAs would diverge in CI on Linux/Windows; the
# determinism check (run twice on independent fixtures, assert SHAs
# match) is the prior pattern from PRDs #217/#222/#227.
# -------------------------------------------------------------------------


def test_run_filter_3d_snapshot_post_rewrite(make_imageinfo_3d) -> None:
    """POST-REWRITE: 3D Filter output is deterministic across runs.

    PRD #233 Slice 2 fused the per-sigma `xp.any` + `bool(h_mask.all())`
    dispatch into a single `xp.sum(h_mask)` and moved `_get_frob_mask`
    inf-handling onto the subsample. For finite-only fixture data the
    rewrite is bit-identical (verified locally before merge — pre/post
    SHAs match: `5a86...c77` for 3D, `bc62...57d` for 2D on macOS).
    This in-band determinism check survives platform drift in CI.
    """
    info = make_imageinfo_3d()
    filt = Filter(info, _CPU, num_t=2)
    filt.run()
    out1 = np.array(filt.frangi_memmap)
    sha1 = hashlib.sha256(out1.tobytes()).hexdigest()
    _release_filter(filt)

    info2 = make_imageinfo_3d()
    filt2 = Filter(info2, _CPU, num_t=2)
    filt2.run()
    out2 = np.array(filt2.frangi_memmap)
    sha2 = hashlib.sha256(out2.tobytes()).hexdigest()
    _release_filter(filt2)

    assert sha1 == sha2, (
        "POST-REWRITE: Filter is not deterministic on the same 3D fixture; "
        "the dispatch fusion (sum-derived is_dense) or subsample-first "
        "inf handling broke determinism"
    )
    # Sanity: snapshot is non-trivial.
    assert (out1 > 0).any()


def test_run_filter_2d_snapshot_post_rewrite(make_imageinfo_2d) -> None:
    """POST-REWRITE: 2D Filter output is deterministic across runs."""
    info = make_imageinfo_2d()
    filt = Filter(info, _CPU, num_t=2)
    filt.run()
    out1 = np.array(filt.frangi_memmap)
    sha1 = hashlib.sha256(out1.tobytes()).hexdigest()
    _release_filter(filt)

    info2 = make_imageinfo_2d()
    filt2 = Filter(info2, _CPU, num_t=2)
    filt2.run()
    out2 = np.array(filt2.frangi_memmap)
    sha2 = hashlib.sha256(out2.tobytes()).hexdigest()
    _release_filter(filt2)

    assert sha1 == sha2, (
        "POST-REWRITE: Filter is not deterministic on the same 2D fixture; "
        "the dispatch fusion (sum-derived is_dense) or subsample-first "
        "inf handling broke determinism"
    )
    assert (out1 > 0).any()


class _RecordingXp:
    """Wrap a backend module (numpy/cupy/torch_xp) to record every call.

    Each call appends ``(name, arg_shapes)`` to ``self.calls`` where
    ``arg_shapes`` is the tuple of shapes for any positional args that
    expose a real ``.shape`` attribute. Used to pin the per-sigma
    reduction pattern in ``_compute_vesselness`` / ``_get_frob_mask``.

    Note: only catches calls routed through ``self.xp`` — array methods
    like ``arr.all()`` / ``arr.any()`` are invisible to this recorder.
    """

    def __init__(self, real_xp):
        self._real = real_xp
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            shapes = []
            for a in args:
                shape = getattr(a, "shape", None)
                # numpy arrays expose ``.shape`` as a tuple attribute;
                # torch tensors expose it as a method. Only record
                # ndarray-style shapes here — that's all we need.
                if shape is not None and not callable(shape):
                    shapes.append(tuple(shape))
            self.calls.append((name, tuple(shapes)))
            return attr(*args, **kwargs)

        return wrapped


def _xp_calls_with_first_arg_shape(recorder: _RecordingXp, name: str, shape: tuple) -> list:
    return [
        c for c in recorder.calls
        if c[0] == name and c[1] and c[1][0] == shape
    ]


def _record_one_compute_vesselness(filt: Filter) -> tuple[_RecordingXp, tuple]:
    """Run a single ``_compute_vesselness`` pass with a recording xp wrapper.

    Returns the recorder + the frame shape so callers can filter calls
    by full-volume vs subsample shape.
    """
    filt._get_t()
    filt._set_default_sigmas()
    raw = np.asarray(filt.im_info.get_memmap(filt.im_info.im_path)[0], dtype=np.float32)

    recorder = _RecordingXp(filt.xp)
    filt.xp = recorder
    filt._compute_vesselness(raw, mask=True)
    return recorder, raw.shape


def test_compute_vesselness_fuses_any_and_all_into_sum_post_rewrite(make_imageinfo_3d) -> None:
    """POST-REWRITE: ``_compute_vesselness`` dispatches via single ``xp.sum(h_mask)`` per sigma.

    PRD #233 Slice 2 collapsed the prior ``xp.any(h_mask)`` skip-check
    (in ``_compute_vesselness``) + ``bool(h_mask.all())`` dispatch
    (in ``_compute_vesselness_chunkwise``) into a single
    ``xp.sum(h_mask)`` and threaded the resulting ``is_dense`` flag
    through to the chunkwise dispatcher. Per sigma we now expect:

    - At least one ``xp.sum(h_mask)`` call (full-volume bool shape).
    - Zero ``xp.any(h_mask)`` calls in the dispatch path
      (``_get_frob_mask`` no longer scans inf_mask either, so zero
      full-volume ``xp.any`` calls overall).
    """
    info = make_imageinfo_3d()
    filt = Filter(info, _CPU, num_t=2)

    recorder, frame_shape = _record_one_compute_vesselness(filt)

    full_volume_any = _xp_calls_with_first_arg_shape(recorder, "any", frame_shape)
    full_volume_sum = _xp_calls_with_first_arg_shape(recorder, "sum", frame_shape)

    assert filt.sigmas is not None  # set by _record_one_compute_vesselness
    n_sigmas = len(filt.sigmas)
    assert len(full_volume_any) == 0, (
        f"post-rewrite removed both `xp.any(inf_mask)` "
        f"(in _get_frob_mask) and `xp.any(h_mask)` "
        f"(in _compute_vesselness); got {len(full_volume_any)} "
        f"full-volume xp.any calls: {full_volume_any}"
    )
    assert len(full_volume_sum) >= n_sigmas, (
        f"expected ≥ {n_sigmas} full-volume xp.sum calls (1 per sigma "
        f"on h_mask in _compute_vesselness's fused dispatch), got "
        f"{len(full_volume_sum)}: {full_volume_sum}"
    )

    _release_filter(filt)


def test_get_frob_mask_uses_subsample_isfinite_post_rewrite(make_imageinfo_3d) -> None:
    """POST-REWRITE: ``_get_frob_mask`` runs ``xp.isfinite`` on the subsample, not full volume.

    PRD #233 Slice 2 moved inf-detection from the full
    ``frobenius_norm`` to the subsample. Per sigma we now expect:

    - Zero full-volume ``xp.isinf`` calls (the all-inf-fallback path
      only fires when the subsample is empty or all-inf, which the
      finite fixture data never triggers).
    - At least one ``xp.isfinite`` call (on a non-full-volume shape —
      subsample is always smaller than the volume on the test
      fixtures, since the subsampler caps at ``max_threshold_samples``
      and additionally filters ``> 0``).
    """
    info = make_imageinfo_3d()
    filt = Filter(info, _CPU, num_t=2)

    recorder, frame_shape = _record_one_compute_vesselness(filt)

    full_volume_isinf = _xp_calls_with_first_arg_shape(recorder, "isinf", frame_shape)
    isfinite_calls = [c for c in recorder.calls if c[0] == "isfinite" and c[1]]
    subsample_isfinite_calls = [
        c for c in isfinite_calls if c[1][0] != frame_shape
    ]

    assert filt.sigmas is not None
    n_sigmas = len(filt.sigmas)
    assert len(full_volume_isinf) == 0, (
        f"post-rewrite must not call xp.isinf on the full frobenius_norm; "
        f"got {len(full_volume_isinf)} full-volume xp.isinf calls: "
        f"{full_volume_isinf}"
    )
    assert len(subsample_isfinite_calls) >= n_sigmas, (
        f"expected ≥ {n_sigmas} subsample-shape xp.isfinite calls "
        f"(1 per sigma in _get_frob_mask), got "
        f"{len(subsample_isfinite_calls)}: {subsample_isfinite_calls}"
    )

    _release_filter(filt)
