"""Always-on contract tests for the ``nellie.utils.torch_xp`` shim.

These compare every implemented op to its numpy reference on small random
inputs. They DON'T require an MPS device — they run torch on CPU (which
is enough to verify the shim's API contract). When torch isn't installed
the whole module skips, so default ``pytest`` stays green on installs
that opted out of the ``mps`` extra.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


pytest.importorskip("torch")

# Imported lazily after the importorskip so the module-load price is only
# paid when torch is present.
import torch  # noqa: E402

from nellie.utils import torch_xp  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _to_numpy(t):
    """Convert any torch.Tensor to numpy on CPU; pass numpy through unchanged."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _close(actual, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(_to_numpy(actual), expected, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# Aliases / dtype constants / class proxy
# -----------------------------------------------------------------------------


def test_newaxis_is_none() -> None:
    """``xp.newaxis`` is ``None`` in both numpy and torch indexing."""
    assert torch_xp.newaxis is None


def test_inf_is_math_inf() -> None:
    import math
    assert torch_xp.inf == math.inf


def test_ndarray_isinstance_matches_torch_tensor() -> None:
    t = torch.zeros(3)
    assert isinstance(t, torch_xp.ndarray)
    assert not isinstance([1, 2, 3], torch_xp.ndarray)
    assert not isinstance(np.zeros(3), torch_xp.ndarray)


def test_ndarray_construction_raises() -> None:
    with pytest.raises(TypeError, match="proxy"):
        torch_xp.ndarray()


def test_float32_resolves() -> None:
    assert torch_xp.float32 == torch.float32


def test_float16_resolves() -> None:
    assert torch_xp.float16 == torch.float16


def test_float64_silently_coerces_to_float32(caplog) -> None:
    """Per the slice contract, float64 silently maps to float32 (one log msg)."""
    # Reset the one-time flag so the next access re-emits the notice.
    torch_xp._FLOAT64_NOTICE_EMITTED = False
    with caplog.at_level(logging.INFO, logger="nellie.utils.torch_xp"):
        # Touch the proxy to trigger the announcement.
        resolved = torch_xp.float64._resolve()
    assert resolved == torch.float32
    assert any("float64 → float32 coercion" in rec.getMessage() for rec in caplog.records)


def test_zeros_with_explicit_float64_dtype_returns_float32(caplog) -> None:
    """Explicit ``zeros(shape, dtype=xp.float64)`` should also coerce."""
    torch_xp._FLOAT64_NOTICE_EMITTED = False
    with caplog.at_level(logging.INFO, logger="nellie.utils.torch_xp"):
        out = torch_xp.zeros((4,), dtype=torch_xp.float64)
    assert out.dtype == torch.float32


def test_lazy_dtype_call_constructs_typed_scalar() -> None:
    """``xp.float32(3.0)`` should return a 0-d float32 tensor.

    Mirrors numpy's ``np.float32(3.0)`` — used by
    ``chunking.eigvalsh_3x3_components`` (and any future caller that
    needs a typed scalar constant) to keep arithmetic in float32. If
    this regresses, the closed-form 3x3 eigenvalue path on MPS will
    crash with "_LazyDtype is not callable".
    """
    out = torch_xp.float32(3.0)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.ndim == 0
    assert float(out) == pytest.approx(3.0)


def test_lazy_dtype_call_float64_coerces_silently(caplog) -> None:
    """``xp.float64(3.0)`` should also coerce to float32 (with the log notice)."""
    torch_xp._FLOAT64_NOTICE_EMITTED = False
    with caplog.at_level(logging.INFO, logger="nellie.utils.torch_xp"):
        out = torch_xp.float64(3.0)
    assert out.dtype == torch.float32


def test_tensor_astype_method_added() -> None:
    """torch.Tensor should expose an ``astype(dtype, copy=...)`` method.

    Patched by ``torch_xp._patch_tensor_methods`` on first ``_torch()``
    call. Stage code (Filter, frangi_math) was originally written
    against numpy/cupy where ``arr.astype`` is the canonical cast; the
    patch extends that contract to torch tensors so MPS goes through
    the same code paths without per-call wrapping.

    The patch routes the dtype through ``_resolve_dtype``, so a
    ``float64`` request silently coerces to ``float32`` (with the
    one-time log notice) — matching the rest of the shim's float64
    contract.
    """
    # Touch the shim so the patch fires.
    torch_xp._torch()
    t = torch.zeros(3, dtype=torch.float32)
    out = t.astype(torch.int64)
    assert out.dtype == torch.int64
    # Per numpy semantics, copy=True is the default — even when the
    # target dtype matches, we should get a fresh tensor.
    same_dtype = t.astype(torch.float32, copy=True)
    assert same_dtype.dtype == torch.float32
    assert same_dtype is not t
    # copy=False short-circuits when dtype already matches.
    no_copy = t.astype(torch.float32, copy=False)
    assert no_copy is t


def test_tensor_astype_silently_coerces_float64() -> None:
    """``tensor.astype(torch.float64)`` should silently downgrade to float32.

    Matches the rest of the shim's float64 contract — MPS doesn't
    support double precision, so the patched ``astype`` runs the
    requested dtype through ``_resolve_dtype`` (which emits the
    one-time log notice and returns float32 for float64 requests).
    """
    torch_xp._torch()
    t = torch.zeros(3, dtype=torch.float32)
    out = t.astype(torch.float64)
    assert out.dtype == torch.float32


def test_tensor_get_method_added() -> None:
    """torch.Tensor should expose a ``get()`` method (cupy-compat).

    cupy's ``.get()`` returns a host numpy copy. The patch makes the
    same idiom work for torch tensors — ``hasattr(arr, "get")``
    duck-type checks in stage code (e.g. Filter._run_filter line 691)
    fire on MPS too, so the result moves back to host before being
    written into the on-disk memmap.
    """
    torch_xp._torch()
    t = torch.tensor([1.0, 2.0, 3.0])
    out = t.get()
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


# -----------------------------------------------------------------------------
# Construction ops
# -----------------------------------------------------------------------------


def test_zeros() -> None:
    a = torch_xp.zeros((2, 3))
    assert a.shape == (2, 3)
    assert a.dtype == torch.float32
    np.testing.assert_array_equal(_to_numpy(a), np.zeros((2, 3)))


def test_zeros_int_shape() -> None:
    a = torch_xp.zeros(5)
    assert a.shape == (5,)


def test_zeros_with_bool_dtype() -> None:
    a = torch_xp.zeros((3,), dtype=bool)
    assert a.dtype == torch.bool


def test_zeros_like() -> None:
    src = torch.ones((2, 3))
    a = torch_xp.zeros_like(src)
    assert a.shape == src.shape
    assert a.dtype == src.dtype
    assert torch.all(a == 0)


def test_zeros_like_with_dtype_override() -> None:
    src = torch.ones((2, 3))
    a = torch_xp.zeros_like(src, dtype=bool)
    assert a.dtype == torch.bool


def test_ones() -> None:
    a = torch_xp.ones((2, 3))
    np.testing.assert_array_equal(_to_numpy(a), np.ones((2, 3)))


def test_ones_like() -> None:
    src = torch.zeros((2, 3))
    a = torch_xp.ones_like(src)
    assert torch.all(a == 1)


def test_full_like() -> None:
    src = torch.zeros((4,))
    a = torch_xp.full_like(src, 7.0)
    assert torch.all(a == 7)


def test_arange_stop_only() -> None:
    _close(torch_xp.arange(5), np.arange(5))


def test_arange_start_stop() -> None:
    _close(torch_xp.arange(2, 7), np.arange(2, 7))


def test_arange_with_dtype() -> None:
    out = torch_xp.arange(3, dtype=torch.int32)
    assert out.dtype == torch.int32


def test_asarray_passthrough() -> None:
    src = torch.zeros((2, 3))
    out = torch_xp.asarray(src)
    assert out is src


def test_asarray_from_numpy() -> None:
    arr = np.arange(6).reshape(2, 3).astype(np.float32)
    out = torch_xp.asarray(arr)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(_to_numpy(out), arr)


def test_asarray_from_numpy_with_dtype() -> None:
    out = torch_xp.asarray([1.0, 2.0, 3.0], dtype=torch.float32)
    assert out.dtype == torch.float32


def test_meshgrid_xy_default() -> None:
    """numpy meshgrid defaults to 'xy' indexing."""
    x = torch.arange(3, dtype=torch.float32)
    y = torch.arange(4, dtype=torch.float32)
    xx, yy = torch_xp.meshgrid(x, y)
    np_xx, np_yy = np.meshgrid(np.arange(3.0), np.arange(4.0))
    _close(xx, np_xx)
    _close(yy, np_yy)


# -----------------------------------------------------------------------------
# Element-wise math
# -----------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(seed=12345)


@pytest.fixture
def small_random(rng):
    a = rng.standard_normal((4, 5)).astype(np.float32)
    return a, torch.from_numpy(a)


def test_abs(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.abs(a_t), np.abs(a_np))


def test_sqrt(rng) -> None:
    a = rng.uniform(0.1, 10.0, size=(4, 5)).astype(np.float32)
    _close(torch_xp.sqrt(torch.from_numpy(a)), np.sqrt(a))


def test_exp(rng) -> None:
    a = rng.uniform(-2.0, 2.0, size=(4, 5)).astype(np.float32)
    _close(torch_xp.exp(torch.from_numpy(a)), np.exp(a), rtol=1e-5)


def test_log10(rng) -> None:
    a = rng.uniform(0.1, 10.0, size=(4, 5)).astype(np.float32)
    _close(torch_xp.log10(torch.from_numpy(a)), np.log10(a), rtol=1e-5)


def test_cos(rng) -> None:
    a = rng.uniform(-3.14, 3.14, size=(4, 5)).astype(np.float32)
    _close(torch_xp.cos(torch.from_numpy(a)), np.cos(a))


def test_arccos(rng) -> None:
    a = rng.uniform(-0.99, 0.99, size=(4, 5)).astype(np.float32)
    _close(torch_xp.arccos(torch.from_numpy(a)), np.arccos(a))


def test_sign(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.sign(a_t), np.sign(a_np))


def test_ceil(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.ceil(a_t), np.ceil(a_np))


def test_clip(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.clip(a_t, -0.5, 0.5), np.clip(a_np, -0.5, 0.5))


def test_isfinite() -> None:
    a = np.array([1.0, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    _close(torch_xp.isfinite(torch.from_numpy(a)), np.isfinite(a))


def test_isinf() -> None:
    a = np.array([1.0, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    _close(torch_xp.isinf(torch.from_numpy(a)), np.isinf(a))


def test_nan_to_num() -> None:
    a = np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32)
    _close(torch_xp.nan_to_num(torch.from_numpy(a)), np.nan_to_num(a))


def test_maximum_two_arrays(rng) -> None:
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    _close(torch_xp.maximum(torch.from_numpy(a), torch.from_numpy(b)), np.maximum(a, b))


def test_maximum_with_out(rng) -> None:
    """The ``out=`` write-target convention used in Filter._compute_vesselness."""
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    a_t = torch.from_numpy(a.copy())
    out = a_t.clone()
    torch_xp.maximum(out, torch.from_numpy(b), out=out)
    _close(out, np.maximum(a, b))


# -----------------------------------------------------------------------------
# Reductions and scans
# -----------------------------------------------------------------------------


def test_any_no_axis() -> None:
    a = np.array([[False, False], [True, False]])
    assert bool(torch_xp.any(torch.from_numpy(a))) == bool(np.any(a))


def test_any_with_axis() -> None:
    a = np.array([[True, False], [False, False]])
    _close(torch_xp.any(torch.from_numpy(a), axis=1), np.any(a, axis=1))


def test_sum_no_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.sum(a_t), np.sum(a_np), rtol=1e-5)


def test_sum_with_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.sum(a_t, axis=0), np.sum(a_np, axis=0), rtol=1e-5)


def test_sum_with_dtype(small_random) -> None:
    a_np, a_t = small_random
    out = torch_xp.sum(a_t, dtype=torch.float32)
    _close(out, np.sum(a_np), rtol=1e-5)


def test_nansum() -> None:
    a = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float32)
    _close(torch_xp.nansum(torch.from_numpy(a)), np.nansum(a), rtol=1e-5)


def test_max_no_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.max(a_t), np.max(a_np))


def test_max_with_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.max(a_t, axis=0), np.max(a_np, axis=0))


def test_min_no_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.min(a_t), np.min(a_np))


def test_min_with_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.min(a_t, axis=1), np.min(a_np, axis=1))


def test_argmin_no_axis(small_random) -> None:
    a_np, a_t = small_random
    assert int(torch_xp.argmin(a_t)) == int(np.argmin(a_np))


def test_argmin_with_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.argmin(a_t, axis=0), np.argmin(a_np, axis=0))


def test_argmax_with_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.argmax(a_t, axis=1), np.argmax(a_np, axis=1))


def test_cumsum(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.cumsum(a_t, axis=0), np.cumsum(a_np, axis=0), rtol=1e-5)


def test_cumsum_no_axis_flattens(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.cumsum(a_t), np.cumsum(a_np), rtol=1e-5)


def test_percentile(rng) -> None:
    a = rng.standard_normal((20, 30)).astype(np.float32)
    _close(torch_xp.percentile(torch.from_numpy(a), 25), np.percentile(a, 25), rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# Comparison / selection
# -----------------------------------------------------------------------------


def test_where_three_arg(rng) -> None:
    cond = rng.integers(0, 2, size=(4, 5)).astype(bool)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    _close(
        torch_xp.where(torch.from_numpy(cond), torch.from_numpy(a), torch.from_numpy(b)),
        np.where(cond, a, b),
    )


def test_where_one_arg() -> None:
    """``where(cond)`` returns a tuple of 1-D index tensors per dim."""
    cond = np.array([[True, False], [False, True]], dtype=bool)
    out = torch_xp.where(torch.from_numpy(cond))
    assert isinstance(out, tuple)
    assert len(out) == 2
    np_idx = np.where(cond)
    for axis in range(2):
        np.testing.assert_array_equal(_to_numpy(out[axis]), np_idx[axis])


def test_argwhere() -> None:
    cond = np.array([[True, False], [False, True]], dtype=bool)
    _close(torch_xp.argwhere(torch.from_numpy(cond)), np.argwhere(cond))


def test_flatnonzero() -> None:
    a = np.array([0, 1, 0, 2, 3, 0], dtype=np.int64)
    _close(torch_xp.flatnonzero(torch.from_numpy(a)), np.flatnonzero(a))


def test_take_along_axis(rng) -> None:
    arr = rng.standard_normal((3, 4)).astype(np.float32)
    idx = np.argsort(arr, axis=1)
    out = torch_xp.take_along_axis(torch.from_numpy(arr), torch.from_numpy(idx), axis=1)
    _close(out, np.take_along_axis(arr, idx, axis=1))


def test_argsort_default_axis(small_random) -> None:
    a_np, a_t = small_random
    _close(torch_xp.argsort(a_t), np.argsort(a_np, axis=-1))


# -----------------------------------------------------------------------------
# Combining / shape
# -----------------------------------------------------------------------------


def test_stack(rng) -> None:
    a = rng.standard_normal((3,)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    out = torch_xp.stack([torch.from_numpy(a), torch.from_numpy(b)], axis=0)
    _close(out, np.stack([a, b], axis=0))


def test_concatenate(rng) -> None:
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    out = torch_xp.concatenate([torch.from_numpy(a), torch.from_numpy(b)], axis=0)
    _close(out, np.concatenate([a, b], axis=0))


def test_flip(rng) -> None:
    a = rng.standard_normal((3, 4)).astype(np.float32)
    _close(torch_xp.flip(torch.from_numpy(a), axis=0), np.flip(a, axis=0))


# -----------------------------------------------------------------------------
# Histogram / counting
# -----------------------------------------------------------------------------


def test_bincount() -> None:
    a = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    _close(torch_xp.bincount(torch.from_numpy(a)), np.bincount(a))


def test_histogram_with_range(rng) -> None:
    a = rng.uniform(0, 10, size=200).astype(np.float32)
    counts_xp, edges_xp = torch_xp.histogram(torch.from_numpy(a), bins=10, range=(0, 10))
    counts_np, edges_np = np.histogram(a, bins=10, range=(0, 10))
    np.testing.assert_array_equal(_to_numpy(counts_xp), counts_np)
    _close(edges_xp, edges_np)


# -----------------------------------------------------------------------------
# Gradient
# -----------------------------------------------------------------------------


def test_gradient_axis_arg(rng) -> None:
    """Single-axis path: ``xp.gradient(image, h, axis=0)``."""
    a = rng.standard_normal((6, 7)).astype(np.float32)
    out = torch_xp.gradient(torch.from_numpy(a), 0.5, axis=0)
    expected = np.gradient(a, 0.5, axis=0)
    _close(out, expected, rtol=1e-5, atol=1e-5)


def test_gradient_multi_axis_2d(rng) -> None:
    """Multi-axis 2-D: ``xp.gradient(image, h0, h1)``."""
    a = rng.standard_normal((6, 7)).astype(np.float32)
    out = torch_xp.gradient(torch.from_numpy(a), 0.5, 0.7)
    np_g = np.gradient(a, 0.5, 0.7)
    assert isinstance(out, tuple) and len(out) == 2
    for axis in range(2):
        _close(out[axis], np_g[axis], rtol=1e-5, atol=1e-5)


def test_gradient_multi_axis_3d(rng) -> None:
    """Multi-axis 3-D: ``xp.gradient(image, h0, h1, h2)``."""
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    out = torch_xp.gradient(torch.from_numpy(a), 0.5, 0.7, 1.1)
    np_g = np.gradient(a, 0.5, 0.7, 1.1)
    assert isinstance(out, tuple) and len(out) == 3
    for axis in range(3):
        _close(out[axis], np_g[axis], rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Linalg
# -----------------------------------------------------------------------------


def test_linalg_eigvalsh(rng) -> None:
    """Symmetric matrix eigenvalues match numpy.linalg.eigvalsh."""
    n = 4
    raw = rng.standard_normal((n, n)).astype(np.float32)
    sym = (raw + raw.T) / 2
    out = torch_xp.linalg.eigvalsh(torch.from_numpy(sym))
    _close(out, np.linalg.eigvalsh(sym), rtol=1e-4, atol=1e-4)


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------


def test_finfo() -> None:
    info = torch_xp.finfo(torch.float32)
    assert info.eps > 0
    assert info.tiny > 0


def test_finfo_from_lazy_dtype() -> None:
    info = torch_xp.finfo(torch_xp.float32)
    np_info = np.finfo(np.float32)
    assert info.eps == pytest.approx(np_info.eps, rel=1e-3)


def test_asnumpy_passes_through_numpy() -> None:
    arr = np.array([1, 2, 3])
    out = torch_xp.asnumpy(arr)
    assert isinstance(out, np.ndarray)


def test_asnumpy_converts_tensor() -> None:
    t = torch.arange(5)
    out = torch_xp.asnumpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.arange(5))


# -----------------------------------------------------------------------------
# Unimplemented op trap
# -----------------------------------------------------------------------------


def test_missing_op_raises_clear_error() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        torch_xp.totally_made_up_op_for_test
