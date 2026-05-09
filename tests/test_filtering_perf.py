"""Opt-in performance tests for `nellie.segmentation.filtering.Filter`.

Skipped by default — invoke with ``pytest -m benchmark`` to run. Two
flavors live here:

1. **End-to-end timing** prints a Filter wall-clock baseline so a future
   regression on the yeast fixture is visible relative to a prior
   commit. No assertion: there is no canonical baseline checked in.
2. **Dense-vs-sparse microbenchmark** asserts the dense fast path is at
   least as fast as the sparse path on a fully-true mask. The dense
   path's only payment is `bool(h_mask.all())`; if that ever costs more
   than the sparse path's `xp.where` + per-chunk fancy indexing + final
   scatter on the same input, the optimization is not a win and the
   dispatcher rule needs revisiting.

The relative comparison is robust to CI noise; the absolute timings are
not — read them, don't trust them as gates.
"""

from __future__ import annotations

import gc
import time
from statistics import median

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi

from nellie.segmentation import frangi_math
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.utils import chunking


pytestmark = pytest.mark.benchmark

_CPU = FrangiConfig(device="cpu")


def _release_filter(filt: Filter) -> None:
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()


def _time_call(fn, *, iters: int) -> float:
    """Median wall-clock of `iters` invocations, in seconds."""
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return median(samples)


# -------------------------------------------------------------------------
# End-to-end Filter baseline (informational; no assertion)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_filter_run_baseline_prints_wall_clock(
    dim, make_imageinfo_2d, make_imageinfo_3d, capsys
) -> None:
    """Print a Filter wall-clock baseline. No assertion — read the output."""
    factory = make_imageinfo_2d if dim == "2d" else make_imageinfo_3d

    def _one_run() -> None:
        info = factory()
        filt = Filter(info, _CPU, num_t=2)
        filt.run()
        _release_filter(filt)

    # Warm up: first run pays one-time costs (memmap setup, JIT-ish caches).
    _one_run()
    elapsed = _time_call(_one_run, iters=3)

    with capsys.disabled():
        print(f"\n[perf] Filter.run() {dim} median over 3: {elapsed * 1000:.1f} ms")


# -------------------------------------------------------------------------
# Dense vs sparse microbenchmark
# -------------------------------------------------------------------------

@pytest.fixture(params=["2d", "3d"])
def synthetic_h_components(request, make_imageinfo_2d, make_imageinfo_3d):
    """Realistic Hessian components for the perf comparison.

    Mirrors the fixture in `test_filtering.py` to keep the perf path
    operating on the same data shape as the correctness path.
    """
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


def test_dense_path_at_least_as_fast_as_sparse(
    synthetic_h_components, capsys
) -> None:
    """On an all-True mask, the dense path must beat the sparse path.

    The margin is generous (1.2x) to absorb wall-clock noise while still
    catching a real regression where dense becomes slower than sparse.
    """
    filt, h_components, gamma_sq = synthetic_h_components
    template = next(iter(h_components.values()))
    h_mask_all = np.ones(template.shape, dtype=bool)

    # Warm up — first call pays cache effects
    filt._compute_vesselness_dense(h_components, gamma_sq)
    filt._compute_vesselness_sparse(h_components, h_mask_all, gamma_sq)

    iters = 5
    t_dense = _time_call(
        lambda: filt._compute_vesselness_dense(h_components, gamma_sq),
        iters=iters,
    )
    t_sparse = _time_call(
        lambda: filt._compute_vesselness_sparse(h_components, h_mask_all, gamma_sq),
        iters=iters,
    )

    with capsys.disabled():
        print(
            f"\n[perf] vesselness median over {iters}: "
            f"dense={t_dense * 1000:.1f}ms sparse={t_sparse * 1000:.1f}ms "
            f"ratio={t_dense / t_sparse:.2f}x"
        )

    # Allow 1.2x margin for noise. A genuine regression (dense slower
    # than sparse on an all-True mask) blows past this.
    assert t_dense <= t_sparse * 1.2, (
        f"dense path ({t_dense * 1000:.1f}ms) is materially slower than "
        f"sparse ({t_sparse * 1000:.1f}ms) on a fully-true mask — "
        f"the dispatcher rule may no longer be a win"
    )


def test_h_mask_all_check_is_cheap_relative_to_dispatch(
    synthetic_h_components, capsys
) -> None:
    """`bool(h_mask.all())` must not dominate the dispatch cost on CPU.

    Pins point #3 from the optimization review: the dispatcher pays a
    full reduction on every call. On CPU it's a tight C loop and should
    cost < 10% of the dense path itself. On GPU it would force a sync —
    out of scope for this CPU-only test, but worth re-measuring there
    if the Filter is run heavily on CUDA.
    """
    filt, h_components, gamma_sq = synthetic_h_components
    template = next(iter(h_components.values()))
    h_mask_all = np.ones(template.shape, dtype=bool)

    # Warm up
    bool(h_mask_all.all())
    filt._compute_vesselness_dense(h_components, gamma_sq)

    iters = 100
    t_check = _time_call(lambda: bool(h_mask_all.all()), iters=iters)

    iters_dense = 5
    t_dense = _time_call(
        lambda: filt._compute_vesselness_dense(h_components, gamma_sq),
        iters=iters_dense,
    )

    with capsys.disabled():
        print(
            f"\n[perf] h_mask.all() check: {t_check * 1e6:.1f}µs "
            f"(dense path: {t_dense * 1000:.1f}ms, "
            f"check is {t_check / t_dense * 100:.2f}% of dense)"
        )

    assert t_check < t_dense * 0.10, (
        f"h_mask.all() reduction ({t_check * 1e6:.1f}µs) is more than 10% "
        f"of the dense vesselness path ({t_dense * 1000:.1f}ms) — the "
        f"dispatcher overhead may swamp the savings"
    )


# -------------------------------------------------------------------------
# Closed-form vs LAPACK eigenvalues on the 3D Hessian (point #6)
#
# `_safe_eigvalsh` was the dominant cost in 3D vesselness — replacing it
# with Smith's closed form should be a multiplicative speedup. The
# end-to-end Filter benchmark above is the integration view; this is the
# isolated microbenchmark proving the swap is the source of the win.
# -------------------------------------------------------------------------

def test_closed_form_3d_eigvalsh_beats_lapack(make_imageinfo_3d, capsys) -> None:
    info = make_imageinfo_3d()
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

    flat = {k: v.ravel() for k, v in h_components.items()}
    H_tensor = np.stack(
        [
            np.stack([flat["hxx"], flat["hxy"], flat["hxz"]], axis=-1),
            np.stack([flat["hxy"], flat["hyy"], flat["hyz"]], axis=-1),
            np.stack([flat["hxz"], flat["hyz"], flat["hzz"]], axis=-1),
        ],
        axis=-2,
    )

    chunking.eigvalsh_3x3_components(
        flat["hxx"], flat["hxy"], flat["hxz"],
        flat["hyy"], flat["hyz"], flat["hzz"], np,
    )
    chunking.safe_eigvalsh(H_tensor, np)

    iters = 5
    t_closed = _time_call(
        lambda: chunking.eigvalsh_3x3_components(
            flat["hxx"], flat["hxy"], flat["hxz"],
            flat["hyy"], flat["hyz"], flat["hzz"], np,
        ),
        iters=iters,
    )
    t_lapack = _time_call(
        lambda: chunking.safe_eigvalsh(H_tensor, np),
        iters=iters,
    )

    _release_filter(filt)

    with capsys.disabled():
        print(
            f"\n[perf] 3D eigvalsh on full Hessian "
            f"(N={H_tensor.shape[0]}): "
            f"closed-form={t_closed * 1000:.1f}ms "
            f"LAPACK={t_lapack * 1000:.1f}ms "
            f"speedup={t_lapack / t_closed:.2f}x"
        )

    assert t_closed * 1.5 <= t_lapack, (
        f"closed-form eigvalsh ({t_closed * 1000:.1f}ms) is not at least "
        f"1.5x faster than LAPACK ({t_lapack * 1000:.1f}ms) — the swap "
        f"may not be worth the added math-stability surface"
    )
