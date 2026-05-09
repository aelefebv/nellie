"""Tests for the closed-form 3x3 symmetric eigenvalue helpers in `chunking`.

`safe_eigvalsh` (LAPACK) is the reference. The closed-form helpers must
agree with it to float32 tolerance on real Hessians and handle the
documented degenerate cases (diagonal matrices, scalar identity,
repeated eigenvalues) without producing NaNs.
"""

from __future__ import annotations

import numpy as np
import pytest

from nellie.utils.chunking import (
    eigvalsh_3x3_components,
    eigvalsh_3x3_symmetric,
    safe_eigvalsh,
)


# -------------------------------------------------------------------------
# Equivalence with LAPACK on random symmetric Hessians
# -------------------------------------------------------------------------

@pytest.fixture
def random_symmetric_batch():
    """A batch of (N, 3, 3) symmetric float32 matrices drawn from a wide range."""
    rng = np.random.default_rng(seed=20260509)
    n = 5_000
    # Mix of well-conditioned and tight-eigenvalue cases by varying scale
    a = rng.standard_normal((n, 3, 3)).astype(np.float32) * rng.uniform(
        0.1, 10.0, size=(n, 1, 1)
    ).astype(np.float32)
    H = (a + a.transpose(0, 2, 1)) * np.float32(0.5)
    return H


def test_eigvalsh_3x3_symmetric_matches_lapack(random_symmetric_batch) -> None:
    H = random_symmetric_batch
    expected = safe_eigvalsh(H, np)
    actual = eigvalsh_3x3_symmetric(H, np)

    assert actual.shape == expected.shape == (H.shape[0], 3)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_eigvalsh_3x3_components_matches_tensor(random_symmetric_batch) -> None:
    """The components form must produce the same output as the tensor form."""
    H = random_symmetric_batch
    h11 = H[..., 0, 0]
    h12 = H[..., 0, 1]
    h13 = H[..., 0, 2]
    h22 = H[..., 1, 1]
    h23 = H[..., 1, 2]
    h33 = H[..., 2, 2]

    via_tensor = eigvalsh_3x3_symmetric(H, np)
    via_components = eigvalsh_3x3_components(h11, h12, h13, h22, h23, h33, np)

    np.testing.assert_array_equal(via_tensor, via_components)


def test_eigenvalues_sorted_by_absolute_value(random_symmetric_batch) -> None:
    """`safe_eigvalsh` returns eigenvalues sorted by abs ascending; closed form must match."""
    eigs = eigvalsh_3x3_symmetric(random_symmetric_batch, np)
    abs_eigs = np.abs(eigs)
    assert (abs_eigs[:, 0] <= abs_eigs[:, 1] + 1e-6).all()
    assert (abs_eigs[:, 1] <= abs_eigs[:, 2] + 1e-6).all()


# -------------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------------

def test_diagonal_matrix() -> None:
    """Diagonal matrices: eigenvalues are the diagonal entries (then abs-sorted)."""
    H = np.zeros((1, 3, 3), dtype=np.float32)
    H[0, 0, 0] = 3.0
    H[0, 1, 1] = -1.0
    H[0, 2, 2] = 2.0

    eigs = eigvalsh_3x3_symmetric(H, np)
    expected = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(eigs, expected, atol=1e-5)


def test_scalar_identity_matrix() -> None:
    """A scalar multiple of the identity has all three eigenvalues equal — degenerate case."""
    H = np.zeros((1, 3, 3), dtype=np.float32)
    H[0, 0, 0] = H[0, 1, 1] = H[0, 2, 2] = 4.5

    eigs = eigvalsh_3x3_symmetric(H, np)
    np.testing.assert_allclose(eigs, np.full((1, 3), 4.5, dtype=np.float32), atol=1e-5)
    assert np.isfinite(eigs).all(), "p=0 degenerate case must not produce NaN"


def test_zero_matrix() -> None:
    """All-zero input: every eigenvalue is zero, no NaNs."""
    H = np.zeros((1, 3, 3), dtype=np.float32)
    eigs = eigvalsh_3x3_symmetric(H, np)
    np.testing.assert_array_equal(eigs, np.zeros((1, 3), dtype=np.float32))


def test_two_repeated_eigenvalues() -> None:
    """Diag(a, a, b) — two repeated eigenvalues. Closed form must still agree with LAPACK."""
    H = np.diag(np.array([2.0, 2.0, 5.0], dtype=np.float32))[None, ...]
    eigs = eigvalsh_3x3_symmetric(H, np)
    expected = safe_eigvalsh(H, np)
    np.testing.assert_allclose(eigs, expected, atol=1e-5)


def test_negative_eigenvalues() -> None:
    """Hessians with all-negative eigenvalues (bright tube-like structures)."""
    H = -np.eye(3, dtype=np.float32)[None, ...] * 3.0
    eigs = eigvalsh_3x3_symmetric(H, np)
    np.testing.assert_allclose(eigs, np.full((1, 3), -3.0, dtype=np.float32), atol=1e-5)


def test_empty_batch() -> None:
    """Zero-length batch: returns shape (0, 3) without erroring."""
    H = np.zeros((0, 3, 3), dtype=np.float32)
    eigs = eigvalsh_3x3_symmetric(H, np)
    assert eigs.shape == (0, 3)


# -------------------------------------------------------------------------
# Float32 numerical robustness
# -------------------------------------------------------------------------

def test_no_nans_on_extreme_inputs() -> None:
    """Mixed scales + tight eigenvalues stress the arccos clip — must stay finite."""
    rng = np.random.default_rng(seed=42)
    n = 1_000
    a = rng.standard_normal((n, 3, 3)).astype(np.float32) * np.float32(1e-6)
    H_small = (a + a.transpose(0, 2, 1)) * np.float32(0.5)

    a = rng.standard_normal((n, 3, 3)).astype(np.float32) * np.float32(1e6)
    H_large = (a + a.transpose(0, 2, 1)) * np.float32(0.5)

    for H in (H_small, H_large):
        eigs = eigvalsh_3x3_symmetric(H, np)
        assert np.isfinite(eigs).all(), "closed form must not emit NaN/inf"


# -------------------------------------------------------------------------
# End-to-end Filter equivalence (closed form vs LAPACK on the 3D fixture)
#
# Pins that swapping the 3D eigenvalue path from `safe_eigvalsh` to the
# closed form does not perturb the Frangi output beyond float32 noise on
# real microscopy data — the existing value-band tests would catch a
# gross break, but this test catches subtle drift.
# -------------------------------------------------------------------------

def test_filter_3d_output_matches_lapack_path(make_imageinfo_3d, monkeypatch) -> None:
    import gc

    from nellie.segmentation.filtering import Filter, FrangiConfig
    from nellie.utils import chunking

    config = FrangiConfig(device="cpu")

    def _release(filt: Filter) -> None:
        filt.frangi_memmap = None
        filt.im_memmap = None
        gc.collect()

    f1 = Filter(make_imageinfo_3d(), config, num_t=2)
    f1.run()
    out_closed = np.array(f1.frangi_memmap)
    _release(f1)

    real_components = chunking.eigvalsh_3x3_components

    def lapack_path(h11, h12, h13, h22, h23, h33, xp):
        H = xp.stack(
            [
                xp.stack([h11, h12, h13], axis=-1),
                xp.stack([h12, h22, h23], axis=-1),
                xp.stack([h13, h23, h33], axis=-1),
            ],
            axis=-2,
        )
        return chunking.safe_eigvalsh(H, xp)

    monkeypatch.setattr(chunking, "eigvalsh_3x3_components", lapack_path)
    f2 = Filter(make_imageinfo_3d(), config, num_t=2)
    f2.run()
    out_lapack = np.array(f2.frangi_memmap)
    _release(f2)

    monkeypatch.setattr(chunking, "eigvalsh_3x3_components", real_components)

    np.testing.assert_allclose(out_closed, out_lapack, atol=1e-5, rtol=1e-3)
