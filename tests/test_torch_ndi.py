"""Always-on contract tests for the ``nellie.utils.torch_ndi`` shim.

Compares every implemented op to its scipy.ndimage reference on small
random inputs. The tests use torch on CPU (no MPS hardware needed). When
torch isn't installed the whole module skips, so default ``pytest`` stays
green on installs that opted out of the ``mps`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi


pytest.importorskip("torch")

import torch  # noqa: E402

from nellie.utils import torch_ndi  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _close(actual, expected, rtol=1e-3, atol=1e-3):
    np.testing.assert_allclose(_to_numpy(actual), expected, rtol=rtol, atol=atol)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def img_2d(rng):
    arr = rng.standard_normal((10, 12)).astype(np.float32)
    return arr, torch.from_numpy(arr)


@pytest.fixture
def img_3d(rng):
    arr = rng.standard_normal((6, 8, 10)).astype(np.float32)
    return arr, torch.from_numpy(arr)


# -----------------------------------------------------------------------------
# gaussian_filter
# -----------------------------------------------------------------------------


def test_gaussian_filter_2d_scalar_sigma(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.gaussian_filter(a_t, sigma=1.5, mode="reflect")
    expected = scipy_ndi.gaussian_filter(a_np, sigma=1.5, mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_gaussian_filter_2d_per_axis_sigma(img_2d) -> None:
    a_np, a_t = img_2d
    sigmas = (1.0, 2.0)
    out = torch_ndi.gaussian_filter(a_t, sigma=sigmas, mode="reflect")
    expected = scipy_ndi.gaussian_filter(a_np, sigma=sigmas, mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_gaussian_filter_3d_per_axis_sigma(img_3d) -> None:
    a_np, a_t = img_3d
    sigmas = (1.0, 1.5, 2.0)
    out = torch_ndi.gaussian_filter(a_t, sigma=sigmas, mode="reflect")
    expected = scipy_ndi.gaussian_filter(a_np, sigma=sigmas, mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_gaussian_filter_constant_mode(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.gaussian_filter(a_t, sigma=1.0, mode="constant", cval=0.0)
    expected = scipy_ndi.gaussian_filter(a_np, sigma=1.0, mode="constant", cval=0.0)
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_gaussian_filter_with_output_arg(img_2d) -> None:
    """Mirrors Filter._compute_vesselness_chunkwise's incremental-sigma update."""
    a_np, a_t = img_2d
    out_buf = a_t.clone()
    returned = torch_ndi.gaussian_filter(out_buf, sigma=1.0, output=out_buf, mode="reflect")
    expected = scipy_ndi.gaussian_filter(a_np, sigma=1.0, mode="reflect")
    _close(out_buf, expected, rtol=1e-3, atol=1e-3)
    # scipy's contract: returned tensor IS the output buffer.
    assert returned is out_buf


def test_gaussian_filter_zero_sigma_is_passthrough() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = torch_ndi.gaussian_filter(torch.from_numpy(a), sigma=0)
    _close(out, a)


# -----------------------------------------------------------------------------
# uniform_filter
# -----------------------------------------------------------------------------


def test_uniform_filter_2d(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.uniform_filter(a_t, size=3, mode="reflect")
    expected = scipy_ndi.uniform_filter(a_np, size=3, mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_uniform_filter_3d(img_3d) -> None:
    a_np, a_t = img_3d
    out = torch_ndi.uniform_filter(a_t, size=3, mode="reflect")
    expected = scipy_ndi.uniform_filter(a_np, size=3, mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_uniform_filter_per_axis_size(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.uniform_filter(a_t, size=(3, 5), mode="reflect")
    expected = scipy_ndi.uniform_filter(a_np, size=(3, 5), mode="reflect")
    _close(out, expected, rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# convolve
# -----------------------------------------------------------------------------


def test_convolve_2d_symmetric_kernel(img_2d) -> None:
    """The networking stage uses ``ones((3, 3))`` — a symmetric kernel."""
    a_np, a_t = img_2d
    weights = np.ones((3, 3), dtype=np.float32)
    out = torch_ndi.convolve(a_t, torch.from_numpy(weights), mode="constant", cval=0.0)
    expected = scipy_ndi.convolve(a_np, weights, mode="constant", cval=0.0)
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_convolve_3d_symmetric_kernel(img_3d) -> None:
    """networking 3D path: ``ones((3, 3, 3))``."""
    a_np, a_t = img_3d
    weights = np.ones((3, 3, 3), dtype=np.float32)
    out = torch_ndi.convolve(a_t, torch.from_numpy(weights), mode="constant", cval=0.0)
    expected = scipy_ndi.convolve(a_np, weights, mode="constant", cval=0.0)
    _close(out, expected, rtol=1e-3, atol=1e-3)


def test_convolve_2d_asymmetric_kernel_matches_scipy(rng) -> None:
    """Asymmetric kernel verifies the scipy "true convolution" semantics
    (we pre-flip the kernel before forwarding to torch's correlation)."""
    a = rng.standard_normal((10, 12)).astype(np.float32)
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    out = torch_ndi.convolve(torch.from_numpy(a), torch.from_numpy(weights),
                             mode="constant", cval=0.0)
    expected = scipy_ndi.convolve(a, weights, mode="constant", cval=0.0)
    _close(out, expected, rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# gaussian_laplace
# -----------------------------------------------------------------------------


def test_gaussian_laplace_2d(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.gaussian_laplace(a_t, sigma=1.0, mode="reflect")
    expected = scipy_ndi.gaussian_laplace(a_np, sigma=1.0, mode="reflect")
    _close(out, expected, rtol=1e-2, atol=1e-2)


def test_gaussian_laplace_3d(img_3d) -> None:
    a_np, a_t = img_3d
    out = torch_ndi.gaussian_laplace(a_t, sigma=1.0, mode="reflect")
    expected = scipy_ndi.gaussian_laplace(a_np, sigma=1.0, mode="reflect")
    _close(out, expected, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# maximum_filter
# -----------------------------------------------------------------------------


def test_maximum_filter_2d(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.maximum_filter(a_t, size=3, mode="constant", cval=-np.inf)
    expected = scipy_ndi.maximum_filter(a_np, size=3, mode="constant", cval=-np.inf)
    _close(out, expected, atol=1e-5)


def test_maximum_filter_3d(img_3d) -> None:
    a_np, a_t = img_3d
    out = torch_ndi.maximum_filter(a_t, size=3, mode="constant", cval=-np.inf)
    expected = scipy_ndi.maximum_filter(a_np, size=3, mode="constant", cval=-np.inf)
    _close(out, expected, atol=1e-5)


def test_maximum_filter_per_axis_size(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.maximum_filter(a_t, size=(3, 5), mode="constant", cval=-np.inf)
    expected = scipy_ndi.maximum_filter(a_np, size=(3, 5), mode="constant", cval=-np.inf)
    _close(out, expected, atol=1e-5)


def test_maximum_filter_with_output_arg(img_2d) -> None:
    """Mirrors hu_tracking's ``ndi.maximum_filter(..., output=...)`` call."""
    a_np, a_t = img_2d
    out_buf = a_t.clone()
    returned = torch_ndi.maximum_filter(
        out_buf, size=3, output=out_buf, mode="reflect"
    )
    expected = scipy_ndi.maximum_filter(a_np, size=3, mode="reflect")
    _close(out_buf, expected, atol=1e-5)
    assert returned is out_buf


# -----------------------------------------------------------------------------
# minimum_filter
# -----------------------------------------------------------------------------


def test_minimum_filter_2d(img_2d) -> None:
    a_np, a_t = img_2d
    out = torch_ndi.minimum_filter(a_t, size=3, mode="constant", cval=np.inf)
    expected = scipy_ndi.minimum_filter(a_np, size=3, mode="constant", cval=np.inf)
    _close(out, expected, atol=1e-5)


def test_minimum_filter_3d(img_3d) -> None:
    a_np, a_t = img_3d
    out = torch_ndi.minimum_filter(a_t, size=3, mode="constant", cval=np.inf)
    expected = scipy_ndi.minimum_filter(a_np, size=3, mode="constant", cval=np.inf)
    _close(out, expected, atol=1e-5)


# -----------------------------------------------------------------------------
# Structural ops (CPU round-trip)
# -----------------------------------------------------------------------------


def test_binary_opening_2d() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    a[5, 5] = False  # carve a hole
    out = torch_ndi.binary_opening(torch.from_numpy(a))
    expected = scipy_ndi.binary_opening(a)
    np.testing.assert_array_equal(_to_numpy(out), expected)


def test_binary_opening_3d() -> None:
    a = np.zeros((6, 6, 6), dtype=bool)
    a[1:5, 1:5, 1:5] = True
    out = torch_ndi.binary_opening(torch.from_numpy(a))
    expected = scipy_ndi.binary_opening(a)
    np.testing.assert_array_equal(_to_numpy(out), expected)


def test_binary_fill_holes_2d() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    a[4:6, 4:6] = False
    out = torch_ndi.binary_fill_holes(torch.from_numpy(a))
    expected = scipy_ndi.binary_fill_holes(a)
    np.testing.assert_array_equal(_to_numpy(out), expected)


def test_label_2d() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[1:3, 1:3] = True
    a[6:8, 6:8] = True
    labels, n = torch_ndi.label(torch.from_numpy(a))
    expected_labels, expected_n = scipy_ndi.label(a)
    np.testing.assert_array_equal(_to_numpy(labels), expected_labels)
    assert n == expected_n


def test_label_3d_with_structure() -> None:
    a = np.zeros((6, 6, 6), dtype=bool)
    a[1:3, 1:3, 1:3] = True
    a[4:6, 4:6, 4:6] = True
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n = torch_ndi.label(
        torch.from_numpy(a), structure=torch.from_numpy(structure)
    )
    expected_labels, expected_n = scipy_ndi.label(a, structure=structure)
    np.testing.assert_array_equal(_to_numpy(labels), expected_labels)
    assert n == expected_n
