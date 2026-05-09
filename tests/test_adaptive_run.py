"""Tests for the MPS-aware extensions to ``nellie.utils.adaptive_run``.

Covers the slice 1 acceptance criteria:

- ``device_cascade(device) → list[str]`` returns the right order per
  platform / requested device.
- ``resolve_backend("mps")`` returns the torch shim tuple (mocked).
- ``resolve_backend("gpu")`` is platform-aware (Mac → MPS, else CUDA).
- ``normalize_device`` accepts ``"mps"``.
- ``is_oom_error`` recognizes the MPS string.
- ``is_gpu_unavailable_error`` recognizes the torch+MPS unavailable patterns.
- ``get_mps_free_bytes`` returns sensible values when probes are mocked.
- ``_MEMORY_HEADROOM_BY_DEVICE`` is 0.7 for CUDA, 0.5 for MPS.
- Cascade callers honor ``device="mps"`` even when the stage is not
  MPS-onboarded (graceful CPU fallback for hierarchical / voxel_reassignment).

All tests run without torch installed (we mock the import path).
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest

from nellie.utils import adaptive_run


# -----------------------------------------------------------------------------
# Helpers — mock torch availability
# -----------------------------------------------------------------------------


@contextmanager
def _mock_torch_with_mps(*, available: bool, built: bool = True):
    """Inject a fake torch module exposing the MPS-availability surface."""
    real_torch = sys.modules.get("torch")
    real_xp = sys.modules.get("nellie.utils.torch_xp")
    real_ndi = sys.modules.get("nellie.utils.torch_ndi")

    fake_torch = types.SimpleNamespace()
    fake_backends = types.SimpleNamespace()
    fake_mps = types.SimpleNamespace(
        is_available=lambda: available,
        is_built=lambda: built,
        driver_allocated_memory=lambda: 0,
    )
    fake_backends.mps = fake_mps
    fake_torch.backends = fake_backends
    fake_torch.mps = fake_mps

    sys.modules["torch"] = fake_torch
    fake_xp = types.SimpleNamespace(__name__="nellie.utils.torch_xp")
    fake_xp_ndi = types.SimpleNamespace(__name__="nellie.utils.torch_ndi")
    sys.modules["nellie.utils.torch_xp"] = fake_xp
    sys.modules["nellie.utils.torch_ndi"] = fake_xp_ndi
    try:
        yield fake_torch
    finally:
        if real_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = real_torch
        if real_xp is None:
            sys.modules.pop("nellie.utils.torch_xp", None)
        else:
            sys.modules["nellie.utils.torch_xp"] = real_xp
        if real_ndi is None:
            sys.modules.pop("nellie.utils.torch_ndi", None)
        else:
            sys.modules["nellie.utils.torch_ndi"] = real_ndi


@contextmanager
def _mock_no_torch():
    """Force ``import torch`` to raise ModuleNotFoundError."""
    real_torch = sys.modules.pop("torch", None)
    real_xp = sys.modules.pop("nellie.utils.torch_xp", None)
    real_ndi = sys.modules.pop("nellie.utils.torch_ndi", None)
    sys.modules["torch"] = None  # forces ImportError on subsequent import
    try:
        yield
    finally:
        sys.modules.pop("torch", None)
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        if real_xp is not None:
            sys.modules["nellie.utils.torch_xp"] = real_xp
        if real_ndi is not None:
            sys.modules["nellie.utils.torch_ndi"] = real_ndi


# -----------------------------------------------------------------------------
# normalize_device
# -----------------------------------------------------------------------------


def test_normalize_device_accepts_mps() -> None:
    assert adaptive_run.normalize_device("mps") == "mps"


def test_normalize_device_accepts_existing_values() -> None:
    assert adaptive_run.normalize_device("auto") == "auto"
    assert adaptive_run.normalize_device("cpu") == "cpu"
    assert adaptive_run.normalize_device("gpu") == "gpu"
    assert adaptive_run.normalize_device("cuda") == "gpu"  # alias


def test_normalize_device_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        adaptive_run.normalize_device("bogus")


def test_normalize_device_default_is_auto() -> None:
    assert adaptive_run.normalize_device(None) == "auto"


# -----------------------------------------------------------------------------
# device_cascade
# -----------------------------------------------------------------------------


def test_device_cascade_cpu() -> None:
    assert adaptive_run.device_cascade("cpu") == ["cpu"]


def test_device_cascade_explicit_mps_pinned() -> None:
    """``device="mps"`` is an explicit pin; no CPU fallback."""
    assert adaptive_run.device_cascade("mps") == ["mps"]


def test_device_cascade_explicit_cuda_pinned() -> None:
    """``device="cuda"`` is an explicit pin to CuPy; no fallback."""
    assert adaptive_run.device_cascade("cuda") == ["gpu"]


def test_device_cascade_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        adaptive_run.device_cascade("bogus")


def test_device_cascade_auto_on_mac_with_mps() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         mock.patch.object(adaptive_run, "mps_available", return_value=True):
        assert adaptive_run.device_cascade("auto") == ["mps", "cpu"]


def test_device_cascade_auto_on_mac_without_mps() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         mock.patch.object(adaptive_run, "mps_available", return_value=False):
        assert adaptive_run.device_cascade("auto") == ["cpu"]


def test_device_cascade_auto_on_linux_with_cuda() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=False), \
         mock.patch.object(adaptive_run, "gpu_available", return_value=True):
        assert adaptive_run.device_cascade("auto") == ["gpu", "cpu"]


def test_device_cascade_auto_on_linux_without_cuda() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=False), \
         mock.patch.object(adaptive_run, "gpu_available", return_value=False):
        assert adaptive_run.device_cascade("auto") == ["cpu"]


def test_device_cascade_gpu_on_mac_is_mps() -> None:
    """``device="gpu"`` is platform-aware: MPS on Darwin."""
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         mock.patch.object(adaptive_run, "mps_available", return_value=True):
        assert adaptive_run.device_cascade("gpu") == ["mps", "cpu"]


def test_device_cascade_gpu_on_linux_is_cuda() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=False), \
         mock.patch.object(adaptive_run, "gpu_available", return_value=True):
        assert adaptive_run.device_cascade("gpu") == ["gpu", "cpu"]


# -----------------------------------------------------------------------------
# resolve_backend (with mocked torch)
# -----------------------------------------------------------------------------


def test_resolve_backend_mps_with_torch_returns_shim_tuple() -> None:
    with _mock_torch_with_mps(available=True):
        xp, ndi, device_type = adaptive_run.resolve_backend("mps")
    assert device_type == "mps"
    assert xp.__name__ == "nellie.utils.torch_xp"
    assert ndi.__name__ == "nellie.utils.torch_ndi"


def test_resolve_backend_mps_without_torch_raises() -> None:
    with _mock_no_torch(), pytest.raises(RuntimeError, match="MPS backend requested"):
        adaptive_run.resolve_backend("mps")


def test_resolve_backend_mps_when_torch_lacks_mps_raises() -> None:
    """torch installed but MPS not built/available → require=True raises."""
    with _mock_torch_with_mps(available=False), pytest.raises(RuntimeError):
        adaptive_run.resolve_backend("mps")


def test_resolve_backend_gpu_on_mac_resolves_to_mps() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         _mock_torch_with_mps(available=True):
        xp, ndi, device_type = adaptive_run.resolve_backend("gpu")
    assert device_type == "mps"


def test_resolve_backend_auto_on_mac_with_mps_resolves_to_mps() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         _mock_torch_with_mps(available=True):
        xp, ndi, device_type = adaptive_run.resolve_backend("auto")
    assert device_type == "mps"


def test_resolve_backend_auto_on_mac_without_mps_falls_back_to_cpu() -> None:
    with mock.patch.object(adaptive_run, "_is_darwin", return_value=True), \
         _mock_no_torch():
        xp, ndi, device_type = adaptive_run.resolve_backend("auto")
    assert device_type == "cpu"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        adaptive_run.resolve_backend("bogus")


# -----------------------------------------------------------------------------
# Memory headroom lookup
# -----------------------------------------------------------------------------


def test_memory_headroom_cuda_is_seven_tenths() -> None:
    assert adaptive_run._MEMORY_HEADROOM_BY_DEVICE["cuda"] == pytest.approx(0.7)


def test_memory_headroom_mps_is_one_half() -> None:
    assert adaptive_run._MEMORY_HEADROOM_BY_DEVICE["mps"] == pytest.approx(0.5)


def test_memory_headroom_lookup_falls_back_for_unknown() -> None:
    """Unknown device names fall back to the legacy CUDA value."""
    assert adaptive_run._headroom("nonexistent_device") == adaptive_run._MEMORY_HEADROOM


def test_memory_headroom_legacy_alias_unchanged() -> None:
    """The legacy scalar should still equal the CUDA value (back-compat)."""
    assert adaptive_run._MEMORY_HEADROOM == pytest.approx(0.7)


# -----------------------------------------------------------------------------
# get_mps_free_bytes
# -----------------------------------------------------------------------------


def test_get_mps_free_bytes_returns_none_without_torch() -> None:
    with _mock_no_torch():
        assert adaptive_run.get_mps_free_bytes() is None


def test_get_mps_free_bytes_subtracts_allocated() -> None:
    with _mock_torch_with_mps(available=True) as fake_torch, \
         mock.patch.object(adaptive_run, "get_cpu_available_bytes", return_value=1_000_000_000):
        fake_torch.mps.driver_allocated_memory = lambda: 250_000_000
        result = adaptive_run.get_mps_free_bytes()
    assert result == 750_000_000


def test_get_mps_free_bytes_returns_none_when_cpu_probe_fails() -> None:
    with _mock_torch_with_mps(available=True), \
         mock.patch.object(adaptive_run, "get_cpu_available_bytes", return_value=None):
        result = adaptive_run.get_mps_free_bytes()
    assert result is None


def test_get_mps_free_bytes_honors_high_watermark_env(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    with _mock_torch_with_mps(available=True) as fake_torch, \
         mock.patch.object(adaptive_run, "get_cpu_available_bytes", return_value=1_000_000_000):
        fake_torch.mps.driver_allocated_memory = lambda: 0
        result = adaptive_run.get_mps_free_bytes()
    assert result == 500_000_000  # capped by watermark


def test_get_mps_free_bytes_floor_is_zero() -> None:
    """When allocated > available, free bytes can't go negative."""
    with _mock_torch_with_mps(available=True) as fake_torch, \
         mock.patch.object(adaptive_run, "get_cpu_available_bytes", return_value=100):
        fake_torch.mps.driver_allocated_memory = lambda: 1000
        result = adaptive_run.get_mps_free_bytes()
    assert result == 0


# -----------------------------------------------------------------------------
# Error classifiers
# -----------------------------------------------------------------------------


def test_is_oom_error_matches_mps_string() -> None:
    exc = RuntimeError("MPS backend out of memory (Total system memory: 16 GB)")
    assert adaptive_run.is_oom_error(exc)


def test_is_oom_error_still_matches_legacy_strings() -> None:
    """Don't regress the existing CuPy / generic patterns."""
    assert adaptive_run.is_oom_error(RuntimeError("Out of memory: allocation"))
    assert adaptive_run.is_oom_error(MemoryError())


def test_is_oom_error_negatives() -> None:
    assert not adaptive_run.is_oom_error(RuntimeError("device side assert"))
    assert not adaptive_run.is_oom_error(ValueError("unrelated"))


def test_is_gpu_unavailable_error_matches_mps_unavailable() -> None:
    """The string raised by ``try_import_torch_mps(require=True)``."""
    exc = RuntimeError(
        "MPS backend requested but no MPS device is available "
        "(torch.backends.mps.is_available() is False)."
    )
    assert adaptive_run.is_gpu_unavailable_error(exc)


def test_is_gpu_unavailable_error_matches_torch_missing() -> None:
    exc = RuntimeError(
        "MPS backend requested but torch is not installed. "
        "Install with `pip install 'nellie[mps]'`."
    )
    assert adaptive_run.is_gpu_unavailable_error(exc)


def test_is_gpu_unavailable_error_matches_module_not_found() -> None:
    exc = ModuleNotFoundError(name="torch")
    assert adaptive_run.is_gpu_unavailable_error(exc)


def test_is_gpu_unavailable_error_still_matches_legacy_strings() -> None:
    """Don't regress the existing CuPy patterns."""
    assert adaptive_run.is_gpu_unavailable_error(
        RuntimeError("GPU backend requested but CuPy is not installed.")
    )
    assert adaptive_run.is_gpu_unavailable_error(ModuleNotFoundError(name="cupy"))


# -----------------------------------------------------------------------------
# free_gpu_memory honors torch_xp
# -----------------------------------------------------------------------------


def test_free_gpu_memory_calls_torch_mps_empty_cache() -> None:
    """When ``xp`` is the torch_xp shim, free_gpu_memory should call torch.mps.empty_cache."""
    with _mock_torch_with_mps(available=True) as fake_torch:
        called = []
        fake_torch.mps.empty_cache = lambda: called.append(True)
        fake_xp = types.SimpleNamespace(__name__="nellie.utils.torch_xp")
        adaptive_run.free_gpu_memory(fake_xp)
    assert called == [True]


def test_free_gpu_memory_noop_on_numpy() -> None:
    import numpy as np
    # Should not raise.
    adaptive_run.free_gpu_memory(np)


# -----------------------------------------------------------------------------
# Cascade-caller refactor: stages handle device="mps" gracefully
# -----------------------------------------------------------------------------


def test_filter_construction_with_device_mps_no_torch_raises_clean_error(make_imageinfo_2d) -> None:
    """User on a Mac without torch installed who passes ``device="mps"``
    explicitly hits the cascade's "GPU unavailable" path. ``device="mps"``
    is an explicit pin (no CPU fallback) — Filter's constructor calls
    ``resolve_backend(device)`` eagerly, so the error surfaces at
    construction. We verify the error is recognized by ``is_gpu_unavailable_error``
    so the cascade in :meth:`Filter.run` would react correctly if reached.
    """
    from nellie.segmentation.filtering import Filter, FrangiConfig

    info = make_imageinfo_2d()
    config = FrangiConfig(device="mps")
    with _mock_no_torch():
        with pytest.raises(RuntimeError) as excinfo:
            Filter(info, config, num_t=2)
        assert adaptive_run.is_gpu_unavailable_error(excinfo.value)


def test_hierarchical_unimplemented_op_classifies_as_unavailable() -> None:
    """A non-onboarded stage hitting ``xp.unimplemented_op`` on MPS should
    raise ``NotImplementedError`` from the shim with a recognizable
    message — that error is then classified as "GPU unavailable" so the
    cascade falls back to CPU rather than crashing the run.
    """
    # Simulate what would happen inside Hierarchy._run_hierarchy if it
    # called an op the torch_xp shim doesn't implement.
    exc = NotImplementedError(
        "torch_xp.cumsum_along_some_axis is not implemented. The MPS shim "
        "only covers the union of ops used by the four onboarded stages..."
    )
    assert adaptive_run.is_gpu_unavailable_error(exc)


def test_unrelated_not_implemented_does_not_classify_as_unavailable() -> None:
    """Avoid false-positive: a NotImplementedError that ISN'T from the shim
    must not be misclassified as a backend-unavailable signal."""
    exc = NotImplementedError("Some unrelated thing isn't ready")
    assert not adaptive_run.is_gpu_unavailable_error(exc)


# -----------------------------------------------------------------------------
# All seven stages: their Configs accept device="mps" without raising.
# -----------------------------------------------------------------------------


def test_all_stage_configs_accept_device_mps() -> None:
    """Every stage Config must accept ``device="mps"`` since this slice
    makes "mps" a canonical device value via ``normalize_device``."""
    from nellie.segmentation.filtering import FrangiConfig
    from nellie.segmentation.labelling import LabelConfig
    from nellie.segmentation.networking import NetworkConfig
    from nellie.segmentation.mocap_marking import MarkersConfig
    from nellie.tracking.hu_tracking import HuMomentTrackingConfig
    from nellie.tracking.voxel_reassignment import VoxelReassignerConfig
    from nellie.feature_extraction.hierarchical import HierarchyConfig

    for cls in (
        FrangiConfig, LabelConfig, NetworkConfig, MarkersConfig,
        HuMomentTrackingConfig, VoxelReassignerConfig, HierarchyConfig,
    ):
        cfg = cls(device="mps")
        assert cfg.device == "mps"
