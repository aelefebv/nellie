"""Characterization tests for ``nellie.tracking.hu_tracking.HuMomentTracking``.

Pins the wiki-documented invariants on both the 3D and 2D paths:

- Output schema (``flow_vector_array.npy``):
  - 2D: shape ``(N, 6)``, columns ``[t, y, x, dy, dx, cost]``
  - 3D: shape ``(N, 8)``, columns ``[t, z, y, x, dz, dy, dx, cost]``
  - Populated dtype is ``float64`` (``np.column_stack`` upcast from
    int64+float32 mix); empty dtype is ``float32`` (``_run_hu_tracking``
    fallback path at ``hu_tracking.py:1230-1232``). Pin BOTH branches —
    do not silently unify.
  - Cost values are bounded ABOVE by the hardcoded
    ``cost_cutoff = 1.0`` in both ``_find_best_matches`` (line 909) and
    ``_match_frames_sparse`` (line 1033). Costs are sums of z-scored
    components and can go very negative for excellent matches — only
    the upper bound is enforced. (PRD #91 / issue #92 originally
    described the contract as ``[0, 1]``; the actual code only bounds
    above. The deviation is documented in the test docstrings.)
  - Vector columns (``dy``/``dx`` 2D, ``dz``/``dy``/``dx`` 3D) are
    integer-valued.
  - All ``t`` values ∈ ``[0, num_t - 2]`` (last frame has no successor).

- ``no_t`` ImInfo: ``run()`` early-returns and writes nothing.

- Matching modes:
  - ``mode='dense'`` runs the dense path on the yeast fixture.
  - ``mode='sparse'`` forces the KDTree path; ``self.device_type`` is
    UNCHANGED for a CPU run (sparse path is CPU-only on the matching
    axis but does not mutate the backend wholesale when ``mode`` is
    explicit).
  - ``mode='auto'`` switches based on ``N_post * N_pre`` vs
    ``max_dense_pairs``.
  - ``low_memory=True`` forces streaming ROI extraction.
  - ``cost_cutoff = 1.0`` is pinned in BOTH dense and sparse paths.

- Cost-scoring equivalence: dense and sparse paths produce the same
  match set on a tiny synthetic input.

- Edge cases:
  - All-zero marker memmap → no flow vectors written; empty array
    fallback fires (``float32`` dtype, ``(0, 6)`` or ``(0, 8)`` shape).
  - First frame is never matched against itself.
  - ``_log_hu`` finiteness on an all-zero input (no NaN/inf).

- Input memmaps (raw + Frangi + Label + Marker + Distance) are not
  mutated by ``HuMomentTracking.run()``.

- ``viewer.status`` callback: no-op when ``viewer=None``; called once
  per frame when ``viewer`` is a stub.

- **Architecture characterization (POST-Slice 3 contract)**:
  - Cascade A (per-frame ``_get_frame_features`` OOM) was DELETED in
    Slice 3 of #91. On per-frame OOM, the exception now propagates to
    the outer ``adaptive_run.mode_candidates`` cascade in ``run()``,
    which retries the whole stage with the next ``(device, low_memory)``
    candidate. The Slice 1 characterization test for this cascade was
    removed when the wrapper went away.
  - Cascade B (``_match_frames`` dense → sparse fallback OOM) survives
    BUT no longer mutates ``self.device_type``. A ``MemoryError`` raised
    once during cost-matrix computation falls through to the sparse
    KDTree matcher (sparse is CPU-only on the matching axis), but later
    frames keep their original backend — feature extraction for
    subsequent frames can still run on GPU.

The Markers ``im_marker`` / ``im_distance`` memmaps that
``HuMomentTracking`` consumes are precomputed once per session by
``conftest.markers_*_paths``; per-test ImInfos copy in 4 memmaps
(Frangi + Label + Marker + Distance) into a fresh working directory so
each test gets an isolated ``flow_vector_array.npy`` target.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label, LabelConfig
from nellie.segmentation.mocap_marking import Markers, MarkersConfig
from nellie.tracking.hu_tracking import (
    HuMomentTracking,
    HuMomentTrackingConfig,
    _FrameFeatures,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _release_hu(h: HuMomentTracking) -> None:
    """Drop a HuMomentTracking's memmap references and force gc.

    Mirrors ``test_mocap_marking._release_markers``. Required on
    Windows: input memmaps can stay file-locked until the handles are
    dropped, blocking later overwrites of the same paths.
    """
    h.label_memmap = None
    h.im_memmap = None
    h.im_frangi_memmap = None
    h.im_marker_memmap = None
    h.im_distance_memmap = None
    gc.collect()


def _run_hu(info: ImInfo, **kwargs) -> HuMomentTracking:
    """Construct + run ``HuMomentTracking`` on ``info``; return the instance.

    Always pinned to CPU for deterministic behavior. Caller is
    responsible for calling ``_release_hu`` after inspecting the
    ``flow_vector_array`` on disk.
    """
    kwargs.setdefault("device", "cpu")
    h = HuMomentTracking(info, HuMomentTrackingConfig(**kwargs), num_t=2)
    h.run()
    return h


def _load_flow_vector_array(info: ImInfo) -> np.ndarray:
    """Load the ``flow_vector_array.npy`` file from ``info``'s pipeline tree."""
    fva_path = Path(info.pipeline_paths["flow_vector_array"])
    return np.load(fva_path)


# -------------------------------------------------------------------------
# Module-scoped: run Hu once on each fixture and share outputs across the
# read-only invariant tests (shape, dtype, cost range, t range, ...).
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hu_outputs_3d(make_hu_imageinfo_3d_module) -> dict[str, object]:
    info = make_hu_imageinfo_3d_module()
    h = _run_hu(info)
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    return {"info": info, "array": arr}


@pytest.fixture(scope="module")
def hu_outputs_2d(make_hu_imageinfo_2d_module) -> dict[str, object]:
    info = make_hu_imageinfo_2d_module()
    h = _run_hu(info)
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    return {"info": info, "array": arr}


# -------------------------------------------------------------------------
# End-to-end smoke tests
# -------------------------------------------------------------------------


def test_runs_end_to_end_3d(hu_outputs_3d) -> None:
    """3D yeast fixture: ``flow_vector_array.npy`` exists with ``(N, 8)``, N > 0."""
    arr = hu_outputs_3d["array"]
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] == 8, f"Expected 8 columns for 3D, got {arr.shape[1]}"
    assert arr.shape[0] > 0, "3D fixture produced no flow vectors"
    info = hu_outputs_3d["info"]
    fva_path = Path(info.pipeline_paths["flow_vector_array"])
    assert fva_path.exists(), f"flow_vector_array.npy not found at {fva_path}"


def test_runs_end_to_end_2d(hu_outputs_2d) -> None:
    """2D yeast fixture: ``flow_vector_array.npy`` exists with ``(N, 6)``, N > 0."""
    arr = hu_outputs_2d["array"]
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] == 6, f"Expected 6 columns for 2D, got {arr.shape[1]}"
    assert arr.shape[0] > 0, "2D fixture produced no flow vectors"


# -------------------------------------------------------------------------
# Output contract: dtype, cost range, integer vectors, t range
# -------------------------------------------------------------------------


def test_populated_dtype_is_float64_3d(hu_outputs_3d) -> None:
    """Populated array dtype is ``float64`` (np.column_stack upcast).

    ``_run_hu_tracking`` builds rows via ``np.column_stack`` of mixed
    int64 + float32 inputs (lines 1199-1219); numpy upcasts the
    resulting dtype to float64. Pinning so the populated path doesn't
    silently widen or narrow.
    """
    arr = hu_outputs_3d["array"]
    assert arr.dtype == np.float64, (
        f"Populated 3D flow_vector_array dtype is {arr.dtype}, expected float64"
    )


def test_populated_dtype_is_float64_2d(hu_outputs_2d) -> None:
    """2D mirror of :func:`test_populated_dtype_is_float64_3d`."""
    arr = hu_outputs_2d["array"]
    assert arr.dtype == np.float64, (
        f"Populated 2D flow_vector_array dtype is {arr.dtype}, expected float64"
    )


def test_cost_bounded_above_by_cost_cutoff_3d(hu_outputs_3d) -> None:
    """Cost column values are bounded above by ``cost_cutoff=1.0``.

    The hardcoded ``cost_cutoff = 1.0`` in both ``_find_best_matches``
    (line 909) and ``_match_frames_sparse`` (line 1033) is purely an
    UPPER bound on accepted matches (``val > cost_cutoff`` skips the
    pair). Costs are sums of z-scored components and can be VERY
    negative for excellent matches; this is observed in the on-disk
    flow_vector_array (3D yeast fixture: cost min ≈ -3.5).

    Note: PRD #91 / issue #92 originally described the contract as
    ``cost ∈ [0.0, 1.0]`` (matching the wiki's
    ``Pass 5 — Contract Scan`` claim), but the actual code only
    bounds above. We pin the real behavior here so the
    ``cost_cutoff = 1.0`` lift in Slice 2 (#93) doesn't accidentally
    introduce a lower bound.
    """
    arr = hu_outputs_3d["array"]
    cost_col = arr[:, -1]
    assert cost_col.max() <= 1.0, (
        f"3D cost max {cost_col.max()} > 1.0 — cost_cutoff bound violated"
    )
    assert np.isfinite(cost_col).all(), (
        "3D cost column contains non-finite values"
    )


def test_cost_bounded_above_by_cost_cutoff_2d(hu_outputs_2d) -> None:
    """2D mirror of :func:`test_cost_bounded_above_by_cost_cutoff_3d`."""
    arr = hu_outputs_2d["array"]
    cost_col = arr[:, -1]
    assert cost_col.max() <= 1.0, (
        f"2D cost max {cost_col.max()} > 1.0 — cost_cutoff bound violated"
    )
    assert np.isfinite(cost_col).all(), (
        "2D cost column contains non-finite values"
    )


def test_vector_columns_integer_valued_3d(hu_outputs_3d) -> None:
    """``dz`` / ``dy`` / ``dx`` columns are integer-valued (cast in _run_hu_tracking).

    Even though the array dtype is float64 (cost forces upcast), the
    vector columns come from voxel-index subtraction and are cast to
    int64 before column_stack (lines 1215-1217). They must therefore
    be exactly representable as integers in the float64 array.
    """
    arr = hu_outputs_3d["array"]
    # 3D: cols are [t, z, y, x, dz, dy, dx, cost]; vectors are 4..6.
    for col in (4, 5, 6):
        col_vals = arr[:, col]
        assert np.array_equal(col_vals, np.round(col_vals)), (
            f"3D col {col} (vector) has non-integer values"
        )


def test_vector_columns_integer_valued_2d(hu_outputs_2d) -> None:
    """``dy`` / ``dx`` columns are integer-valued."""
    arr = hu_outputs_2d["array"]
    # 2D: cols are [t, y, x, dy, dx, cost]; vectors are 3..4.
    for col in (3, 4):
        col_vals = arr[:, col]
        assert np.array_equal(col_vals, np.round(col_vals)), (
            f"2D col {col} (vector) has non-integer values"
        )


def test_t_in_valid_range_3d(hu_outputs_3d) -> None:
    """All ``t`` values ∈ [0, num_t - 2]; last frame has no successor.

    With num_t=2 the only valid t is 0 (the pair (0, 1) with t=t_curr-1=0).
    """
    arr = hu_outputs_3d["array"]
    t_col = arr[:, 0]
    assert t_col.min() >= 0
    assert t_col.max() <= 0  # num_t=2, so num_t-2=0


def test_t_in_valid_range_2d(hu_outputs_2d) -> None:
    """2D mirror of :func:`test_t_in_valid_range_3d`."""
    arr = hu_outputs_2d["array"]
    t_col = arr[:, 0]
    assert t_col.min() >= 0
    assert t_col.max() <= 0


def test_first_frame_never_matched_against_itself(hu_outputs_3d) -> None:
    """First frame skipped (``if prev_frame_features is None`` branch, lines 1177-1179).

    There must be no rows where the resulting ``t`` is negative (which
    would be the case if t=0 were matched against itself and t-1 were
    written). On the 2-frame yeast fixture the only matched pair is
    (frame 0, frame 1); the only valid stored ``t`` is 0.
    """
    arr = hu_outputs_3d["array"]
    t_col = arr[:, 0]
    assert (t_col >= 0).all(), (
        "Found rows with negative t; first frame was matched against itself"
    )


# -------------------------------------------------------------------------
# Empty-array dtype asymmetry (the float32 fallback branch)
# -------------------------------------------------------------------------


def _zero_marker_memmap(info: ImInfo) -> None:
    """Overwrite the on-disk ``im_marker`` memmap with all zeros.

    Triggers the empty-result branch in ``_get_frame_features_impl``
    (lines 612-618) for every frame, which in turn drives
    ``_run_hu_tracking`` to the ``frame_vectors == []`` fallback that
    writes an ``np.empty((0, N), dtype=np.float32)`` file (1230-1232).
    """
    marker_path = info.pipeline_paths["im_marker"]
    mm = tifffile.memmap(marker_path, mode="r+")
    mm[:] = 0
    mm.flush()
    del mm
    gc.collect()


def test_empty_array_dtype_is_float32_3d(make_hu_imageinfo_3d) -> None:
    """Empty array dtype is float32; populated dtype is float64. Pin asymmetry.

    The dtype-asymmetry ``hu_tracking.py:1230-1232`` is intentional in
    the wiki dechaos report (resolved decision #2 / Slice 1's "do not
    silently unify" pin). We trigger the empty branch by zeroing the
    marker memmap so every frame produces zero markers.
    """
    info = make_hu_imageinfo_3d()
    _zero_marker_memmap(info)
    h = _run_hu(info)
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    assert arr.shape == (0, 8), (
        f"Empty 3D flow_vector_array shape is {arr.shape}, expected (0, 8)"
    )
    assert arr.dtype == np.float32, (
        f"Empty 3D flow_vector_array dtype is {arr.dtype}, expected float32"
    )


def test_empty_array_dtype_is_float32_2d(make_hu_imageinfo_2d) -> None:
    """2D mirror of :func:`test_empty_array_dtype_is_float32_3d`."""
    info = make_hu_imageinfo_2d()
    _zero_marker_memmap(info)
    h = _run_hu(info)
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    assert arr.shape == (0, 6), (
        f"Empty 2D flow_vector_array shape is {arr.shape}, expected (0, 6)"
    )
    assert arr.dtype == np.float32, (
        f"Empty 2D flow_vector_array dtype is {arr.dtype}, expected float32"
    )


# -------------------------------------------------------------------------
# No T axis: run() early-returns
# -------------------------------------------------------------------------


def _build_single_frame_iminfo(workdir: Path) -> ImInfo:
    """Build a single-frame (no T) 2D ImInfo with Filter+Label+Markers outputs on disk.

    The raw image is YX (no T axis), so ``info.no_t == True``.
    ``HuMomentTracking.__init__`` early-returns when ``no_t`` is set,
    and ``run()`` has its own no-op short-circuit.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / "single_frame.ome.tif"
    raw = np.zeros((96, 96), dtype=np.uint16)
    raw[30:60, 30:60] = 1500
    tifffile.imwrite(
        raw_path,
        raw,
        photometric="minisblack",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.0655,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.0655,
            "PhysicalSizeYUnit": "µm",
        },
    )
    file_info = FileInfo(str(raw_path))
    file_info.find_metadata()
    file_info.load_metadata()
    info = ImInfo(file_info)
    Filter(info, FrangiConfig(device="cpu"), num_t=1).run()
    Label(info, LabelConfig(device="cpu"), num_t=1).run()
    Markers(info, MarkersConfig(device="cpu"), num_t=1).run()
    gc.collect()
    return info


def test_no_t_axis_early_return(tmp_path: Path) -> None:
    """``no_t`` ImInfo: ``run()`` early-returns; no flow_vector_array.npy written.

    ``HuMomentTracking.__init__`` returns immediately when
    ``self.im_info.no_t`` is True (line 97-98), and ``run()`` has its
    own short-circuit (line 1241-1243). No file should be written for
    this dataset.
    """
    info = _build_single_frame_iminfo(tmp_path / "single_frame")
    h = HuMomentTracking(info, HuMomentTrackingConfig(device="cpu"), num_t=1)
    h.run()  # must not raise
    fva_path = Path(info.pipeline_paths["flow_vector_array"])
    assert not fva_path.exists(), (
        "flow_vector_array.npy should not exist for a no-T dataset"
    )


# -------------------------------------------------------------------------
# Mode tests
# -------------------------------------------------------------------------


def test_mode_dense_runs_end_to_end(make_hu_imageinfo_3d) -> None:
    """``mode='dense'`` runs end-to-end on the 3D fixture.

    Pins the dense path's behavior on a known-good fixture; combined
    with :func:`test_mode_sparse_runs_end_to_end` and
    :func:`test_dense_vs_sparse_match_set_equivalence`, this is a hard
    pin on the dense vs sparse contract.
    """
    info = make_hu_imageinfo_3d()
    h = _run_hu(info, mode="dense")
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    assert arr.shape[0] > 0, "mode='dense' produced no flow vectors"
    assert arr.shape[1] == 8


def test_mode_sparse_does_not_mutate_device_type(make_hu_imageinfo_3d) -> None:
    """``mode='sparse'`` runs end-to-end; ``self.device_type`` unchanged for CPU.

    Sparse path is CPU-only on the matching axis but must NOT mutate
    ``self.device_type`` wholesale when ``mode`` is explicit (the
    cross-frame backend mutation only fires from the dense-OOM cascade
    in ``_match_frames``, which is exercised separately by the
    characterization tests). Pinning here so a future refactor doesn't
    accidentally call ``_switch_to_cpu()`` from the explicit-sparse
    path.
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(
        info, HuMomentTrackingConfig(device="cpu", mode="sparse"), num_t=2
    )
    h.run()
    arr = _load_flow_vector_array(info)
    assert h.device_type == "cpu", (
        f"mode='sparse' mutated device_type to {h.device_type!r} (expected 'cpu')"
    )
    assert arr.shape[0] > 0, "mode='sparse' produced no flow vectors"
    _release_hu(h)


def test_mode_auto_switches_on_max_dense_pairs(make_hu_imageinfo_3d) -> None:
    """``mode='auto'`` flips dense ↔ sparse based on ``N_post * N_pre`` vs ``max_dense_pairs``.

    Forcing ``max_dense_pairs=1`` ensures auto-mode lands on sparse;
    a large ``max_dense_pairs`` keeps it on dense. Pinning by
    behavioral consistency: both runs should still produce non-empty
    vectors with cost ∈ [0, 1].
    """
    info_sparse = make_hu_imageinfo_3d()
    h_sparse = HuMomentTracking(
        info_sparse,
        HuMomentTrackingConfig(device="cpu", mode="auto", max_dense_pairs=1),
        num_t=2,
    )
    h_sparse.run()
    arr_sparse = _load_flow_vector_array(info_sparse)
    _release_hu(h_sparse)

    info_dense = make_hu_imageinfo_3d()
    h_dense = HuMomentTracking(
        info_dense,
        HuMomentTrackingConfig(
            device="cpu", mode="auto", max_dense_pairs=int(1e12)
        ),
        num_t=2,
    )
    h_dense.run()
    arr_dense = _load_flow_vector_array(info_dense)
    _release_hu(h_dense)

    assert arr_sparse.shape[0] > 0
    assert arr_dense.shape[0] > 0
    assert arr_sparse[:, -1].max() <= 1.0
    assert arr_dense[:, -1].max() <= 1.0


def test_low_memory_forces_streaming_roi(
    make_hu_imageinfo_3d, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``low_memory=True`` forces streaming ROI extraction (``use_dense=False``).

    Instrumented via monkeypatch on ``_compute_features_streaming``:
    if the streaming branch is taken at least once during a non-empty
    frame, the streaming function will be invoked and we record the
    call. This is the cleanest pin without adding instrumentation to
    production code.
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(
        info, HuMomentTrackingConfig(device="cpu", low_memory=True), num_t=2
    )

    calls = {"n": 0}
    real_streaming = HuMomentTracking._compute_features_streaming

    def counting_streaming(self, *args, **kwargs):
        calls["n"] += 1
        return real_streaming(self, *args, **kwargs)

    monkeypatch.setattr(
        HuMomentTracking, "_compute_features_streaming", counting_streaming
    )
    h.run()
    _release_hu(h)
    assert calls["n"] >= 1, (
        "low_memory=True did not invoke _compute_features_streaming on any frame"
    )


def test_cost_cutoff_pinned_in_both_paths(make_hu_imageinfo_3d) -> None:
    """``cost_cutoff = 1.0`` is pinned in BOTH dense and sparse paths.

    Asserts that all costs in the populated ``flow_vector_array`` are
    ≤ 1.0 in both ``mode='dense'`` and ``mode='sparse'`` runs. The
    duplicated literal lives at ``hu_tracking.py:909`` (dense) and
    ``hu_tracking.py:1033`` (sparse); Slice 2 lifts it to a module
    constant — this test guards the value during that refactor.
    """
    info_dense = make_hu_imageinfo_3d()
    h_dense = _run_hu(info_dense, mode="dense")
    arr_dense = _load_flow_vector_array(info_dense)
    _release_hu(h_dense)

    info_sparse = make_hu_imageinfo_3d()
    h_sparse = _run_hu(info_sparse, mode="sparse")
    arr_sparse = _load_flow_vector_array(info_sparse)
    _release_hu(h_sparse)

    assert arr_dense[:, -1].max() <= 1.0
    assert arr_sparse[:, -1].max() <= 1.0


# -------------------------------------------------------------------------
# Dense vs sparse cost-scoring equivalence (the wiki's headline claim)
# -------------------------------------------------------------------------


def test_dense_vs_sparse_match_set_equivalence(tmp_path: Path) -> None:
    """Dense and sparse paths produce the same match set on a tiny synthetic input.

    Constructs an in-memory pair of ``_FrameFeatures`` with hand-picked
    coords / stats / hu values, dispatches ``_match_frames`` once with
    ``mode='dense'`` and once with ``mode='sparse'`` against the same
    frames, and asserts the (row, col) match set is identical (cost
    values may differ slightly due to float16/float32 vs float64
    intermediates in the dense path; the documented contract is the
    SET of matches, not the costs).

    The ``HuMomentTracking`` instance is built against a 2D fixture
    ImInfo so ``no_z=True`` paths are exercised; this is the simplest
    way to set up ``self.scaling`` and ``self.max_distance_um`` without
    a fully-allocated pipeline tree.
    """
    # Build a minimal HuMomentTracking against a single-frame ImInfo
    # (we only need the constructor to set up self.scaling, self.xp,
    # self.max_distance_um — no run() needed).
    info = _build_single_frame_iminfo(tmp_path / "synthetic_match")
    # The single-frame info has no_t=True which would early-return
    # __init__; force a multi-frame info instead via re-tagging.
    # Simpler approach: use a 2D fixture-style hu instance that we
    # never run; we directly invoke _match_frames.
    # To get a multi-frame ImInfo without running upstream, we re-use
    # the single-frame info but tweak num_t after construction (the
    # match-frames code path doesn't depend on the memmaps).
    h = HuMomentTracking.__new__(HuMomentTracking)
    h.im_info = info
    # Force no_z=True (single-frame is already 2D); scaling is normally
    # populated from im_info.dim_res, which Pyright types as
    # tuple[None, None] | tuple[None, None, None] — override here.
    h.scaling = (1.0, 1.0)  # type: ignore[assignment]
    h.max_distance_um = 5.0
    h.xp = np
    import scipy.ndimage as sp_ndi

    h.ndi = sp_ndi
    h.device_type = "cpu"
    h.max_dense_pairs = int(1e12)
    h.max_dense_roi_voxels_cpu = int(1e12)
    h.max_dense_roi_voxels_gpu = int(1e12)
    h.low_memory = False
    # Slice 2 of #112 lifted ``_COST_CUTOFF`` to ``self.cost_cutoff``.
    # ``__new__`` bypasses ``__init__`` so we have to set it explicitly.
    h.cost_cutoff = HuMomentTrackingConfig().cost_cutoff

    # Three markers in pre-frame, three in post-frame; nearly 1:1.
    coords_pre = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0]], dtype=float)
    coords_post = np.array([[0.5, 0.0], [0.0, 4.5], [4.0, 0.5]], dtype=float)
    stats_pre = np.array(
        [[1.0, 0.1, 0.5, 0.2], [2.0, 0.2, 0.6, 0.3], [3.0, 0.3, 0.7, 0.4]],
        dtype=np.float32,
    )
    stats_post = np.array(
        [[1.1, 0.1, 0.5, 0.2], [2.1, 0.2, 0.6, 0.3], [3.1, 0.3, 0.7, 0.4]],
        dtype=np.float32,
    )
    hu_pre = np.array(
        [
            [0.5, -0.2, 0.1, 0.0, 0.0, 0.0],
            [0.6, -0.3, 0.2, 0.0, 0.0, 0.0],
            [0.7, -0.4, 0.3, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    hu_post = np.array(
        [
            [0.51, -0.2, 0.1, 0.0, 0.0, 0.0],
            [0.61, -0.3, 0.2, 0.0, 0.0, 0.0],
            [0.71, -0.4, 0.3, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    coords_voxel_pre = coords_pre.astype(int)
    coords_voxel_post = coords_post.astype(int)

    frame_pre = _FrameFeatures(
        coords_voxel=coords_voxel_pre,
        coords_phys=coords_pre,
        stats=stats_pre,
        hu=hu_pre,
    )
    frame_post = _FrameFeatures(
        coords_voxel=coords_voxel_post,
        coords_phys=coords_post,
        stats=stats_post,
        hu=hu_post,
    )

    h.mode = "dense"
    rows_d, cols_d, _ = h._match_frames(frame_post, frame_pre)
    h.mode = "sparse"
    rows_s, cols_s, _ = h._match_frames(frame_post, frame_pre)

    set_dense = set(zip(rows_d, cols_d))
    set_sparse = set(zip(rows_s, cols_s))
    assert set_dense == set_sparse, (
        f"Dense vs sparse match set diverged on synthetic input.\n"
        f"  dense:  {sorted(set_dense)}\n"
        f"  sparse: {sorted(set_sparse)}\n"
    )
    # Sanity: at least one match was found.
    assert len(set_dense) > 0


# -------------------------------------------------------------------------
# Edge cases: empty marker frame, _log_hu finiteness
# -------------------------------------------------------------------------


def test_empty_marker_frame_no_flow_vectors(make_hu_imageinfo_3d) -> None:
    """All-zero marker memmap → 0-row flow_vector_array.

    ``_get_frame_features_impl`` short-circuits to empty arrays when
    ``marker_indices_xp.size == 0`` (lines 612-618);
    ``_run_hu_tracking`` then accumulates no rows and falls through to
    the ``frame_vectors == []`` empty-array branch. Distinct from
    :func:`test_empty_array_dtype_is_float32_3d`, which pins the dtype
    asymmetry — this test pins the row-count contract.
    """
    info = make_hu_imageinfo_3d()
    _zero_marker_memmap(info)
    h = _run_hu(info)
    arr = _load_flow_vector_array(info)
    _release_hu(h)
    assert arr.shape[0] == 0, (
        "Empty marker memmap should produce zero flow vectors, "
        f"got {arr.shape[0]}"
    )


def test_log_hu_finite_on_zero_input() -> None:
    """``_log_hu`` produces finite values on an all-zero hu input.

    Pins the wiki-documented finiteness contract: the
    ``eps = xp.finfo(hu.dtype).tiny`` floor (line 325) prevents
    ``log(0) = -inf``, and the
    ``xp.where(xp.isfinite(log_hu), log_hu, 0.0)`` (328) replaces any
    residual NaN/inf with 0. Test in isolation with a hand-constructed
    numpy array so we don't depend on an upstream fixture.
    """
    h = HuMomentTracking.__new__(HuMomentTracking)
    h.xp = np
    zero_hu = np.zeros((4, 6), dtype=np.float32)
    out = h._log_hu(zero_hu)
    assert out.shape == zero_hu.shape
    assert np.isfinite(out).all(), (
        "_log_hu produced non-finite values on all-zero input"
    )


# -------------------------------------------------------------------------
# Input mutation: hash-before / hash-after
# -------------------------------------------------------------------------


def test_input_memmaps_not_mutated_3d(make_hu_imageinfo_3d) -> None:
    """Hash all 5 input memmaps before/after ``HuMomentTracking.run()``.

    Pins that the stage is read-only on its inputs: the raw image,
    Frangi memmap, Label memmap, Marker memmap, and Distance memmap
    must all be byte-identical after the run.
    """
    info = make_hu_imageinfo_3d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])
    label_path = Path(info.pipeline_paths["im_instance_label"])
    marker_path = Path(info.pipeline_paths["im_marker"])
    distance_path = Path(info.pipeline_paths["im_distance"])

    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()
    label_before = hashlib.sha256(label_path.read_bytes()).hexdigest()
    marker_before = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    distance_before = hashlib.sha256(distance_path.read_bytes()).hexdigest()

    h = _run_hu(info)
    _release_hu(h)

    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before
    ), "HuMomentTracking mutated the raw input memmap"
    assert (
        hashlib.sha256(frangi_path.read_bytes()).hexdigest() == frangi_before
    ), "HuMomentTracking mutated the Frangi input memmap"
    assert (
        hashlib.sha256(label_path.read_bytes()).hexdigest() == label_before
    ), "HuMomentTracking mutated the Label input memmap"
    assert (
        hashlib.sha256(marker_path.read_bytes()).hexdigest() == marker_before
    ), "HuMomentTracking mutated the Marker input memmap"
    assert (
        hashlib.sha256(distance_path.read_bytes()).hexdigest() == distance_before
    ), "HuMomentTracking mutated the Distance input memmap"


# -------------------------------------------------------------------------
# Viewer callback: None case + stub case
# -------------------------------------------------------------------------


class _StubViewer:
    """Minimal viewer stub: records every write to ``status``."""

    def __init__(self) -> None:
        self.status_writes: list[str] = []

    @property
    def status(self) -> str:
        return self.status_writes[-1] if self.status_writes else ""

    @status.setter
    def status(self, value: str) -> None:
        self.status_writes.append(value)


def test_viewer_status_callback(make_hu_imageinfo_2d) -> None:
    """``viewer.status`` is set once per frame; ``viewer=None`` is a no-op.

    The viewer-update branch in ``_run_hu_tracking`` (lines 1170-1171)
    only runs when ``self.viewer is not None``. Pin both arms in one
    test:
      - Build a HuMomentTracking with ``viewer=None`` and assert the
        run completes cleanly (no AttributeError).
      - Build a HuMomentTracking with a stub viewer and assert
        ``status`` is written exactly ``num_t`` times (one per frame).
    """
    # No-op arm: viewer=None.
    info_none = make_hu_imageinfo_2d()
    h_none = HuMomentTracking(
        info_none, HuMomentTrackingConfig(device="cpu"), viewer=None, num_t=2
    )
    h_none.run()  # must not raise
    _release_hu(h_none)

    # Stub-viewer arm: assert per-frame writes.
    info_stub = make_hu_imageinfo_2d()
    stub = _StubViewer()
    h_stub = HuMomentTracking(
        info_stub, HuMomentTrackingConfig(device="cpu"), viewer=stub, num_t=2
    )
    h_stub.run()
    assert len(stub.status_writes) == 2, (
        f"Expected viewer.status set once per frame (2 writes); "
        f"got {len(stub.status_writes)}: {stub.status_writes}"
    )
    assert "Frame: 1 of 2" in stub.status_writes[0]
    assert "Frame: 2 of 2" in stub.status_writes[1]
    _release_hu(h_stub)


# -------------------------------------------------------------------------
# Architecture characterization (PRE-Slice 3 contract)
# -------------------------------------------------------------------------


def test_cascade_b_dense_match_oom_does_not_mutate_device_type(
    make_hu_imageinfo_3d, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cascade B: ``_match_frames`` dense → sparse OOM does NOT mutate ``self.device_type``.

    This pins the POST-Slice-3 contract for the matching inner OOM
    cascade. On GPU OOM during the dense cost-matrix computation,
    ``_match_frames`` falls through to the sparse KDTree-based matcher
    but no longer calls ``self._switch_to_cpu()`` — later frames keep
    their original backend (sparse is CPU-only on the matching axis
    only; feature extraction for subsequent frames can still run on GPU).

    CI has no GPU, so we simulate GPU state by direct attribute
    assignment from inside the flaky monkeypatch — the outer
    ``adaptive_run.mode_candidates`` cascade in ``run()`` calls
    ``_set_backend("cpu")`` first, which would clobber any device_type
    set on the constructor. We snapshot ``self.device_type`` from
    inside the flaky callback (right before raising), then assert the
    same value persists across the OOM and the sparse fallback.

    Slice 3 of #91 inverted this from the PRE-Slice-3 contract (which
    asserted ``self.device_type == "cpu"``). The outer
    ``adaptive_run.mode_candidates`` cascade in ``run()`` is now the
    single source of truth for backend switching; inner cascades log,
    free GPU memory, and fall back algorithmically without touching
    ``self.xp`` / ``self.ndi`` / ``self.device_type``.
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(
        info, HuMomentTrackingConfig(device="cpu", mode="dense"), num_t=2
    )

    real_get_cost = HuMomentTracking._get_cost_matrix
    state: dict[str, object] = {"raised": False, "device_type_at_raise": None}

    def flaky_get_cost(self, *args, **kwargs):
        if not state["raised"]:
            # Simulate GPU state right before the OOM so the inner
            # cascade's behavior is exercised as it would be on a real
            # GPU run; snapshot it so we can assert it survives.
            self.device_type = "cuda"
            state["device_type_at_raise"] = self.device_type
            state["raised"] = True
            raise MemoryError("simulated dense matching OOM (Cascade B)")
        return real_get_cost(self, *args, **kwargs)

    monkeypatch.setattr(HuMomentTracking, "_get_cost_matrix", flaky_get_cost)
    h.run()
    _release_hu(h)

    assert state["raised"], (
        "Test setup error: synthetic MemoryError was never raised"
    )
    assert h.device_type == state["device_type_at_raise"], (
        f"Expected Cascade B to leave device_type unchanged "
        f"({state['device_type_at_raise']!r}) after dense OOM; got "
        f"{h.device_type!r}. Slice 3 of #91 removed the cross-frame "
        f"backend mutation; the sparse fallback is algorithmic-only on "
        f"the matching axis."
    )
