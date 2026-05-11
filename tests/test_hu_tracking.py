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

from nellie.im_info import ImInfo, load_image
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
    info = load_image(raw_path)
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
# Direct synthetic tests for `_calculate_normalized_moments`
#
# Pin the contract of the moment math kernel (the most arithmetic-heavy
# function in the per-frame Hu pipeline). PRD #191 / Slice 2 (#193) will
# rewrite this body from a (N, H, W, 4, 4) broadcast to a two-step
# batched matmul; these tests are the regression bar (count / shape /
# allclose-with-tolerance / analytical-value), platform-stable in shape
# and tolerant of float32 precision drift per ADR 0008.
# -------------------------------------------------------------------------


def _make_bare_hu_for_moments() -> HuMomentTracking:
    """Build a HuMomentTracking instance bypassing __init__ for moment-math tests.

    `_calculate_normalized_moments` only reads `self.xp`. The
    `__new__` + manual attribute pattern mirrors
    `test_log_hu_finite_on_zero_input` above and avoids the full
    constructor's ImInfo / memmap setup.
    """
    h = HuMomentTracking.__new__(HuMomentTracking)
    h.xp = np
    return h


def test_calculate_normalized_moments_shape() -> None:
    """`(N, H, W)` input returns ``(N, 4, 4)`` of a float dtype.

    Pins the (N, 4, 4) eta shape contract that downstream
    ``_calculate_hu_moments`` consumes. Doesn't pin the exact float
    dtype (float32 vs float64): numpy's promotion rules between
    ``int64`` (from ``arange`` in the broadcast formulation) and
    ``float32`` differ across numpy 1.x and 2.x; the matmul rewrite
    in Slice 2 (#193) keeps the input dtype throughout. As long as
    the kind is float, downstream is happy.
    """
    h = _make_bare_hu_for_moments()
    rng = np.random.default_rng(7)
    images = rng.uniform(0.0, 1.0, size=(3, 7, 9)).astype(np.float32)
    out = h._calculate_normalized_moments(images)
    assert out.shape == (3, 4, 4)
    assert out.dtype.kind == "f", f"Expected float dtype, got {out.dtype}"


def test_calculate_normalized_moments_empty_input() -> None:
    """Empty ``(0, H, W)`` input returns ``(0, 4, 4)`` cleanly.

    Pins the boundary behavior on zero-marker frames — the per-frame
    early-out at ``hu_tracking.py:584`` already guards
    ``_calculate_normalized_moments`` from being called on (0, H, W),
    but the kernel itself must not crash if the guard is ever bypassed
    (defense in depth + the matmul reformulation in Slice 2 (#193)
    needs to handle empty leading dims uniformly).
    """
    h = _make_bare_hu_for_moments()
    images = np.zeros((0, 7, 9), dtype=np.float32)
    out = h._calculate_normalized_moments(images)
    assert out.shape == (0, 4, 4)


def test_calculate_normalized_moments_all_zeros_finite() -> None:
    """All-zero image input returns finite eta (no NaN/inf).

    Pins the divide-by-zero guards: the ``+ 1e-12`` floor in both the
    centroid division (``M[:, 0, 0] + 1e-12``) and the denom
    calculation. Slice 2 (#193) keeps the same epsilon-floor pattern;
    this test ensures the rewrite preserves the no-NaN contract.
    """
    h = _make_bare_hu_for_moments()
    images = np.zeros((4, 7, 9), dtype=np.float32)
    out = h._calculate_normalized_moments(images)
    assert out.shape == (4, 4, 4)
    assert np.isfinite(out).all(), "All-zero input produced non-finite eta"


def test_calculate_normalized_moments_single_pixel_analytical() -> None:
    """Single non-zero pixel: eta_{0,0} = 1, all other eta = 0.

    For an image with a single mass m at position (h0, w0):

        M_{p,q}    = sum_{h,w} I[h,w] * w^p * h^q = m * w0^p * h0^q
        x_bar      = M_{1,0} / M_{0,0} = w0
        y_bar      = M_{0,1} / M_{0,0} = h0
        mu_{p,q}   = m * (w0 - x_bar)^p * (h0 - y_bar)^q
                   = m * 0^p * 0^q
                   = m  if (p, q) == (0, 0) else 0
        denom_{p,q} = M_{0,0}^((p+q+2)/2) = m^((p+q+2)/2)
        eta_{p,q}  = mu_{p,q} / denom_{p,q}
                   = m / m^1   if (p, q) == (0, 0)  → 1
                   = 0 / m^k   else                  → 0

    Pinning this analytical result locks the centroid math
    (M_{1,0}/M_{0,0} and M_{0,1}/M_{0,0}) AND the denom exponent
    formula ``(p+q+2)/2`` together. If the matmul rewrite (Slice 2
    #193) accidentally swaps p/q axes or drops the +2 in the denom,
    this test fails immediately.
    """
    h = _make_bare_hu_for_moments()
    images = np.zeros((1, 7, 9), dtype=np.float32)
    h0, w0 = 3, 4
    mass = 5.0
    images[0, h0, w0] = mass

    out = h._calculate_normalized_moments(images)

    expected = np.zeros((1, 4, 4), dtype=out.dtype)
    expected[0, 0, 0] = 1.0
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


def _broadcast_normalized_moments_reference(images: np.ndarray) -> np.ndarray:
    """Reference broadcast implementation of `_calculate_normalized_moments`.

    Verbatim copy of the pre-PRD #191 body (the (N, H, W, 4, 4) broadcast
    formulation). Lives inline in the test file so the equivalence pin in
    :func:`test_calculate_normalized_moments_matmul_matches_broadcast_3d`
    survives even after the production code drops the broadcast version.
    Pure numpy — independent of HuMomentTracking instance state.
    """
    extended_images = images[:, :, :, None, None]
    height, width = images.shape[1], images.shape[2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x[None, :, :, None, None]
    y = y[None, :, :, None, None]
    powers = np.arange(4)
    powers_x = powers[None, None, None, :, None]
    powers_y = powers[None, None, None, None, :]
    M = np.sum(extended_images * (x ** powers_x) * (y ** powers_y), axis=(1, 2))
    x_bar = (M[:, 1, 0] / (M[:, 0, 0] + 1e-12))[:, None, None, None, None]
    y_bar = (M[:, 0, 1] / (M[:, 0, 0] + 1e-12))[:, None, None, None, None]
    mu = np.sum(
        extended_images
        * (x - x_bar) ** powers_x
        * (y - y_bar) ** powers_y,
        axis=(1, 2),
    )
    i_plus_j = np.arange(4)[:, None] + np.arange(4)[None, :]
    denom = (M[:, 0, 0][:, None, None] ** ((i_plus_j[None, :, :] + 2) / 2.0)) + 1e-12
    return mu / denom


def test_calculate_normalized_moments_matmul_matches_broadcast_3d(make_hu_imageinfo_3d) -> None:
    """Matmul output ≈ broadcast reference on realistic 3D orthogonal projections.

    Drives the **realistic** input the production code sees: build a
    HuMomentTracking against the yeast 3D fixture, capture
    ``intensity_sub_volumes`` for ``t=0`` post-``_get_sub_volumes``, run
    ``_get_orthogonal_projections`` to get the three (N, H, W) projections,
    then assert the matmul implementation matches the inline broadcast
    reference at ``rtol=1e-5, atol=1e-7`` for each projection.

    Per ADR 0008, BLAS reorders the spatial-axis reduction so the matmul
    output is not bit-equal to the broadcast version. ``rtol=1e-5`` is the
    bound on the relative drift at float32 precision on realistic Hu
    moment magnitudes — same intra-platform precedent as the equivalence
    tests in PRDs #173 / #184. Both paths run in the same process so the
    BLAS implementation is constant (no cross-platform variance to
    confound the assertion).
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(info, HuMomentTrackingConfig(device="cpu"), num_t=2)
    h._allocate_memory()

    # Drive the per-frame ROI extraction to obtain realistic sub-volumes.
    # `_get_frame_features` does the dense ROI extraction internally; we
    # peek at the same intermediates by replaying the relevant fragment
    # of `_get_frame_features` for t=0.
    t = 0
    intensity_frame = np.asarray(h.im_memmap[t]).astype(np.float32, copy=False)
    distance_frame = np.asarray(h.im_distance_memmap[t])
    distance_max_frame = distance_frame.copy()
    h.ndi.maximum_filter(distance_max_frame, size=3, output=distance_max_frame)
    distance_max_frame *= 2

    marker_frame = np.asarray(h.im_marker_memmap[t]) > 0
    marker_indices = np.argwhere(marker_frame)

    # Skip the test gracefully if the fixture happens to have no markers
    # in the first frame. (The synthetic suite already pins the
    # zero-marker behavior; this test depends on having realistic data.)
    if marker_indices.shape[0] == 0:
        _release_hu(h)
        pytest.skip("3D fixture frame 0 has no markers; equivalence test needs realistic input.")

    region_bounds = h._get_im_bounds(marker_indices, distance_max_frame)
    marker_mask = marker_frame
    max_radius = int(np.ceil(np.max(distance_max_frame[marker_mask])).item()) * 2 + 1
    intensity_sub_volumes = h._get_sub_volumes(intensity_frame, region_bounds, max_radius)

    z_proj, y_proj, x_proj = h._get_orthogonal_projections(intensity_sub_volumes)

    for label, projection in (("z", z_proj), ("y", y_proj), ("x", x_proj)):
        out_matmul = h._calculate_normalized_moments(projection)
        out_ref = _broadcast_normalized_moments_reference(projection)
        np.testing.assert_allclose(
            out_matmul,
            out_ref,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"matmul/broadcast disagreement on {label}-projection (rtol=1e-5)",
        )

    _release_hu(h)


def test_calculate_normalized_moments_translation_invariance() -> None:
    """eta is translation-invariant: shifting a blob preserves eta.

    Central moments mu_{p,q} are translation-invariant by construction
    (the centroid x_bar, y_bar shifts with the blob). Normalized
    moments eta = mu / M_{0,0}^k inherit this — same blob at different
    positions in the same-sized image must yield the same eta.

    This is the strongest mathematical pin on the centroid+central-
    moment computation: if the rewrite's centered coordinate tables
    (``x_centered = x_arr - x_bar``) get the broadcasting shape wrong,
    or if the centroids are computed against the wrong axis, this
    test fails.
    """
    h = _make_bare_hu_for_moments()
    blob = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    images_a = np.zeros((1, 11, 11), dtype=np.float32)
    images_b = np.zeros((1, 11, 11), dtype=np.float32)
    images_a[0, 1:4, 1:4] = blob
    images_b[0, 6:9, 6:9] = blob

    out_a = h._calculate_normalized_moments(images_a)
    out_b = h._calculate_normalized_moments(images_b)
    np.testing.assert_allclose(out_a, out_b, rtol=1e-5, atol=1e-7)


# -------------------------------------------------------------------------
# Direct synthetic tests for `_get_cost_matrix`
#
# Pin the contract of the per-frame matching cost-matrix builder. PRD #196
# / Slice 2 (#198) will refactor this body from broadcast (N, N, F) tensors
# at float64 to per-feature streaming at float32; these tests are the
# regression bar (shape / dtype / mask / single-pair) — platform-stable in
# shape and tolerant of float32 precision drift per ADR 0009.
# -------------------------------------------------------------------------


def _make_bare_hu_for_cost_matrix(max_distance_um: float = 1.0) -> HuMomentTracking:
    """Build a HuMomentTracking instance bypassing __init__ for cost-matrix tests.

    `_get_cost_matrix` reads ``self.xp``, ``self.max_distance_um`` (via
    ``_get_distance_mask``), and ``self.device_type`` (also via
    ``_get_distance_mask``, branching on ``"cuda"``/``"mps"`` for on-GPU
    pairwise math vs the CPU ``cdist``). Mirror of
    `_make_bare_hu_for_moments`.
    """
    h = HuMomentTracking.__new__(HuMomentTracking)
    h.xp = np
    h.max_distance_um = max_distance_um
    h.device_type = "cpu"
    return h


def test_get_cost_matrix_shape() -> None:
    """`_get_cost_matrix` returns ``(N_post, N_pre)`` of float32.

    Pins the cost-matrix output shape contract for downstream
    ``_find_best_matches``. Slice 2 (#198) keeps the shape contract;
    only the internal computation changes.
    """
    h = _make_bare_hu_for_cost_matrix(max_distance_um=10.0)
    rng = np.random.default_rng(29)
    coords_post = rng.uniform(0.0, 5.0, size=(3, 3)).astype(np.float64)
    coords_pre = rng.uniform(0.0, 5.0, size=(4, 3)).astype(np.float64)
    stats_post = rng.normal(0.0, 1.0, size=(3, 4)).astype(np.float32)
    stats_pre = rng.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
    hu_post = rng.normal(0.0, 1.0, size=(3, 6)).astype(np.float32)
    hu_pre = rng.normal(0.0, 1.0, size=(4, 6)).astype(np.float32)

    cost = h._get_cost_matrix(coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre)
    assert cost.shape == (3, 4)
    assert cost.dtype == np.float32, f"Expected float32 cost matrix, got {cost.dtype}"


def test_get_cost_matrix_empty_inputs_short_circuit() -> None:
    """Empty stats/hu inputs short-circuit to ``(0, 0)`` float16.

    Pins the early-out at ``hu_tracking.py:889-895``: if any of the four
    feature matrices has zero size, the function returns
    ``xp.zeros((0, 0), dtype=xp.float16)`` without touching the
    distance/feature math. Slice 2 (#198) preserves this short-circuit.
    """
    h = _make_bare_hu_for_cost_matrix()
    coords = np.zeros((0, 3), dtype=np.float64)
    empty_f = np.zeros((0, 4), dtype=np.float32)
    empty_h = np.zeros((0, 6), dtype=np.float32)

    cost = h._get_cost_matrix(coords, coords, empty_f, empty_f, empty_h, empty_h)
    assert cost.shape == (0, 0)
    assert cost.dtype == np.float16


def test_get_cost_matrix_all_masked_returns_inf() -> None:
    """All pairs > max_distance_um apart → cost matrix is all `+inf`.

    The early-out in ``_zscore_normalize`` at ``hu_tracking.py:856`` —
    ``if float(sum_mask) == 0.0: return xp.full_like(m, xp.inf)`` —
    propagates through the nansum to produce an all-`+inf` cost matrix.
    Pinned because Slice 2 (#198)'s streaming refactor needs to handle
    the all-masked path identically (the equivalent in the streaming
    version is ``cost = xp.where(mask, ..., xp.inf)`` with mask all
    False → all inf).
    """
    h = _make_bare_hu_for_cost_matrix(max_distance_um=1.0)
    # Two coords at (0, 0, 0) and (100, 100, 100); pre at (200, 200, 200)
    # and (300, 300, 300). All pairwise distances >> max_distance_um.
    coords_post = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    coords_pre = np.array([[200.0, 200.0, 200.0], [300.0, 300.0, 300.0]])
    rng = np.random.default_rng(31)
    stats_post = rng.normal(0.0, 1.0, size=(2, 4)).astype(np.float32)
    stats_pre = rng.normal(0.0, 1.0, size=(2, 4)).astype(np.float32)
    hu_post = rng.normal(0.0, 1.0, size=(2, 6)).astype(np.float32)
    hu_pre = rng.normal(0.0, 1.0, size=(2, 6)).astype(np.float32)

    cost = h._get_cost_matrix(coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre)
    assert cost.shape == (2, 2)
    assert np.isinf(cost).all(), f"Expected all `+inf` cost; got {cost}"
    assert (cost > 0).all(), f"Expected `+inf` (positive); got negative inf in {cost}"


def _broadcast_cost_matrix_reference(
    h: HuMomentTracking,
    coords_post_phys,
    coords_pre_phys,
    stats_vecs,
    pre_stats_vecs,
    hu_vecs,
    pre_hu_vecs,
) -> np.ndarray:
    """Reference broadcast implementation of `_get_cost_matrix`.

    Verbatim copy of the pre-PRD #196 logic: float64 promotion + (N, N, F)
    tensors via the (now-deleted) ``_get_difference_matrix`` +
    ``_zscore_normalize`` + nansum + float16 cast. Lives inline in the
    test file so the equivalence pin in
    :func:`test_get_cost_matrix_streaming_matches_broadcast_3d` survives
    after Slice 2 (#198) drops the helpers.
    """
    xp = h.xp
    if (
        int(np.prod(stats_vecs.shape)) == 0
        or int(np.prod(pre_stats_vecs.shape)) == 0
        or int(np.prod(hu_vecs.shape)) == 0
        or int(np.prod(pre_hu_vecs.shape)) == 0
    ):
        return xp.zeros((0, 0), dtype=xp.float16)

    distance_matrix, distance_mask = h._get_distance_mask(coords_post_phys, coords_pre_phys)

    def _diff_matrix(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        m1_reshaped = m1[:, xp.newaxis, :].astype(xp.float64)
        m2_reshaped = m2[xp.newaxis, :, :].astype(xp.float64)
        return xp.abs(m1_reshaped - m2_reshaped)

    def _zscore(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if int(np.prod(m.shape)) == 0:
            return m
        mask_exp = mask[..., None]
        sum_mask = xp.sum(mask_exp)
        if float(sum_mask) == 0.0:
            return xp.full_like(m, xp.inf)
        mean_vals = xp.sum(m * mask_exp, axis=(0, 1)) / sum_mask
        var_vals = xp.sum((m - mean_vals) ** 2 * mask_exp, axis=(0, 1)) / sum_mask
        std_vals = xp.sqrt(var_vals) + 1e-8
        m = (m - mean_vals) / std_vals
        m = xp.where(mask_exp, m, xp.inf)
        return m

    z_score_distance = _zscore(distance_matrix[..., xp.newaxis], distance_mask).astype(xp.float16)

    stats_diff = _diff_matrix(stats_vecs, pre_stats_vecs)
    z_stats = _zscore(stats_diff, distance_mask)
    z_stats = (z_stats / stats_diff.shape[2]).astype(xp.float16)

    hu_diff = _diff_matrix(hu_vecs, pre_hu_vecs)
    z_hu = _zscore(hu_diff, distance_mask)
    z_hu = (z_hu / hu_diff.shape[2]).astype(xp.float16)

    z_score_matrix = xp.concatenate((z_score_distance, z_stats, z_hu), axis=2).astype(xp.float16)
    cost_matrix = xp.nansum(z_score_matrix, axis=2).astype(xp.float16)
    return cost_matrix.astype(xp.float32)


def test_get_cost_matrix_streaming_matches_broadcast_3d(make_hu_imageinfo_3d) -> None:
    """Streaming `_get_cost_matrix` ≈ broadcast reference on yeast 3D fixture.

    Drives the production matching pipeline through:
      1. The new per-feature streaming implementation (production code)
      2. The inline broadcast-reference reimplementation
        (`_broadcast_cost_matrix_reference`)

    Asserts approx-equal at ``rtol=1e-3, atol=1e-3`` per ADR 0009. Both paths
    run in the same process so ``scipy.spatial.distance.cdist`` is constant
    (no cross-platform variance to confound the assertion).

    `+inf` entries (masked-out pairs) are checked separately — ``np.isclose``
    returns False for ``inf`` vs ``inf`` by default. Both paths must produce
    `+inf` at the same positions.
    """
    info = make_hu_imageinfo_3d()
    h = HuMomentTracking(info, HuMomentTrackingConfig(device="cpu"), num_t=2)
    h._allocate_memory()

    feat_t0 = h._get_frame_features(0)
    feat_t1 = h._get_frame_features(1)

    if feat_t0.coords_phys.shape[0] == 0 or feat_t1.coords_phys.shape[0] == 0:
        _release_hu(h)
        pytest.skip(
            "3D fixture frame 0 or 1 has no markers; equivalence test needs realistic input."
        )

    cost_streaming = h._get_cost_matrix(
        feat_t1.coords_phys, feat_t0.coords_phys,
        feat_t1.stats, feat_t0.stats,
        feat_t1.hu, feat_t0.hu,
    )

    cost_broadcast = _broadcast_cost_matrix_reference(
        h,
        feat_t1.coords_phys, feat_t0.coords_phys,
        feat_t1.stats, feat_t0.stats,
        feat_t1.hu, feat_t0.hu,
    )

    assert cost_streaming.shape == cost_broadcast.shape, (
        f"Shape mismatch: {cost_streaming.shape} vs {cost_broadcast.shape}"
    )

    # `+inf` mask must match exactly — both paths short-circuit identically.
    inf_streaming = np.isinf(cost_streaming)
    inf_broadcast = np.isinf(cost_broadcast)
    np.testing.assert_array_equal(
        inf_streaming,
        inf_broadcast,
        err_msg="`+inf` mask differs between streaming and broadcast cost matrices",
    )

    finite_mask = ~inf_streaming
    if finite_mask.any():
        np.testing.assert_allclose(
            cost_streaming[finite_mask].astype(np.float32),
            cost_broadcast[finite_mask].astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
            err_msg=(
                "Streaming/broadcast cost values disagree beyond rtol=1e-3, atol=1e-3 "
                "(see ADR 0009 for the test-bar rationale)"
            ),
        )

    _release_hu(h)


def test_get_cost_matrix_single_matchable_pair_finite_elsewhere_inf() -> None:
    """Exactly one (post, pre) pair within max_distance_um → finite there, +inf elsewhere.

    Pins the per-pair masking semantics: positions where
    ``distance < max_distance_um`` get a finite cost (z-scored
    feature differences); positions outside that radius get `+inf`.
    Slice 2 (#198)'s ``cost = xp.where(mask, cost, xp.inf)`` final
    step preserves this exact mask propagation.
    """
    h = _make_bare_hu_for_cost_matrix(max_distance_um=1.0)
    # post[0] is close to pre[0] (matchable); post[1] is far from pre[0]
    coords_post = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    coords_pre = np.array([[0.1, 0.0, 0.0]])
    rng = np.random.default_rng(37)
    stats_post = rng.normal(0.0, 1.0, size=(2, 4)).astype(np.float32)
    stats_pre = rng.normal(0.0, 1.0, size=(1, 4)).astype(np.float32)
    hu_post = rng.normal(0.0, 1.0, size=(2, 6)).astype(np.float32)
    hu_pre = rng.normal(0.0, 1.0, size=(1, 6)).astype(np.float32)

    cost = h._get_cost_matrix(coords_post, coords_pre, stats_post, stats_pre, hu_post, hu_pre)
    assert cost.shape == (2, 1)
    assert np.isfinite(cost[0, 0]), f"Expected finite cost at the matchable pair; got {cost[0, 0]}"
    assert np.isinf(cost[1, 0]) and cost[1, 0] > 0, (
        f"Expected `+inf` at the unmatchable pair; got {cost[1, 0]}"
    )


# -------------------------------------------------------------------------
# Direct synthetic tests for `_find_best_matches`
#
# Issue #201 (bundled cleanups) vectorizes the Python row/col loop with
# `np.flatnonzero` + boolean mask + `.tolist()`. These tests pin the
# observable contract: row+col concatenation order, the `<= cost_cutoff`
# inclusive boundary, and the empty-matrix short-circuit. No fixture —
# bare `HuMomentTracking` with `xp` and `cost_cutoff` only.
# -------------------------------------------------------------------------


def _make_bare_hu_for_find_best_matches(cost_cutoff: float = 1.0) -> HuMomentTracking:
    h = HuMomentTracking.__new__(HuMomentTracking)
    h.xp = np
    h.cost_cutoff = cost_cutoff
    return h


def test_find_best_matches_empty_cost_matrix() -> None:
    """`(0, 0)` cost matrix returns `([], [], [])` short-circuit."""
    h = _make_bare_hu_for_find_best_matches()
    cost = np.zeros((0, 0), dtype=np.float32)
    row_matches, col_matches, costs = h._find_best_matches(cost)
    assert row_matches == []
    assert col_matches == []
    assert costs == []


def test_find_best_matches_all_above_cutoff_skipped() -> None:
    """Every row/col min above `cost_cutoff` → no matches returned.

    Pins the strict `>` skip semantics: values strictly above
    cost_cutoff are skipped; values equal to cost_cutoff are kept
    (covered separately by
    :func:`test_find_best_matches_cutoff_boundary_inclusive`).
    """
    h = _make_bare_hu_for_find_best_matches(cost_cutoff=1.0)
    cost = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    row_matches, col_matches, costs = h._find_best_matches(cost)
    assert row_matches == []
    assert col_matches == []
    assert costs == []


def test_find_best_matches_basic_concatenated_row_col() -> None:
    """Row candidates come first, then column candidates — concatenation order pinned.

    On a small (2, 2) cost matrix where all entries pass the cutoff,
    the function returns four candidate triples: two row-based plus
    two column-based. The pre-vectorization Python loop appends row
    candidates first (in row-index order), then column candidates
    (in col-index order); the vectorized refactor must preserve this
    exact ordering or downstream `_run_hu_tracking`'s
    `frame_vector_array` row order changes silently.
    """
    h = _make_bare_hu_for_find_best_matches(cost_cutoff=1.0)
    cost = np.array([[0.5, 0.9], [0.8, 0.3]], dtype=np.float32)
    row_matches, col_matches, costs = h._find_best_matches(cost)
    # Row candidates: row 0 best is j=0 (val 0.5); row 1 best is j=1 (val 0.3)
    # Col candidates: col 0 best is i=0 (val 0.5); col 1 best is i=1 (val 0.3)
    assert row_matches == [0, 1, 0, 1]
    assert col_matches == [0, 1, 0, 1]
    np.testing.assert_allclose(costs, [0.5, 0.3, 0.5, 0.3], rtol=1e-6)


def test_find_best_matches_cutoff_boundary_inclusive() -> None:
    """Value exactly equal to `cost_cutoff` is kept (`>` skip, not `>=`).

    The check at the original :func:`_find_best_matches` is
    ``if val > self.cost_cutoff: continue`` — strict `>`. Pinning
    the inclusive boundary so the vectorized refactor doesn't
    silently flip to `>=` (which would drop matches at the cutoff
    edge — a real-world regression class).
    """
    h = _make_bare_hu_for_find_best_matches(cost_cutoff=1.0)
    # Row 0 min is 0.5 (j=0); row 1 min is 1.0 (j=1, exactly at cutoff)
    # Col 0 min is 0.5 (i=0); col 1 min is 1.0 (i=1, exactly at cutoff)
    cost = np.array([[0.5, 1.5], [2.0, 1.0]], dtype=np.float32)
    row_matches, col_matches, costs = h._find_best_matches(cost)
    # All four candidates pass the inclusive cutoff
    assert row_matches == [0, 1, 0, 1]
    assert col_matches == [0, 1, 0, 1]
    np.testing.assert_allclose(costs, [0.5, 1.0, 0.5, 1.0], rtol=1e-6)


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


# -------------------------------------------------------------------------
# HuMomentTrackingConfig validation (__post_init__)
# -------------------------------------------------------------------------

def test_hu_config_default_constructs() -> None:
    HuMomentTrackingConfig()


def test_hu_config_rejects_bad_device() -> None:
    with pytest.raises(ValueError, match="device"):
        HuMomentTrackingConfig(device="bogus")


def test_hu_config_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        HuMomentTrackingConfig(mode="hybrid")


@pytest.mark.parametrize("mode", ["auto", "dense", "sparse"])
def test_hu_config_accepts_valid_mode(mode: str) -> None:
    HuMomentTrackingConfig(mode=mode)


@pytest.mark.parametrize("field", [
    "max_distance_um", "max_dense_pairs",
    "max_dense_roi_voxels_cpu", "max_dense_roi_voxels_gpu", "cost_cutoff",
])
def test_hu_config_rejects_nonpositive_numeric(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        HuMomentTrackingConfig(**{field: 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=field):
        HuMomentTrackingConfig(**{field: -1})  # type: ignore[arg-type]
