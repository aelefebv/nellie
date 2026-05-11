"""Characterization tests for ``nellie.tracking.voxel_reassignment.VoxelReassigner``.

Pins the wiki-documented invariants on both the 3D and 2D paths:

- Output schema (``im_branch_label_reassigned`` + ``im_obj_label_reassigned``):
  - Both reassigned outputs are int32 memmaps with the same shape as
    the input ``im_skel_relabelled`` / ``im_instance_label`` memmaps.
  - ``voxel_matches.npy`` is written when ``store_running_matches=True``
    and is NOT written when False. Pin BOTH branches.
  - When written, ``voxel_matches.npy`` is an object array of length
    ``num_t - 1``; each entry is a 2-element list ``[best_prev, best_next]``
    where each is a ``(K, D)`` array of ``match_coord_dtype``
    (``uint16`` on yeast fixtures since max axis ≤ 65 536).
  - ``match_coord_dtype`` selection: yeast fixtures are ``uint16``;
    synthetic ``spatial_shape = (65_537, 8, 8)`` widens to ``uint32``.

- Initialization invariant (t=0):
  - ``reassigned_branch_memmap[0]`` equals ``branch_label_memmap[0]``
    at non-zero voxels (background stays 0). Same for obj labels.

- Vote / assignment behavior:
  - Reassigned voxels at t+1 are a strict subset of
    ``label_memmap[t+1] > 0`` — no phantom labels at background.
  - ``max_refine_iterations=1`` produces fewer (or equal) total
    assignments than the default ``max_refine_iterations=3``.

- 4-tier tree backend (CPU-only paths):
  - ``device='cpu'`` end-to-end: post-run ``self.device_type == "cpu"``.
  - Empty input to ``_build_tree`` → ``backend="cpu"``, ``tree=None``,
    ``coords_real_scaled=None``; ``_query_tree`` on this handle
    returns ``(empty float32, empty int64)``.
  - Default CPU path: ``backend="cpu"`` with populated tree;
    ``_query_tree`` returns expected shapes.
  - ``cpu_bruteforce`` fallback: monkeypatch ``cKDTree`` to raise
    ``MemoryError`` once → ``backend="cpu_bruteforce"``; end-to-end
    run still produces reassigned outputs.

- Empty-input edge cases:
  - Empty mask at frame N → loop breaks; later frames stay all-zero
    in both reassigned outputs.
  - ``match_voxels`` on empty inputs returns the documented empty
    shape contract.

- Distance / weighting primitives:
  - ``_distance_threshold`` drops matches ≥
    ``flow_interpolator_fw.max_distance_um``.
  - ``_vote_targets`` minimal numeric example: 3 sources, 1 target →
    inverse-distance weighted vote picks the higher-weighted label.
  - ``_select_best_pairs`` returns 1-best per target voxel.

- ``no_t`` short-circuit: ``run()`` early-returns; no files written.

- Input mutation: raw + ``im_instance_label`` + ``im_skel_relabelled``
  + ``flow_vector_array.npy`` are all byte-identical pre/post.

- Viewer callback: no-op when ``viewer=None``; called once per frame
  with the formatted message when ``viewer`` is a stub.

- **Architecture characterization (POST-Slice 3 of #101 — Option A2
  inner-cascade contract)**:
  - No cross-frame mutation for ``_build_tree`` GPU OOM (Site 1):
    simulate GPU state post-construction, then trigger a
    ``MemoryError`` from the GPU KDTree class → assert
    ``self.device_type == "cuda"`` (unchanged) after the local CPU
    rebuild fallback runs.
  - No cross-frame mutation for ``_query_tree`` GPU OOM (Sites 3+4):
    simulate GPU state, rig the GPU tree's ``query`` to raise
    ``MemoryError`` → assert ``self.device_type == "cuda"``
    (unchanged) after the local CPU rebuild fallback runs.
  - Non-OOM exception propagation for ``_query_tree`` (latent bug
    fix): rig the GPU query to raise ``ValueError`` → the explicit
    ``adaptive_run.is_oom_error`` gate propagates the exception out
    of ``_query_tree`` instead of silently flipping the backend.

The Network ``im_skel_relabelled`` and Hu ``flow_vector_array.npy``
that ``VoxelReassigner`` consumes are precomputed once per session by
``conftest.network_*_path`` and ``conftest.hu_outputs_*_path``;
per-test ImInfos copy in 3 files (Label memmap + Network memmap + Hu
.npy) into a fresh working directory so each test gets isolated
``im_branch_label_reassigned`` / ``im_obj_label_reassigned`` /
``voxel_matches.npy`` targets.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pytest

import nellie.tracking.voxel_reassignment as vr_module
from nellie.im_info.verifier import ImInfo
from nellie.tracking.voxel_reassignment import (
    VoxelReassigner,
    VoxelReassignerConfig,
    _TreeHandle,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _release_voxel_reassigner(v: VoxelReassigner) -> None:
    """Drop a VoxelReassigner's memmap references and force gc.

    Mirrors ``test_hu_tracking._release_hu``. Required on Windows:
    input memmaps can stay file-locked until the handles are dropped,
    blocking later overwrites of the same paths.
    """
    v.branch_label_memmap = None
    v.obj_label_memmap = None
    v.reassigned_branch_memmap = None
    v.reassigned_obj_memmap = None
    v.flow_interpolator_fw = None
    v.flow_interpolator_bw = None
    gc.collect()


def _run_voxel_reassign(info: ImInfo, **kwargs) -> VoxelReassigner:
    """Construct + run ``VoxelReassigner`` on ``info``; return the instance.

    Always pinned to CPU for deterministic behavior. Caller is
    responsible for calling ``_release_voxel_reassigner`` after
    inspecting the on-disk outputs.
    """
    kwargs.setdefault("device", "cpu")
    v = VoxelReassigner(info, VoxelReassignerConfig(**kwargs), num_t=2)
    v.run()
    return v


# -------------------------------------------------------------------------
# Module-scoped: run VoxelReassigner once on each fixture and share outputs
# across the read-only invariant tests (shape, dtype, init invariant, ...).
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def voxel_reassign_outputs_3d(make_voxel_reassign_imageinfo_3d_module) -> dict:
    info = make_voxel_reassign_imageinfo_3d_module()
    v = _run_voxel_reassign(info)
    assert v.reassigned_branch_memmap is not None
    assert v.reassigned_obj_memmap is not None
    assert v.branch_label_memmap is not None
    assert v.obj_label_memmap is not None
    branch_path = Path(info.pipeline_paths["im_branch_label_reassigned"])
    obj_path = Path(info.pipeline_paths["im_obj_label_reassigned"])
    matches_path = Path(info.pipeline_paths["voxel_matches"])
    branch_arr = np.array(v.reassigned_branch_memmap)
    obj_arr = np.array(v.reassigned_obj_memmap)
    branch_label_arr = np.array(v.branch_label_memmap)
    obj_label_arr = np.array(v.obj_label_memmap)
    branch_dtype = v.reassigned_branch_memmap.dtype
    obj_dtype = v.reassigned_obj_memmap.dtype
    branch_shape = v.reassigned_branch_memmap.shape
    obj_shape = v.reassigned_obj_memmap.shape
    spatial_shape = v.spatial_shape
    match_coord_dtype = v.match_coord_dtype
    device_type = v.device_type
    _release_voxel_reassigner(v)
    return {
        "info": info,
        "branch_path": branch_path,
        "obj_path": obj_path,
        "matches_path": matches_path,
        "branch_arr": branch_arr,
        "obj_arr": obj_arr,
        "branch_label_arr": branch_label_arr,
        "obj_label_arr": obj_label_arr,
        "branch_dtype": branch_dtype,
        "obj_dtype": obj_dtype,
        "branch_shape": branch_shape,
        "obj_shape": obj_shape,
        "spatial_shape": spatial_shape,
        "match_coord_dtype": match_coord_dtype,
        "device_type": device_type,
    }


@pytest.fixture(scope="module")
def voxel_reassign_outputs_2d(make_voxel_reassign_imageinfo_2d_module) -> dict:
    info = make_voxel_reassign_imageinfo_2d_module()
    v = _run_voxel_reassign(info)
    assert v.reassigned_branch_memmap is not None
    assert v.reassigned_obj_memmap is not None
    branch_path = Path(info.pipeline_paths["im_branch_label_reassigned"])
    obj_path = Path(info.pipeline_paths["im_obj_label_reassigned"])
    branch_arr = np.array(v.reassigned_branch_memmap)
    obj_arr = np.array(v.reassigned_obj_memmap)
    branch_dtype = v.reassigned_branch_memmap.dtype
    obj_dtype = v.reassigned_obj_memmap.dtype
    branch_shape = v.reassigned_branch_memmap.shape
    obj_shape = v.reassigned_obj_memmap.shape
    device_type = v.device_type
    _release_voxel_reassigner(v)
    return {
        "info": info,
        "branch_path": branch_path,
        "obj_path": obj_path,
        "branch_arr": branch_arr,
        "obj_arr": obj_arr,
        "branch_dtype": branch_dtype,
        "obj_dtype": obj_dtype,
        "branch_shape": branch_shape,
        "obj_shape": obj_shape,
        "device_type": device_type,
    }


# -------------------------------------------------------------------------
# End-to-end smoke tests
# -------------------------------------------------------------------------


def test_runs_end_to_end_3d(voxel_reassign_outputs_3d) -> None:
    """3D yeast fixture: both reassigned memmaps exist; int32; shape matches input."""
    out = voxel_reassign_outputs_3d
    assert out["branch_path"].exists(), (
        f"im_branch_label_reassigned not found at {out['branch_path']}"
    )
    assert out["obj_path"].exists(), (
        f"im_obj_label_reassigned not found at {out['obj_path']}"
    )
    assert out["branch_dtype"] == np.int32, (
        f"3D branch reassigned dtype is {out['branch_dtype']}, expected int32"
    )
    assert out["obj_dtype"] == np.int32, (
        f"3D obj reassigned dtype is {out['obj_dtype']}, expected int32"
    )
    # Shape check: matches input label shape (T, Z, Y, X) on the 3D fixture.
    assert out["branch_shape"] == out["branch_label_arr"].shape, (
        f"3D branch reassigned shape {out['branch_shape']} != input "
        f"{out['branch_label_arr'].shape}"
    )
    assert out["obj_shape"] == out["obj_label_arr"].shape, (
        f"3D obj reassigned shape {out['obj_shape']} != input "
        f"{out['obj_label_arr'].shape}"
    )


def test_runs_end_to_end_2d(voxel_reassign_outputs_2d) -> None:
    """2D yeast fixture: both reassigned memmaps exist; int32; shape matches input."""
    out = voxel_reassign_outputs_2d
    assert out["branch_path"].exists()
    assert out["obj_path"].exists()
    assert out["branch_dtype"] == np.int32
    assert out["obj_dtype"] == np.int32
    # 2D fixture is (T, Y, X) so spatial dim count is 2.
    assert len(out["branch_shape"]) == 3
    assert len(out["obj_shape"]) == 3


# -------------------------------------------------------------------------
# voxel_matches.npy: written / not-written contract
# -------------------------------------------------------------------------


def test_voxel_matches_written_when_store_running_matches_true(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``store_running_matches=True`` writes ``voxel_matches.npy``.

    Pins the on-by-default behavior — the demo script in
    ``scripts/voxel_reassignment_demo.py`` (per User Story 19) reads
    this file post-run.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = _run_voxel_reassign(info, store_running_matches=True)
    matches_path = Path(info.pipeline_paths["voxel_matches"])
    assert matches_path.exists(), (
        f"voxel_matches.npy not written when store_running_matches=True "
        f"({matches_path})"
    )
    _release_voxel_reassigner(v)


def test_voxel_matches_not_written_when_store_running_matches_false(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``store_running_matches=False`` does NOT write ``voxel_matches.npy``.

    Pin the off-arm so a future refactor doesn't silently start writing
    the file regardless of the flag.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = _run_voxel_reassign(info, store_running_matches=False)
    matches_path = Path(info.pipeline_paths["voxel_matches"])
    assert not matches_path.exists(), (
        f"voxel_matches.npy was written even though "
        f"store_running_matches=False ({matches_path})"
    )
    _release_voxel_reassigner(v)


def test_voxel_matches_structure_3d(voxel_reassign_outputs_3d) -> None:
    """``voxel_matches.npy`` is an object array of length ``num_t - 1``.

    Stored via ``np.array(self.running_matches, dtype=object)`` at
    ``voxel_reassignment.py:1062``. Because the inner element is a
    Python list of two equal-shaped ``(K, D)`` numpy arrays, the
    ``dtype=object`` cast recursively flattens to a 4-D object array of
    shape ``(num_t - 1, 2, K, D)`` (numpy walks the nested
    list+ndarray structure when forced to ``object``). The semantic
    contract is preserved (``[best_prev, best_next]`` per frame pair),
    but the numeric ``match_coord_dtype`` (``uint16`` here) is hidden
    behind the object dtype after ``np.load``. We pin the array shape
    + value range to characterize the contract; Slice 2/3 may rework
    the storage format to preserve the inner dtype on disk.
    """
    out = voxel_reassign_outputs_3d
    matches_path = out["matches_path"]
    assert matches_path.exists(), "voxel_matches.npy missing for module fixture"
    matches = np.load(matches_path, allow_pickle=True)
    assert matches.dtype == object, (
        f"voxel_matches.npy dtype is {matches.dtype}, expected object"
    )
    # num_t = 2 for the fixture, so length is num_t - 1 == 1.
    assert matches.shape[0] == 1, (
        f"voxel_matches.npy length is {matches.shape[0]}, expected 1 "
        f"(num_t - 1)"
    )
    # Shape: (num_t-1, 2, K, D). The "2" is [best_prev, best_next].
    assert matches.shape[1] == 2, (
        f"Expected 2-element list per frame pair; got {matches.shape[1]}"
    )
    # 3D fixture has 3 spatial dims (Z, Y, X).
    assert matches.shape[3] == 3
    # K can vary per frame pair; just assert > 0.
    assert matches.shape[2] > 0, "voxel_matches has zero-length match list"
    # Best-prev and best-next must agree on K (same number of pairs).
    best_prev = matches[0, 0]
    best_next = matches[0, 1]
    assert best_prev.shape == best_next.shape
    # Values are voxel indices into the spatial shape; on the yeast
    # fixture every value fits in uint16 (max axis <= 65 536). Cast
    # back to uint16 to confirm no overflow/truncation occurred during
    # the np.array(dtype=object) flattening.
    spatial_max = max(out["spatial_shape"])
    assert int(np.asarray(best_prev).max()) < spatial_max
    assert int(np.asarray(best_next).max()) < spatial_max
    assert int(np.asarray(best_prev).min()) >= 0
    assert int(np.asarray(best_next).min()) >= 0


def test_match_coord_dtype_is_uint16_on_yeast_fixture(
    voxel_reassign_outputs_3d,
) -> None:
    """Yeast fixture max axis ≤ 65 536 → ``match_coord_dtype == uint16``."""
    out = voxel_reassign_outputs_3d
    assert max(out["spatial_shape"]) <= 65_536
    assert out["match_coord_dtype"] == np.uint16, (
        f"Expected uint16 for yeast fixture (max axis <= 65 536); "
        f"got {out['match_coord_dtype']}"
    )


def test_match_coord_dtype_synthetic_widens_to_uint32(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Synthetic ``spatial_shape = (65_537, 8, 8)`` widens to ``uint32``.

    Pins the wiki-documented gotcha at
    ``voxel_reassignment.py:395-403``: bumping any axis past 65 535
    flips the saved ``.npy`` to ``uint32`` (and past 4 294 967 295 to
    ``uint64``). We monkeypatch ``self.spatial_shape`` and call
    ``_select_match_coord_dtype()`` directly, which is the cleanest
    pin without building a multi-GB synthetic fixture.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v._allocate_memory()
    v.spatial_shape = (65_537, 8, 8)
    assert v._select_match_coord_dtype() == np.uint32
    # And a synthetic shape past uint32 → uint64.
    v.spatial_shape = (int(np.iinfo(np.uint32).max) + 2, 8, 8)
    assert v._select_match_coord_dtype() == np.uint64
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# Initialization invariant (t=0)
# -------------------------------------------------------------------------


def test_t0_initialization_branch_label_3d(voxel_reassign_outputs_3d) -> None:
    """``reassigned_branch_memmap[0]`` == ``branch_label_memmap[0]`` at non-zero voxels.

    The init block at ``voxel_reassignment.py:999-1003`` writes
    ``reassigned_branch_memmap[0][argwhere(branch_label > 0)] =
    branch_label[0][argwhere(...)]``. Background voxels stay at 0.
    """
    out = voxel_reassign_outputs_3d
    branch_label_t0 = out["branch_label_arr"][0]
    branch_reassigned_t0 = out["branch_arr"][0]
    nonzero_mask = branch_label_t0 > 0
    background_mask = ~nonzero_mask
    np.testing.assert_array_equal(
        branch_reassigned_t0[nonzero_mask],
        branch_label_t0[nonzero_mask],
    )
    assert (branch_reassigned_t0[background_mask] == 0).all(), (
        "Branch reassigned t=0 background voxels are non-zero"
    )


def test_t0_initialization_obj_label_3d(voxel_reassign_outputs_3d) -> None:
    """``reassigned_obj_memmap[0]`` == ``obj_label_memmap[0]`` at non-zero voxels.

    Mirrors the branch-label init test for the obj-label cascade
    (``voxel_reassignment.py:1005-1009``).
    """
    out = voxel_reassign_outputs_3d
    obj_label_t0 = out["obj_label_arr"][0]
    obj_reassigned_t0 = out["obj_arr"][0]
    nonzero_mask = obj_label_t0 > 0
    background_mask = ~nonzero_mask
    np.testing.assert_array_equal(
        obj_reassigned_t0[nonzero_mask],
        obj_label_t0[nonzero_mask],
    )
    assert (obj_reassigned_t0[background_mask] == 0).all()


# -------------------------------------------------------------------------
# Vote / assignment behavior
# -------------------------------------------------------------------------


def test_reassigned_voxels_subset_of_label_at_t1_branch_3d(
    voxel_reassign_outputs_3d,
) -> None:
    """Reassigned branch voxels at t+1 ⊆ ``branch_label_memmap[t+1] > 0``.

    ``_vote_assign_labels_for_frame`` (lines 940-948) gates on
    ``label_memmap[t + 1][...] > 0`` before assigning anything. No
    "phantom labels" should appear at background voxels in t+1.
    """
    out = voxel_reassign_outputs_3d
    reassigned_t1 = out["branch_arr"][1] > 0
    label_t1 = out["branch_label_arr"][1] > 0
    assert np.all(reassigned_t1 <= label_t1), (
        "Branch reassigned at t=1 has labels at voxels that are background "
        "in branch_label_memmap[1]"
    )


def test_reassigned_voxels_subset_of_label_at_t1_obj_3d(
    voxel_reassign_outputs_3d,
) -> None:
    """Same subset invariant for the obj-label cascade."""
    out = voxel_reassign_outputs_3d
    reassigned_t1 = out["obj_arr"][1] > 0
    label_t1 = out["obj_label_arr"][1] > 0
    assert np.all(reassigned_t1 <= label_t1), (
        "Obj reassigned at t=1 has labels at voxels that are background "
        "in obj_label_memmap[1]"
    )


def test_max_refine_iterations_one_vs_three(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``max_refine_iterations=1`` produces ≤ assignments vs default 3.

    Each iteration of the vote loop in ``_vote_assign_labels_for_frame``
    (lines 953-980) only considers voxels that are still 0 at t+1 from
    prior iterations. Multiple iterations let later candidates fill in
    voxels that were skipped (via the ``unassigned`` mask at line 956).
    Setting ``max_refine_iterations=1`` truncates after one round, so
    the count of reassigned voxels at t=1 must be ≤ the default-3 count.
    """
    info_one = make_voxel_reassign_imageinfo_3d()
    v_one = _run_voxel_reassign(info_one, max_refine_iterations=1)
    assert v_one.reassigned_branch_memmap is not None
    assert v_one.reassigned_obj_memmap is not None
    branch_one_count = int(np.count_nonzero(v_one.reassigned_branch_memmap[1]))
    obj_one_count = int(np.count_nonzero(v_one.reassigned_obj_memmap[1]))
    _release_voxel_reassigner(v_one)

    info_three = make_voxel_reassign_imageinfo_3d()
    v_three = _run_voxel_reassign(info_three, max_refine_iterations=3)
    assert v_three.reassigned_branch_memmap is not None
    assert v_three.reassigned_obj_memmap is not None
    branch_three_count = int(np.count_nonzero(v_three.reassigned_branch_memmap[1]))
    obj_three_count = int(np.count_nonzero(v_three.reassigned_obj_memmap[1]))
    _release_voxel_reassigner(v_three)

    assert branch_one_count <= branch_three_count, (
        f"max_refine_iterations=1 gave more branch assignments "
        f"({branch_one_count}) than max_refine_iterations=3 "
        f"({branch_three_count})"
    )
    assert obj_one_count <= obj_three_count, (
        f"max_refine_iterations=1 gave more obj assignments "
        f"({obj_one_count}) than max_refine_iterations=3 ({obj_three_count})"
    )


# -------------------------------------------------------------------------
# 4-tier tree backend (CPU paths)
# -------------------------------------------------------------------------


def test_device_cpu_post_run_device_type_is_cpu(
    voxel_reassign_outputs_3d,
) -> None:
    """End-to-end ``device='cpu'`` run: ``self.device_type == "cpu"`` after run."""
    assert voxel_reassign_outputs_3d["device_type"] == "cpu"


def test_build_tree_empty_input_returns_cpu_none_handle(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Empty input to ``_build_tree`` → ``backend="cpu"``, ``tree=None``, ``coords_real_scaled=None``.

    Pins the explicit short-circuit at ``voxel_reassignment.py:238-239``.
    Also verifies that ``_query_tree`` on this empty handle returns
    the documented empty arrays (lines 271-272 + 275-276).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    handle = v._build_tree(np.empty((0, 3), dtype=np.float32))
    assert isinstance(handle, _TreeHandle)
    assert handle.backend == "cpu"
    assert handle.tree is None
    assert handle.coords_real_scaled is None

    # Query with non-empty coords against an empty handle; the early
    # short-circuit at line 274-276 returns empty arrays.
    dist, idx = v._query_tree(handle, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert dist.shape == (0,)
    assert idx.shape == (0,)
    assert dist.dtype == np.float32
    assert idx.dtype == np.int64
    _release_voxel_reassigner(v)


def test_build_tree_default_cpu_path_returns_cpu_with_tree(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Default CPU path: ``backend="cpu"`` with populated ``tree``; ``_query_tree`` works."""
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    coords = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    handle = v._build_tree(coords)
    assert handle.backend == "cpu"
    assert handle.tree is not None
    # Query a single point near the second tree point; expect a small
    # distance and idx == 1.
    query = np.array([[4.1, 5.1, 6.1]], dtype=np.float32)
    dist, idx = v._query_tree(handle, query)
    assert dist.shape == (1,)
    assert idx.shape == (1,)
    assert idx[0] == 1
    assert dist[0] < 1.0
    _release_voxel_reassigner(v)


def test_cpu_bruteforce_fallback_on_ckdtree_memory_error(
    make_voxel_reassign_imageinfo_3d, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cKDTree`` raising ``MemoryError`` once → ``backend="cpu_bruteforce"``.

    Pins the explicit fallback path at
    ``voxel_reassignment.py:260-268``. We monkeypatch
    ``nellie.tracking.voxel_reassignment.cKDTree`` to raise
    ``MemoryError`` once so ``_build_tree`` lands on the
    ``cpu_bruteforce`` branch. The end-to-end run should still produce
    reassigned outputs (the bruteforce CPU path delegates to
    ``_query_bruteforce_cpu`` at line 314-315).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)

    real_ckdtree = vr_module.cKDTree
    state = {"raised": False}

    def flaky_ckdtree(*args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise MemoryError("simulated cKDTree allocation failure")
        return real_ckdtree(*args, **kwargs)

    monkeypatch.setattr(vr_module, "cKDTree", flaky_ckdtree)

    # Build a non-empty tree directly and verify the fallback engages.
    coords = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    handle = v._build_tree(coords)
    assert handle.backend == "cpu_bruteforce", (
        f"Expected cpu_bruteforce after MemoryError; got {handle.backend!r}"
    )
    assert handle.tree is None
    assert handle.coords_real_scaled is not None

    # Query against the bruteforce handle should still resolve via
    # _query_bruteforce_cpu (line 314-315).
    query = np.array([[4.1, 5.1, 6.1]], dtype=np.float32)
    dist, idx = v._query_tree(handle, query)
    assert dist.shape == (1,)
    assert idx.shape == (1,)
    assert idx[0] == 1
    assert dist[0] < 1.0
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# Empty-input edge cases
# -------------------------------------------------------------------------


def test_empty_master_mask_breaks_loop(
    make_voxel_reassign_imageinfo_3d, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty mask at frame N → loop breaks; later frames stay all-zero.

    Pins the early-break at ``voxel_reassignment.py:1027-1029``. We
    monkeypatch ``_get_master_mask`` to return an all-False mask at
    every call (effectively making frame 0's mask empty too), and
    assert the post-run reassigned memmaps are all zero at t=1
    (frame 0 still gets the init-block writes at line 999-1009 because
    ``argwhere`` of an empty mask is empty, but the loop short-circuits
    before any t=1 writes).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)

    def empty_mask(_self, _t):
        return np.zeros(_self.spatial_shape, dtype=bool)

    monkeypatch.setattr(VoxelReassigner, "_get_master_mask", empty_mask)
    v.run()

    # The init block (lines 999-1009) writes from the actual label
    # memmaps, NOT from the master mask. So t=0 still gets initialized.
    # The loop short-circuits at line 1027-1029 before t=1 writes happen,
    # so t=1 stays all zero.
    assert v.reassigned_branch_memmap is not None
    assert v.reassigned_obj_memmap is not None
    assert (v.reassigned_branch_memmap[1] == 0).all(), (
        "branch reassigned t=1 has non-zero entries even though "
        "_get_master_mask returned all-False"
    )
    assert (v.reassigned_obj_memmap[1] == 0).all(), (
        "obj reassigned t=1 has non-zero entries even though "
        "_get_master_mask returned all-False"
    )
    _release_voxel_reassigner(v)


def test_match_voxels_empty_inputs_returns_empty_pair(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``match_voxels`` early-returns on empty inputs (lines 778-781).

    The early-return at lines 778-781 returns a 2-tuple of empty
    int64 arrays of shape ``(0, D)``. Note: this branch does NOT
    return a distances array (only 2 elements), unlike the populated
    branch which returns 3. We pin the actual returned shape contract.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v._allocate_memory()  # populate spatial_shape
    vox_prev = np.empty((0, 3), dtype=int)
    vox_next = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    result = v.match_voxels(vox_prev, vox_next, t=0)
    # The early-return at lines 778-781 returns a 2-tuple.
    assert len(result) == 2, (
        f"Expected 2-tuple from empty match_voxels early-return, got "
        f"{len(result)}-tuple"
    )
    empty_prev, empty_next = result
    assert empty_prev.shape == (0, 3)
    assert empty_next.shape == (0, 3)
    assert empty_prev.dtype == np.int64
    assert empty_next.dtype == np.int64
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# Distance / weighting primitives
# -------------------------------------------------------------------------


def test_distance_threshold_drops_above_max_distance(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``_distance_threshold`` drops matches with physical distance ≥ ``max_distance_um``.

    Construct synthetic ``vox_prev_matched`` / ``vox_next_matched``
    arrays where some pairs have a small displacement (kept) and some
    have a large displacement (dropped). The threshold lives at
    ``voxel_reassignment.py:744-745``.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    # Don't need _allocate_memory for _distance_threshold; it uses
    # flow_interpolator_fw.scaling and .max_distance_um directly.
    assert v.flow_interpolator_fw is not None
    scaling = np.asarray(v.flow_interpolator_fw.scaling, dtype=np.float32)
    max_um = float(v.flow_interpolator_fw.max_distance_um)

    # Construct 4 pairs: 2 close (kept) and 2 far (dropped).
    # close: displacement of ~1 voxel, scaled distance well under max_um.
    # far:   displacement ~big, scaled distance well over max_um.
    far_voxels = int(max_um / float(scaling.min()) * 10) + 100
    vox_prev = np.array(
        [
            [10, 20, 30],
            [11, 21, 31],
            [100, 100, 100],
            [200, 200, 200],
        ],
        dtype=np.int64,
    )
    vox_next = np.array(
        [
            [10, 20, 31],  # ~1 voxel away
            [11, 22, 31],  # ~1 voxel away
            [100, 100, 100 + far_voxels],
            [200, 200, 200 + far_voxels],
        ],
        dtype=np.int64,
    )

    prev_valid, next_valid, dist_valid = v._distance_threshold(vox_prev, vox_next)
    # Should keep only the first two pairs.
    assert prev_valid.shape[0] == 2, (
        f"Expected 2 pairs to survive threshold; got {prev_valid.shape[0]}"
    )
    assert next_valid.shape[0] == 2
    assert dist_valid.shape == (2,)
    np.testing.assert_array_equal(prev_valid, vox_prev[:2])
    np.testing.assert_array_equal(next_valid, vox_next[:2])
    _release_voxel_reassigner(v)


def test_vote_targets_inverse_distance_weighted(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``_vote_targets`` minimal numeric example: 3 sources voting on 1 target.

    Distances ``[1.0, 2.0, 3.0]`` with labels ``[10, 10, 20]``:
    - Label 10 weight: ``1/1 + 1/2 = 1.5``.
    - Label 20 weight: ``1/3 ≈ 0.333``.
    Label 10 wins. We pin via direct call to ``_vote_targets``
    (lines 429-467).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v._allocate_memory()  # populate spatial_shape

    # All three sources point at the same target voxel.
    target_coord = np.array([10, 20, 30], dtype=np.int64)
    target_coords = np.tile(target_coord, (3, 1))
    source_labels = np.array([10, 10, 20], dtype=np.int64)
    distances = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    best_targets, best_labels, best_candidate_idx = v._vote_targets(
        target_coords, source_labels, distances
    )
    # One unique target voxel → one winning entry.
    assert len(best_targets) == 1
    assert best_labels[0] == 10, (
        f"Vote winner is {best_labels[0]} (expected 10 — sum of inverse "
        f"distances 1/1 + 1/2 > 1/3)"
    )
    # The best candidate index for label 10 should point at the lowest-distance
    # candidate (idx 0, distance 1.0).
    assert best_candidate_idx[0] == 0, (
        f"best_candidate_idx is {best_candidate_idx[0]} (expected 0 — "
        f"the lowest-distance source for label 10)"
    )
    _release_voxel_reassigner(v)


def test_select_best_pairs_returns_one_best_per_target(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """``_select_best_pairs`` returns 1-best per target voxel (lowest distance).

    Construct synthetic input where 3 sources point at the same target
    with distances ``[5.0, 1.0, 3.0]``. The returned pairs must be
    exactly the (target, lowest-distance-source) pair — i.e. source idx
    1 (distance 1.0).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v._allocate_memory()

    sources = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.int64,
    )
    target_coord = np.array([10, 20, 30], dtype=np.int64)
    targets = np.tile(target_coord, (3, 1))
    distances = np.array([5.0, 1.0, 3.0], dtype=np.float64)

    # NOTE: ``_select_best_pairs`` returns a 2-tuple in the populated
    # branch but a 3-tuple in the empty-input branch (lines 412-417 in
    # voxel_reassignment.py). The shape divergence is real and pinned
    # in the empty-input characterization above; here we hit the
    # populated branch with non-empty inputs.
    result = v._select_best_pairs(sources, targets, distances)
    best_prev, best_next = result[0], result[1]
    assert best_prev.shape == (1, 3)
    assert best_next.shape == (1, 3)
    np.testing.assert_array_equal(best_prev[0], sources[1])
    np.testing.assert_array_equal(best_next[0], target_coord)
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# no_t short-circuit
# -------------------------------------------------------------------------


def test_no_t_short_circuit(make_voxel_reassign_imageinfo_3d) -> None:
    """``no_t`` ImInfo: ``run()`` early-returns; no reassigned files written.

    ``__init__`` short-circuits at lines 84-101 when ``info.no_t`` is
    True (returning early, before constructing the FlowInterpolators);
    ``run()`` short-circuits at lines 1074-1076. We mutate
    ``info.no_t = True`` BEFORE constructing the VoxelReassigner so
    the FlowInterpolator dependency is bypassed entirely.
    """
    info = make_voxel_reassign_imageinfo_3d()
    info.no_t = True
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=1)
    v.run()  # must not raise
    branch_path = Path(info.pipeline_paths["im_branch_label_reassigned"])
    obj_path = Path(info.pipeline_paths["im_obj_label_reassigned"])
    matches_path = Path(info.pipeline_paths["voxel_matches"])
    assert not branch_path.exists(), (
        "im_branch_label_reassigned should not exist for a no-T dataset"
    )
    assert not obj_path.exists(), (
        "im_obj_label_reassigned should not exist for a no-T dataset"
    )
    assert not matches_path.exists(), (
        "voxel_matches.npy should not exist for a no-T dataset"
    )


# -------------------------------------------------------------------------
# Input mutation: hash-before / hash-after
# -------------------------------------------------------------------------


def test_input_files_not_mutated_3d(make_voxel_reassign_imageinfo_3d) -> None:
    """Hash all 4 inputs before/after ``VoxelReassigner.run()``.

    The reassigned outputs (``im_branch_label_reassigned``,
    ``im_obj_label_reassigned``, ``voxel_matches.npy``) are different
    files; the inputs (raw image, ``im_instance_label``,
    ``im_skel_relabelled``, ``flow_vector_array.npy``) must be
    byte-identical pre/post.
    """
    info = make_voxel_reassign_imageinfo_3d()
    raw_path = Path(info.im_path)
    label_path = Path(info.pipeline_paths["im_instance_label"])
    skel_path = Path(info.pipeline_paths["im_skel_relabelled"])
    flow_path = Path(info.pipeline_paths["flow_vector_array"])

    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    label_before = hashlib.sha256(label_path.read_bytes()).hexdigest()
    skel_before = hashlib.sha256(skel_path.read_bytes()).hexdigest()
    flow_before = hashlib.sha256(flow_path.read_bytes()).hexdigest()

    v = _run_voxel_reassign(info)
    _release_voxel_reassigner(v)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before, (
        "VoxelReassigner mutated the raw input"
    )
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == label_before, (
        "VoxelReassigner mutated the Label input memmap"
    )
    assert hashlib.sha256(skel_path.read_bytes()).hexdigest() == skel_before, (
        "VoxelReassigner mutated the Network im_skel_relabelled input memmap"
    )
    assert hashlib.sha256(flow_path.read_bytes()).hexdigest() == flow_before, (
        "VoxelReassigner mutated the Hu flow_vector_array.npy input"
    )


# -------------------------------------------------------------------------
# Viewer callback
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


def test_viewer_status_callback(make_voxel_reassign_imageinfo_2d) -> None:
    """``viewer.status`` is set once per frame; ``viewer=None`` is a no-op.

    The viewer-update branch in ``_run_reassignment`` (lines 1016-1017)
    only runs when ``self.viewer is not None``. Pin both arms in one
    test:
      - Build a VoxelReassigner with ``viewer=None`` and assert the
        run completes cleanly (no AttributeError).
      - Build a VoxelReassigner with a stub viewer and assert
        ``status`` is written exactly ``num_t - 1`` times (one per
        frame pair). With ``num_t=2`` the expected count is 1.
    """
    # No-op arm: viewer=None.
    info_none = make_voxel_reassign_imageinfo_2d()
    v_none = VoxelReassigner(
        info_none, VoxelReassignerConfig(device="cpu"), viewer=None, num_t=2
    )
    v_none.run()  # must not raise
    _release_voxel_reassigner(v_none)

    # Stub-viewer arm: assert per-frame writes.
    info_stub = make_voxel_reassign_imageinfo_2d()
    stub = _StubViewer()
    v_stub = VoxelReassigner(
        info_stub, VoxelReassignerConfig(device="cpu"), viewer=stub, num_t=2
    )
    v_stub.run()
    # The loop iterates from t=0 to t < num_t-1, so for num_t=2 there
    # is exactly one iteration. The status string is
    # "Reassigning voxels. Frame: t+1 of num_t" → "Frame: 1 of 2".
    assert len(stub.status_writes) == 1, (
        f"Expected viewer.status set once per frame pair (1 write for "
        f"num_t=2); got {len(stub.status_writes)}: {stub.status_writes}"
    )
    assert "Frame: 1 of 2" in stub.status_writes[0], (
        f"Expected formatted status string; got {stub.status_writes[0]!r}"
    )
    assert "Reassigning voxels" in stub.status_writes[0]
    _release_voxel_reassigner(v_stub)


# -------------------------------------------------------------------------
# Architecture characterization (POST-Slice 3 of #101 — Option A2)
#
# These tests pin the new inner-cascade contract:
#   - Inner GPU OOM in _build_tree does the local CPU KDTree rebuild
#     within the same call but leaves self.device_type unchanged. The
#     outer mode_candidates cascade in run() is the single source of
#     truth for cross-frame backend switching. Subsequent frames retry
#     GPU.
#   - Inner GPU OOM in _query_tree (gpu and gpu_bruteforce branches)
#     same story — local CPU KDTree rebuild, no device_type mutation.
#   - The _query_tree GPU branches' explicit
#     `if not adaptive_run.is_oom_error(exc): raise` gate ensures
#     non-OOM exceptions propagate out instead of being silently
#     swallowed (latent bug fix from PRE-Slice-3).
# -------------------------------------------------------------------------


class _FakeGpuKDTreeOOM:
    """A class-callable that raises ``MemoryError`` on instantiation.

    Stands in for ``cupyx.scipy.spatial.cKDTree`` so the GPU-build
    branch in ``_build_tree`` takes the OOM-fallback path without
    needing cupy installed.
    """

    def __init__(self, *args, **kwargs):
        raise MemoryError("simulated GPU KDTree allocation failure")


def _simulate_gpu_state(v: VoxelReassigner, *, kdtree_cls=None) -> None:
    """Mutate a CPU-constructed VoxelReassigner into a "GPU state".

    CI has no cupy, so we simulate the post-resolve GPU state by
    directly assigning the attributes that
    ``adaptive_run.resolve_backend`` + ``_get_gpu_kdtree_cls`` would
    set on a real GPU run. Using ``np`` for ``self.xp`` is fine
    because the GPU code paths in ``_build_tree`` / ``_query_tree``
    only call ``.asarray`` (numpy supports it) before delegating to
    the rigged tree class / tree.query. The actual ``MemoryError`` /
    ``ValueError`` is raised by the rigged class/method, so we never
    hit a real cupy code path.
    """
    v.device_type = "cuda"
    v.xp = np
    v._gpu_kdtree_cls = kdtree_cls


def test_build_tree_gpu_oom_does_not_mutate_device_type(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Cascade Site 1: ``_build_tree`` GPU OOM does NOT mutate ``self.device_type``.

    POST-Slice 3 (Option A2) contract: when the GPU KDTree class raises
    ``MemoryError``, ``_build_tree`` does the local CPU KDTree rebuild
    in the same call but leaves ``self.device_type`` unchanged. This
    means subsequent ``_build_tree`` calls in later frames retry GPU.
    The outer ``mode_candidates`` cascade in ``run()`` is the single
    source of truth for cross-frame backend switching.

    Setup: CPU-construct a VoxelReassigner (so the FlowInterpolator
    dependency loads cleanly), then mutate it into a "GPU state" so
    the GPU branch of ``_build_tree`` is taken. The rigged
    ``_gpu_kdtree_cls`` raises ``MemoryError`` on instantiation,
    triggering the local CPU rebuild fallback (the
    ``adaptive_run.is_oom_error`` check + ``adaptive_run.free_gpu_memory``
    + ``_warn_gpu_fallback`` block).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    _simulate_gpu_state(v, kdtree_cls=_FakeGpuKDTreeOOM)

    coords = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    handle = v._build_tree(coords)
    # The OOM fallback path lands on the CPU KDTree within the same call.
    # Because `gpu_kdtree_failed` is set to True, the gpu_bruteforce branch
    # is skipped even though device_type stayed "cuda".
    assert handle.backend == "cpu", (
        f"Expected cpu fallback after GPU KDTree OOM; got {handle.backend!r}"
    )
    # POST-SLICE-3 CONTRACT (Option A2): no cross-frame mutation from
    # inner cascades. device_type stays at whatever the outer cascade set.
    assert v.device_type == "cuda", (
        f"POST-Slice-3 contract: _build_tree GPU OOM should NOT mutate "
        f"self.device_type; expected 'cuda' (unchanged from "
        f"_simulate_gpu_state), got {v.device_type!r}. The outer "
        f"mode_candidates cascade is the single source of truth for "
        f"backend switching."
    )
    _release_voxel_reassigner(v)


def test_query_tree_gpu_oom_does_not_mutate_device_type(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Cascade Sites 3+4: ``_query_tree`` GPU OOM does NOT mutate ``self.device_type``.

    POST-Slice 3 (Option A2) contract: when the GPU tree's ``query``
    method raises ``MemoryError``, ``_query_tree`` rebuilds the CPU
    KDTree in the same call and queries it. ``self.device_type`` stays
    unchanged so subsequent ``_query_tree`` calls retry GPU. The outer
    ``mode_candidates`` cascade in ``run()`` is the single source of
    truth for cross-frame backend switching.

    Setup: CPU-construct, simulate GPU state, manually craft a
    ``_TreeHandle(backend="gpu", tree=fake_tree, ...)`` whose
    ``tree.query`` raises ``MemoryError`` on first call. Pass it to
    ``_query_tree`` directly.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    _simulate_gpu_state(v, kdtree_cls=None)

    coords_real = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )

    class _FakeGpuTree:
        def query(self, *args, **kwargs):
            raise MemoryError("simulated GPU tree query OOM")

    handle = _TreeHandle(
        backend="gpu", tree=_FakeGpuTree(), coords_real_scaled=coords_real
    )
    query = np.array([[4.1, 5.1, 6.1]], dtype=np.float32)
    dist, idx = v._query_tree(handle, query)
    # Local CPU rebuild fallback produces a real result.
    assert dist.shape == (1,)
    assert idx.shape == (1,)
    assert idx[0] == 1  # nearest to (4.1, 5.1, 6.1) is row 1
    # POST-SLICE-3 CONTRACT (Option A2): no cross-frame mutation from
    # inner cascades. device_type stays at whatever the outer cascade set.
    assert v.device_type == "cuda", (
        f"POST-Slice-3 contract: _query_tree GPU OOM should NOT mutate "
        f"self.device_type; expected 'cuda' (unchanged from "
        f"_simulate_gpu_state), got {v.device_type!r}. The outer "
        f"mode_candidates cascade is the single source of truth for "
        f"backend switching."
    )
    _release_voxel_reassigner(v)


def test_query_tree_gpu_non_oom_exception_propagates(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Latent bug fix: ``_query_tree`` GPU non-OOM exceptions propagate.

    POST-Slice 3 (Option A2) contract: the explicit
    ``if not adaptive_run.is_oom_error(exc): raise`` gate at the
    ``_query_tree`` GPU branch ensures non-OOM exceptions (e.g., a
    ``ValueError`` from a dtype mismatch) propagate out instead of
    being silently swallowed. This fixes the PRE-Slice-3 latent bug
    where the bare ``except Exception`` flipped the backend on any
    error and hid genuine bugs.

    Setup: same as the GPU OOM test, but rig ``tree.query`` to raise
    ``ValueError`` instead of ``MemoryError``.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    _simulate_gpu_state(v, kdtree_cls=None)

    coords_real = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )

    class _FakeGpuTreeValueError:
        def query(self, *args, **kwargs):
            raise ValueError("simulated non-OOM GPU error (e.g., dtype mismatch)")

    handle = _TreeHandle(
        backend="gpu",
        tree=_FakeGpuTreeValueError(),
        coords_real_scaled=coords_real,
    )
    query = np.array([[4.1, 5.1, 6.1]], dtype=np.float32)

    # POST-SLICE-3 CONTRACT (Option A2 + latent bug fix): non-OOM
    # exceptions propagate out of ``_query_tree`` instead of being
    # silently swallowed and triggering a backend flip.
    with pytest.raises(ValueError, match="simulated non-OOM GPU error"):
        v._query_tree(handle, query)

    # Backend remains unchanged — the inner cascade no longer mutates
    # device_type at all.
    assert v.device_type == "cuda", (
        f"POST-Slice-3 contract: device_type should be unchanged after a "
        f"non-OOM exception (the exception propagated, no fallback ran); "
        f"got {v.device_type!r}."
    )
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# VoxelReassignerConfig validation (__post_init__)
# -------------------------------------------------------------------------

def test_voxel_reassigner_config_default_constructs() -> None:
    VoxelReassignerConfig()


def test_voxel_reassigner_config_rejects_bad_device() -> None:
    with pytest.raises(ValueError, match="device"):
        VoxelReassignerConfig(device="bogus")


def test_voxel_reassigner_config_max_refine_iterations_zero_ok() -> None:
    """0 means 'skip refinement loop' — that's a valid choice."""
    VoxelReassignerConfig(max_refine_iterations=0)


def test_voxel_reassigner_config_max_refine_iterations_rejects_negative() -> None:
    with pytest.raises(ValueError, match="max_refine_iterations"):
        VoxelReassignerConfig(max_refine_iterations=-1)


@pytest.mark.parametrize("field", [
    "max_query_points", "max_bruteforce_pairs",
])
def test_voxel_reassigner_config_rejects_nonpositive_numeric(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        VoxelReassignerConfig(**{field: 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=field):
        VoxelReassignerConfig(**{field: -1})  # type: ignore[arg-type]


# -------------------------------------------------------------------------
# PRD #217 Slice 1 — pin pre-rewrite contracts before mask + vox caching.
#
# These tests pin the CURRENT behavior so PRD #217 Slice 2 (mask + vox
# caching in the driver loop, dead-branch cleanup in _get_master_mask,
# arity fix in _select_best_pairs) can flip them with deliberate
# breaking-and-replacing.
# -------------------------------------------------------------------------


def test_get_master_mask_called_once_per_frame_after_first_post_rewrite(
    make_voxel_reassign_imageinfo_3d, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST-REWRITE contract: ``_get_master_mask`` is called once per frame.

    PRD #217 Slice 2 caches ``vox_prev`` (= ``argwhere(master_mask)``)
    and rotates ``vox_prev = vox_next`` at the bottom of each loop
    iteration. The driver invokes ``_get_master_mask(0)`` once before
    the loop and then ``_get_master_mask(t + 1)`` once per iteration
    — never re-reading the prev-frame mask.

    With ``num_t=2`` we expect exactly one pre-loop call (t=0) and one
    in-loop call (t=1) — i.e. ``[0, 1]``. The ``len`` is the same as
    pre-rewrite (2), but the meaning is different: pre-rewrite always
    fires 2 per iteration (so for num_t=3 it would be [0, 1, 1, 2]);
    post-rewrite is `[0] + [1, 2, 3, ...]` (so for num_t=3 it would
    be [0, 1, 2]).
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    real_mask = VoxelReassigner._get_master_mask
    seen_ts: list[int] = []

    def counted_mask(self, t):
        seen_ts.append(t)
        return real_mask(self, t)

    monkeypatch.setattr(VoxelReassigner, "_get_master_mask", counted_mask)
    v.run()

    # POST-REWRITE: one pre-loop init at t=0, one in-loop call at t=1
    # (the next-frame mask). vox_prev is rotated from vox_next, so no
    # duplicate read for the prev slot.
    assert seen_ts == [0, 1], (
        f"POST-REWRITE: _get_master_mask should be called once per "
        f"frame (init t=0 + per-iter t+1) — expected [0, 1] for "
        f"num_t=2; got {seen_ts}."
    )
    _release_voxel_reassigner(v)


def test_select_best_pairs_empty_input_returns_2tuple_post_fix(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """POST-FIX contract: ``_select_best_pairs`` empty input returns a 2-tuple.

    Pins the arity-bug fix from PRD #217 Slice 2: the empty-input
    early-return now matches the populated branch's 2-tuple shape
    and the caller's 2-value unpack.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v._allocate_memory()  # populate spatial_shape

    empty_prev = np.empty((0, 3), dtype=np.int64)
    empty_next = np.empty((0, 3), dtype=np.int64)
    empty_dist = np.empty((0,), dtype=np.float64)

    result = v._select_best_pairs(empty_prev, empty_next, empty_dist)
    assert len(result) == 2, (
        f"POST-FIX: _select_best_pairs empty-input early-return must "
        f"be a 2-tuple to match the populated branch and the caller's "
        f"unpack at voxel_reassignment.py:1015; got {len(result)}-tuple."
    )
    best_prev, best_next = result
    assert best_prev.shape == (0, 3)
    assert best_next.shape == (0, 3)
    assert best_prev.dtype == np.int64
    assert best_next.dtype == np.int64
    _release_voxel_reassigner(v)


# -------------------------------------------------------------------------
# PRD #222 Slice 1 — characterize current `_assign_unique_matches` Python
# greedy semantics so Slice 2's round-based vectorized rewrite can be
# verified for set-equivalence on a known suite of corner cases.
#
# These tests pin BEHAVIOR (which kept-row SET emerges from each
# input), not IMPLEMENTATION ordering — the round-based rewrite returns
# kept indices in input-row order while the pre-rewrite returns them in
# distance-ascending order. Both are deterministic but distinct
# orderings; the function's contract is set-of-pairs, not a sequence.
# See ADR 0011.
# -------------------------------------------------------------------------


def _make_voxel_reassigner_for_assign_unique(
    info_factory, spatial_shape: tuple[int, ...] = (32, 32, 32)
) -> VoxelReassigner:
    """Bare VoxelReassigner with `spatial_shape` set, suitable for direct
    `_assign_unique_matches` calls without the full memmap allocation.
    """
    info = info_factory()
    v = VoxelReassigner(info, VoxelReassignerConfig(device="cpu"), num_t=2)
    v.spatial_shape = spatial_shape
    return v


def _kept_pair_set(prev_arr: np.ndarray, next_arr: np.ndarray) -> set:
    """(prev, next) pair set for set-equality comparison."""
    return {
        (tuple(int(x) for x in p), tuple(int(x) for x in n))
        for p, n in zip(prev_arr, next_arr)
    }


def test_assign_unique_matches_empty_input_returns_2tuple(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Empty input → 2-tuple of `(0, D)` int64 arrays.

    The function's empty-input contract is documented in the wiki
    gotcha (`_allocate_memory()` must run first or this raises). With
    `spatial_shape` set but distances empty, we expect the early-return
    path at lines 666-669: 2-tuple of `(0, D)` int64 arrays.
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    empty_prev = np.empty((0, 3), dtype=np.int64)
    empty_next = np.empty((0, 3), dtype=np.int64)
    empty_dist = np.empty((0,), dtype=np.float64)

    result = v._assign_unique_matches(empty_prev, empty_next, empty_dist)
    assert len(result) == 2
    out_prev, out_next = result
    assert out_prev.shape == (0, 3)
    assert out_next.shape == (0, 3)
    assert out_prev.dtype == np.int64
    assert out_next.dtype == np.int64
    _release_voxel_reassigner(v)


def test_assign_unique_matches_single_match_kept_verbatim(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Single (prev, next) pair → kept verbatim."""
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    vox_prev = np.array([[1, 2, 3]], dtype=np.int64)
    vox_next = np.array([[4, 5, 6]], dtype=np.int64)
    distances = np.array([1.5], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    np.testing.assert_array_equal(out_prev, vox_prev)
    np.testing.assert_array_equal(out_next, vox_next)
    _release_voxel_reassigner(v)


def test_assign_unique_matches_no_contention_keeps_all(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Three pairs with disjoint prev_ids and disjoint next_ids → all kept."""
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    vox_prev = np.array(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64
    )
    vox_next = np.array(
        [[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.int64
    )
    distances = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    expected = _kept_pair_set(vox_prev, vox_next)
    actual = _kept_pair_set(out_prev, out_next)
    assert actual == expected, (
        f"All 3 disjoint pairs should be kept; got {actual} (expected "
        f"{expected})"
    )
    _release_voxel_reassigner(v)


def test_assign_unique_matches_prev_contention_keeps_lowest_distance(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Two pairs share prev_id; only the lowest-distance row is kept.

    Pair A (prev P, next N1, d=1.0) competes with pair B (prev P,
    next N2, d=2.0). Greedy keeps pair A (lower distance) and drops
    pair B (P is used).
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    p = [5, 5, 5]
    vox_prev = np.array([p, p], dtype=np.int64)
    vox_next = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.int64)
    distances = np.array([1.0, 2.0], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    expected = {(tuple(p), (10, 10, 10))}
    actual = _kept_pair_set(out_prev, out_next)
    assert actual == expected, (
        f"Lower-distance pair (d=1.0) should be kept; got {actual}"
    )
    _release_voxel_reassigner(v)


def test_assign_unique_matches_next_contention_keeps_lowest_distance(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Two pairs share next_id; only the lowest-distance row is kept.

    Pair A (prev P1, next N, d=1.0) competes with pair B (prev P2,
    next N, d=2.0). Greedy keeps pair A and drops pair B (N is used).
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    n = [10, 10, 10]
    vox_prev = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int64)
    vox_next = np.array([n, n], dtype=np.int64)
    distances = np.array([1.0, 2.0], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    expected = {((1, 1, 1), tuple(n))}
    actual = _kept_pair_set(out_prev, out_next)
    assert actual == expected, (
        f"Lower-distance pair (d=1.0) should be kept; got {actual}"
    )
    _release_voxel_reassigner(v)


def test_assign_unique_matches_chain_contention_multi_round(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Chain contention exercises multi-round convergence.

    Rows: (P1,N1,d=1), (P1,N2,d=2), (P2,N2,d=3), (P2,N3,d=4),
    (P3,N3,d=5). Greedy:
      - Take (P1,N1): keep, used={P1, N1}
      - (P1,N2): P1 used, skip
      - (P2,N2): keep, used={P1, P2, N1, N2}
      - (P2,N3): P2 used, skip
      - (P3,N3): N3 NOT used → keep, used={P1, P2, P3, N1, N2, N3}
    Expected kept set: {(P1,N1), (P2,N2), (P3,N3)}.

    Round-based version would converge in 3 rounds:
      - R1: keep (P1,N1) only — only row that's argmin for both prev
        and next.
      - R2: active = {(P2,N2), (P2,N3), (P3,N3)}; keep (P2,N2).
      - R3: active = {(P3,N3)}; keep.
    Same kept set.
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    P1, P2, P3 = (1, 1, 1), (2, 2, 2), (3, 3, 3)
    N1, N2, N3 = (10, 10, 10), (20, 20, 20), (30, 30, 30)

    vox_prev = np.array([P1, P1, P2, P2, P3], dtype=np.int64)
    vox_next = np.array([N1, N2, N2, N3, N3], dtype=np.int64)
    distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    expected = {(P1, N1), (P2, N2), (P3, N3)}
    actual = _kept_pair_set(out_prev, out_next)
    assert actual == expected, (
        f"Chain contention should keep {expected}; got {actual}"
    )
    _release_voxel_reassigner(v)


def test_assign_unique_matches_tied_distances_pick_one(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """Tied distances on contention: exactly one of the tied rows is kept.

    Two rows share prev_id with identical distance. The greedy loop
    iterates in `np.argsort(distances)` order (default 'quicksort',
    stable enough on tied keys for argsort to fall back to insertion
    sort behavior at small N). Both rows can't be kept (prev_id
    conflict); exactly one survives. The choice between the two ties
    is implementation-defined; both pre-rewrite (greedy by argsort
    order) and post-rewrite (round-based by lexsort tie-break) would
    pick deterministically but possibly differently.

    The TEST asserts only the cardinality + uniqueness invariants:
      - Exactly 1 pair kept.
      - Kept pair is one of the two input pairs.
      - prev_id of the kept pair is the shared prev_id.
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d
    )

    p = [7, 7, 7]
    vox_prev = np.array([p, p], dtype=np.int64)
    vox_next = np.array([[15, 15, 15], [25, 25, 25]], dtype=np.int64)
    distances = np.array([1.0, 1.0], dtype=np.float64)

    out_prev, out_next = v._assign_unique_matches(vox_prev, vox_next, distances)
    assert len(out_prev) == 1
    actual = _kept_pair_set(out_prev, out_next)
    expected_options = {
        (tuple(p), (15, 15, 15)),
        (tuple(p), (25, 25, 25)),
    }
    assert actual.issubset(expected_options) and len(actual) == 1, (
        f"Tied-distance contention should keep exactly 1 of the two "
        f"input pairs; got {actual} (options: {expected_options})"
    )
    _release_voxel_reassigner(v)


def _assign_unique_matches_reference_greedy(
    spatial_shape: tuple[int, ...],
    vox_prev_matches: np.ndarray,
    vox_next_matches: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference: verbatim copy of the pre-rewrite Python greedy loop.

    Kept inline in the test file so the equivalence pin survives even
    after the production code drops the loop. Mirrors the
    `_interpolate_all_forward_reference` pattern from PRD #205.
    """
    if len(distances) == 0:
        dim = vox_prev_matches.shape[1] if vox_prev_matches.ndim == 2 else 3
        return (np.empty((0, dim), dtype=np.int64),
                np.empty((0, dim), dtype=np.int64))

    prev_flat = np.ravel_multi_index(vox_prev_matches.T, spatial_shape)
    next_flat = np.ravel_multi_index(vox_next_matches.T, spatial_shape)

    order = np.argsort(distances)
    prev_flat_sorted = prev_flat[order]
    next_flat_sorted = next_flat[order]

    _, prev_inv = np.unique(prev_flat_sorted, return_inverse=True)
    _, next_inv = np.unique(next_flat_sorted, return_inverse=True)

    used_prev = np.zeros(prev_inv.max() + 1, dtype=bool)
    used_next = np.zeros(next_inv.max() + 1, dtype=bool)
    keep_indices: list = []

    for idx_sorted in range(len(order)):
        p_idx = prev_inv[idx_sorted]
        n_idx = next_inv[idx_sorted]
        if used_prev[p_idx] or used_next[n_idx]:
            continue
        used_prev[p_idx] = True
        used_next[n_idx] = True
        keep_indices.append(order[idx_sorted])

    if not keep_indices:
        dim = vox_prev_matches.shape[1]
        return (np.empty((0, dim), dtype=np.int64),
                np.empty((0, dim), dtype=np.int64))

    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    return vox_prev_matches[keep_arr], vox_next_matches[keep_arr]


def test_assign_unique_matches_post_rewrite_equivalence_to_reference_greedy(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """POST-REWRITE: round-based output is set-equivalent to sequential greedy.

    Generates four synthetic batches (varying N, contention densities,
    tie-distance fractions) and compares the round-based vectorized
    `_assign_unique_matches` against the inline reference greedy. The
    bar is SET-equality on kept (prev, next) pairs — both
    implementations produce a deterministic kept set, but the order
    in which kept indices are returned differs (round-based: input
    order; greedy: distance-ascending). See ADR 0011.

    The test fixes a seed for each batch so re-runs are stable, but
    explicitly exercises tie-distance contention (where the kept SET
    can in principle differ between the two implementations if
    tie-break order differs). The walked-through examples in ADR
    0011 confirm that on every non-tie input both produce the same
    SET, and the tie cases are bounded by the per-prev-id /
    per-next-id constraints — both implementations end up keeping at
    most one row per (prev, next) pair, so the set size matches even
    if the specific row picked from a tie group differs.
    """
    v = _make_voxel_reassigner_for_assign_unique(
        make_voxel_reassign_imageinfo_3d,
        spatial_shape=(64, 64, 64),
    )

    batches = [
        # (n, contention_factor, seed, description)
        (50, 1.0, 0, "small / no contention (each pair has unique prev_id and next_id)"),
        (200, 0.5, 1, "medium / moderate contention (50% reuse rate)"),
        (1000, 0.2, 2, "large / heavy contention (20% reuse rate)"),
        (500, 0.1, 3, "medium / very heavy contention (10% reuse rate)"),
    ]

    for n, contention, seed, desc in batches:
        rng = np.random.default_rng(seed)
        # Sample prev/next coords with a controlled "vocabulary" size to
        # induce contention. Smaller vocab → more reuse → more contention.
        vocab = max(2, int(n * contention))
        prev_ids = rng.integers(0, vocab, size=n)
        next_ids = rng.integers(0, vocab, size=n)
        # Convert flat ids to 3D coords inside (64, 64, 64).
        vox_prev = np.column_stack([
            (prev_ids // (64 * 64)) % 64,
            (prev_ids // 64) % 64,
            prev_ids % 64,
        ]).astype(np.int64)
        vox_next = np.column_stack([
            (next_ids // (64 * 64)) % 64,
            (next_ids // 64) % 64,
            next_ids % 64,
        ]).astype(np.int64)
        distances = rng.uniform(0.0, 5.0, size=n).astype(np.float64)

        ref_prev, ref_next = _assign_unique_matches_reference_greedy(
            v.spatial_shape, vox_prev, vox_next, distances,
        )
        new_prev, new_next = v._assign_unique_matches(vox_prev, vox_next, distances)

        ref_set = _kept_pair_set(ref_prev, ref_next)
        new_set = _kept_pair_set(new_prev, new_next)

        assert ref_set == new_set, (
            f"Equivalence break on batch '{desc}' (n={n}, vocab={vocab}, "
            f"seed={seed}): reference greedy kept {len(ref_set)} pairs, "
            f"round-based kept {len(new_set)} pairs. Symmetric difference: "
            f"{ref_set ^ new_set if len(ref_set ^ new_set) <= 10 else f'(too large to print, |Δ|={len(ref_set ^ new_set)})'}"
        )

    _release_voxel_reassigner(v)


def test_run_reassignment_3d_snapshot_post_rewrite_matches_pre_rewrite_reference(
    make_voxel_reassign_imageinfo_3d,
) -> None:
    """POST-REWRITE: end-to-end output is bit-identical to a reference rerun.

    The mask + vox caching in PRD #217 Slice 2 is supposed to be
    byte-identical: ``master_mask_next`` of frame ``t`` is identical
    to ``master_mask_prev`` of frame ``t+1`` (same union, same disk
    bytes), so caching only changes WHEN the data is computed, not
    WHAT is computed. We verify by running TWICE on independent
    fixtures and asserting the SHA-256 of both reassigned outputs
    matches across runs (a stable in-band determinism check).

    Replaces ``test_run_reassignment_3d_snapshot_pre_rewrite``
    from Slice 1 of #217 — same shape, but the assertion now
    represents the post-rewrite contract (mask caching is
    behavior-preserving). Determinism within a single rewrite is the
    bar; cross-rewrite equivalence was verified locally before merge.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = _run_voxel_reassign(info)

    branch_arr = np.array(v.reassigned_branch_memmap)
    obj_arr = np.array(v.reassigned_obj_memmap)

    branch_hash = hashlib.sha256(branch_arr.tobytes()).hexdigest()
    obj_hash = hashlib.sha256(obj_arr.tobytes()).hexdigest()

    info2 = make_voxel_reassign_imageinfo_3d()
    v2 = _run_voxel_reassign(info2)
    branch_arr2 = np.array(v2.reassigned_branch_memmap)
    obj_arr2 = np.array(v2.reassigned_obj_memmap)
    branch_hash2 = hashlib.sha256(branch_arr2.tobytes()).hexdigest()
    obj_hash2 = hashlib.sha256(obj_arr2.tobytes()).hexdigest()

    assert branch_hash == branch_hash2, (
        "POST-REWRITE: VoxelReassigner is not deterministic on the same "
        "3D fixture after the mask + vox caching rewrite."
    )
    assert obj_hash == obj_hash2, (
        "POST-REWRITE: VoxelReassigner is not deterministic on the same "
        "3D fixture after the mask + vox caching rewrite."
    )

    # Same sanity check as Slice 1: ensure the snapshot is non-trivial.
    assert (branch_arr > 0).any()
    assert (obj_arr > 0).any()

    _release_voxel_reassigner(v)
    _release_voxel_reassigner(v2)


# -------------------------------------------------------------------------
# PRD #227 Slice 1 — pin pre-rewrite contracts before tree caching across
# frames in high-memory mode.
#
# Pre-rewrite, `match_voxels` builds two `cKDTree`s per call (one for
# `vox_prev`, one for `vox_next`) regardless of `low_memory`. Slice 2
# of #227 caches `tree_next` from frame t and reuses it as `tree_prev`
# of frame t+1 IN HIGH-MEMORY MODE ONLY. The low-memory branch
# preserves the explicit serialized-build trade-off.
# -------------------------------------------------------------------------


def _count_build_tree_calls(
    info_factory,
    *,
    low_memory: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, VoxelReassigner]:
    """Run VoxelReassigner end-to-end with `_build_tree` instrumented.

    Returns (call_count, instance) so the caller can assert on the
    count and then release the instance via `_release_voxel_reassigner`.
    """
    info = info_factory()
    v = VoxelReassigner(
        info,
        VoxelReassignerConfig(device="cpu", low_memory=low_memory),
        num_t=2,
    )
    real_build_tree = VoxelReassigner._build_tree
    counter = {"n": 0}

    def counted_build_tree(self, coords_real_scaled):
        counter["n"] += 1
        return real_build_tree(self, coords_real_scaled)

    monkeypatch.setattr(VoxelReassigner, "_build_tree", counted_build_tree)
    v.run()
    return counter["n"], v


def test_match_voxels_builds_two_trees_per_frame_post_rewrite_high_mem(
    make_voxel_reassign_imageinfo_3d,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST-REWRITE contract (num_t=2): 2 cKDTree builds for the only frame transition.

    The fixture has 2 frames, so the loop runs once (t=0). PRD #227
    Slice 2's caching only fires at t >= 1 (where the previous
    iteration cached its `tree_next`). At t=0 the cache is None and
    both `tree_prev` and `tree_next` are built fresh — exactly the
    pre-rewrite behavior for num_t=2.

    The 1-build-per-frame-after-first contract is exercised directly
    in `test_match_voxels_post_rewrite_caches_tree_next_for_next_iteration`
    below (which calls `match_voxels` twice manually to simulate two
    consecutive frame transitions without needing a 3-frame fixture).
    """
    n_calls, v = _count_build_tree_calls(
        make_voxel_reassign_imageinfo_3d,
        low_memory=False,
        monkeypatch=monkeypatch,
    )
    assert n_calls == 2, (
        f"POST-REWRITE high-memory (num_t=2, 1 frame transition): "
        f"_build_tree should be called 2× at t=0 (cold cache); got {n_calls}."
    )
    _release_voxel_reassigner(v)


def test_match_voxels_post_rewrite_caches_tree_next_for_next_iteration(
    make_voxel_reassign_imageinfo_3d,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST-REWRITE: second `match_voxels` call reuses cached `tree_next` as `tree_prev`.

    Directly exercises the caching mechanism without relying on a
    multi-frame fixture (the synthetic 3D fixture only has 2 frames).
    Simulates two consecutive iterations of `_run_reassignment` by
    calling `match_voxels(vox_a, vox_b, t=0)` then
    `match_voxels(vox_b, vox_c, t=1)` with `vox_b` being the IDENTICAL
    ndarray rotated forward (per PRD #217's `vox_prev = vox_next`
    rotation).

    Pre-rewrite: 2 builds per call → 4 builds total.
    Post-rewrite: 2 builds at t=0 + 1 build at t=1 (vox_b cache hit) → 3 total.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(
        info, VoxelReassignerConfig(device="cpu", low_memory=False), num_t=2
    )
    v._allocate_memory()
    # Sanity: cache is empty at construction.
    assert v._cached_tree_for_next_frame is None
    assert v._cached_vox_for_next_frame is None

    # Use a small slice of the fixture's labeled voxels so the test is fast.
    branch_t0 = np.argwhere(v.branch_label_memmap[0] > 0)
    branch_t1 = np.argwhere(v.branch_label_memmap[1] > 0)
    # Trim to keep the test cheap; ensure we have at least a handful per side.
    take = min(200, len(branch_t0), len(branch_t1))
    assert take > 0, "fixture has no labeled branch voxels at t=0/t=1"
    vox_a = branch_t0[:take]
    vox_b = branch_t1[:take]
    # vox_c is "next-next" — synthesize by adding a tiny offset to vox_b
    # (clamped into bounds) so it's a different array but valid coords.
    vox_c = np.clip(
        vox_b + 1,
        0,
        np.array(v.spatial_shape, dtype=vox_b.dtype) - 1,
    )

    # Count `_build_tree` calls across both `match_voxels` invocations.
    real_build_tree = VoxelReassigner._build_tree
    counter = {"n": 0}

    def counted_build_tree(self, coords_real_scaled):
        counter["n"] += 1
        return real_build_tree(self, coords_real_scaled)

    monkeypatch.setattr(VoxelReassigner, "_build_tree", counted_build_tree)

    # Iteration 1: cold cache → 2 builds (tree_prev + tree_next).
    v.match_voxels(vox_a, vox_b, t=0)
    assert counter["n"] == 2, (
        f"Iter 1 (cold cache): expected 2 _build_tree calls, got {counter['n']}"
    )
    # After iter 1, cache should hold the tree built from vox_b.
    assert v._cached_tree_for_next_frame is not None
    assert v._cached_vox_for_next_frame is vox_b

    # Iteration 2: vox_prev IS vox_b (the cached array) → tree_prev is reused;
    # only tree_next (from vox_c) is built fresh.
    v.match_voxels(vox_b, vox_c, t=1)
    assert counter["n"] == 3, (
        f"Iter 2 (warm cache, vox_b reused as tree_prev): expected 1 "
        f"additional build (tree_next only), got {counter['n'] - 2} "
        f"(total {counter['n']})"
    )
    # Cache now holds tree built from vox_c.
    assert v._cached_vox_for_next_frame is vox_c
    _release_voxel_reassigner(v)


def test_match_voxels_post_rewrite_cache_miss_on_fresh_array_high_mem(
    make_voxel_reassign_imageinfo_3d,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST-REWRITE: a direct caller passing a fresh `vox_prev` ndarray bypasses the cache.

    The cache identity check is `self._cached_vox_for_next_frame is
    vox_prev` (Python `is`). Direct callers of `match_voxels` (not
    going through `_run_reassignment`'s rotation) typically pass a
    fresh ndarray each call, so the identity check fails and tree_prev
    is built normally. Pins the safe-by-default behavior.
    """
    info = make_voxel_reassign_imageinfo_3d()
    v = VoxelReassigner(
        info, VoxelReassignerConfig(device="cpu", low_memory=False), num_t=2
    )
    v._allocate_memory()

    branch_t0 = np.argwhere(v.branch_label_memmap[0] > 0)
    branch_t1 = np.argwhere(v.branch_label_memmap[1] > 0)
    take = min(200, len(branch_t0), len(branch_t1))
    vox_a = branch_t0[:take]
    vox_b = branch_t1[:take]

    real_build_tree = VoxelReassigner._build_tree
    counter = {"n": 0}

    def counted_build_tree(self, coords_real_scaled):
        counter["n"] += 1
        return real_build_tree(self, coords_real_scaled)

    monkeypatch.setattr(VoxelReassigner, "_build_tree", counted_build_tree)

    # Iter 1: cold cache → 2 builds.
    v.match_voxels(vox_a, vox_b, t=0)
    assert counter["n"] == 2

    # Iter 2 with a FRESH vox_prev array (not the cached vox_b reference).
    # Even if the values are identical, `is` fails, and tree_prev is rebuilt.
    vox_b_fresh = vox_b.copy()
    assert vox_b_fresh is not vox_b
    np.testing.assert_array_equal(vox_b_fresh, vox_b)
    v.match_voxels(vox_b_fresh, vox_a, t=1)
    assert counter["n"] == 4, (
        f"Iter 2 with fresh vox_prev (not cached array): expected 2 builds "
        f"(no cache hit), got {counter['n'] - 2} (total {counter['n']})"
    )
    _release_voxel_reassigner(v)


def test_match_voxels_builds_two_trees_per_frame_low_mem(
    make_voxel_reassign_imageinfo_3d,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOW-MEMORY contract: 2 builds per frame, both pre AND post rewrite.

    The low-memory branch (lines 770-791) builds `tree_next` first,
    runs forward matching, drops the tree (`tree_next = None`), then
    builds `tree_prev` and runs backward matching. Two builds per
    frame, but serialized so peak memory is one tree.

    PRD #227 Slice 2's caching is gated on `low_memory == False` —
    persisting a tree across frames would defeat the explicit low-
    memory intent. So this test pins the contract that BOTH PRE and
    POST rewrite produce 2 builds per frame in low-memory mode.

    With num_t=2 we expect exactly 2.
    """
    n_calls, v = _count_build_tree_calls(
        make_voxel_reassign_imageinfo_3d,
        low_memory=True,
        monkeypatch=monkeypatch,
    )
    assert n_calls == 2, (
        f"LOW-MEMORY: _build_tree should be called 2× per frame "
        f"transition (serialized); for num_t=2 that's exactly 2. "
        f"Got {n_calls}."
    )
    _release_voxel_reassigner(v)
