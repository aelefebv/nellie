---
created: 2026-05-08
modified: 2026-05-08
---

# Dechaos scan — `nellie/tracking/voxel_reassignment.py` (`VoxelReassigner`)

One-shot review of the next pipeline stage to receive the test+hoist treatment. Findings are intended to feed an upcoming three-slice PRD that mirrors PRDs #77 (Network), #84 (Markers), and #91 (Hu). Durable items should be folded into [[voxel-reassignment]] gotchas during/after the slice PRs.

**Reference templates**: Markers (`nellie/segmentation/mocap_marking.py`, post-#90) and Hu (`nellie/tracking/hu_tracking.py`, post-#97) are the closest mirrors — same backend pattern, same outer `adaptive_run.mode_candidates` cascade. Use them as the after-picture. The wiki [[voxel-reassignment]] article already flags the half-hoisted backend helpers and the cross-frame `_switch_to_cpu` mutation as gotchas; this scan operationalizes those into a refactor plan.

---

## Pass 1 — System Map

- **Stage 6 of 7** in `nellie/run.py` (line 105). Two in-tree call sites: `run.py:105` (`VoxelReassigner(im_info, device=device)` — no `low_memory`, no `num_t`) and `nellie_napari/nellie_processor.py:485` (`**step_kwargs` from `get_reassign_params()`, which exposes 7 fields: `num_t`, `store_running_matches`, `max_refine_iterations`, `device`, `low_memory`, `max_query_points`, `max_bruteforce_pairs`).
- **Class**: `VoxelReassigner`, single class plus a `_TreeHandle` dataclass in `nellie/tracking/voxel_reassignment.py`, 1117 lines.
- **Inputs (memmaps via `ImInfo.pipeline_paths`)**: `im_skel_relabelled` (from Network), `im_instance_label` (from Label), plus an indirect dependency on `flow_vector_array.npy` (from Hu, consumed via `FlowInterpolator`). All three converge here. The raw image is also accessed via `FlowInterpolator._allocate_memory` (`flow_interpolation.py:121`).
- **Outputs (memmaps + npy)**:
  - `im_branch_label_reassigned` (memmap, int32)
  - `im_obj_label_reassigned` (memmap, int32)
  - `voxel_matches.npy` (np.save'd object array of `[best_prev, best_next]` pairs per frame, only when `store_running_matches=True`)
- **External deps**: `nellie.utils.adaptive_run` (canonical backend helpers), `numpy`, `scipy.spatial.cKDTree`, optionally `cupy` + `cupyx.scipy.spatial.cKDTree` (NOT `cupyx.scipy.ndimage` — this stage doesn't use ndimage), `nellie.tracking.flow_interpolation.FlowInterpolator`, `nellie.im_info.verifier.ImInfo`.
- **Tests**: none. `tests/test_voxel_reassignment.py` does not exist; deleted in the May 2026 scaffold rebuild and not restored. Wiki ([[now]] watch list) confirms tracking modules are unpinned.
- **Architectural shape**: per-frame outer loop (`_run_reassignment`, 994–1062) → per-frame `match_voxels` (758–843) builds 0–2 trees and runs forward + backward `FlowInterpolator.interpolate_coord` → `_match_voxels_to_centroids` (622–660) does nearest-neighbor query against a `_TreeHandle` → `_vote_assign_labels_for_frame` (907–988) assigns labels via inverse-distance weighted vote. Outer `run()` (1064–1112) uses `adaptive_run.mode_candidates` for `(dev, low_memory)` retries. **Three nested layers of inner adaptive degradation on top of that** — see Pass 3.

## Pass 2 — Boundary Scan

| # | Mixed concern | Where | Suggested seam |
|---|---|---|---|
| 1 | Local backend resolution duplicates `adaptive_run` | `_resolve_backend` (137–151), `_try_import_cupy` (153–179), `_is_oom_error` (181–191), `_free_gpu_memory` (193–199), `_switch_to_cpu` (201–210) | Delete; call `adaptive_run.resolve_backend` / canonical helpers. Same change Network's PR #83, Markers' PR #90, Hu's PR #97 made. **Caveat**: `_try_import_cupy` here returns `(cupy, cupyx.scipy.spatial.cKDTree)` not `(cupy, cupyx.scipy.ndimage)` — the canonical `adaptive_run.try_import_cupy` returns the latter. Keep a small local `_get_gpu_kdtree_cls()` helper to fetch the spatial KDTree class after `adaptive_run.resolve_backend` returns. See Pass 8 / Slice 3. |
| 2 | `self._cp` attribute is fully redundant with `self.xp` on GPU | 80, 144, 150, 194–197, 209, 215, 241, 244, 282, 284–285, 296, 332 | When `self.device_type == "cuda"`, `self._cp == self.xp`. Drop the attribute; replace ~12 callsites with `self.xp` (or local `cp = self.xp` for readability). Same shape as Markers' `self.use_gpu` cleanup (PR #89) and Hu's `self._on_gpu` cleanup (PR #96). |
| 3 | Dev driver in module body | `__main__` block (1115–1117) — already a no-op `logger.info` pointer | Delete (Network/Markers/Hu all did the same in their cleanup slices). |
| 4 | Constructor init-block duplication between `no_t` early-return path and main path | 84–101 vs 103–131. ~14 lines of `self.X = None` repeated almost verbatim | Optional cosmetic cleanup; defer if Slice 2 gets too heavy. The placeholder values in the `no_t` path are dead anyway (the `run()` method early-returns on `no_t` at line 1074, before any of these are accessed). |
| 5 | Per-call inner OOM cascade with cross-frame backend mutation | `_build_tree` (237–268), `_query_tree` (270–317) — 4 `_switch_to_cpu` callsites total (250, 252, 290, 309); persist across all later frames | See Pass 3 / Pass 8. **THE central architectural decision for Slice 3.** |
| 6 | Per-chunk OOM cascade (clean local-only fallback) | `_query_bruteforce_gpu` (340–360), `_query_bruteforce_cpu` (377–391) — chunk_size is halved on OOM, no `self.*` mutation | Keep as-is. This is the *good* per-call OOM pattern — fully local, no cross-frame state. Just route `self._is_oom_error` → `adaptive_run.is_oom_error` and `self._free_gpu_memory()` → `adaptive_run.free_gpu_memory(self.xp)`. |
| 7 | I/O mixed with dispatch in main loop | `_run_reassignment` (994–1062) accumulates `running_matches` in memory and `np.save`'s a single `.npy` file at the end (line 1062) | Keep — single-write boundary. Same shape as Hu's stage-end `np.save` of `flow_vector_array`. |

## Pass 3 — Responsibility Scan

The class wears multiple hats — backend management, KDTree gating across 4 backends (gpu, gpu_bruteforce, cpu, cpu_bruteforce), forward/backward flow matching, vote-based label assignment, pipeline orchestration. Most hats are inherent to the algorithm and the "fat class" pattern is consistent with Filter/Label/Network/Markers/Hu. Cleanups below are **dead code, redundant attributes, and architectural duplication — not split candidates**.

| Item | Location | Status | Action |
|---|---|---|---|
| `_get_t` | 849–857 | Dead. `__init__` (103–105) already resolves `self.num_t`; the `if self.num_t is None` guard is unreachable because both `__init__` and `run()` early-return on `self.im_info.no_t`. The method is also called once from `_run_reassignment` (995), where `self.num_t` is already set. | Delete. (Network deleted analogue in #82; Markers in #89; Hu in #96.) |
| `self.shape = None`, `self.spatial_shape = None`, `self.match_coord_dtype = None` | 96–98, 129–131 | All three overwritten in `_allocate_memory` (868–871). | Drop the init lines from main path; keep in `no_t` early-return path only if external code reads them (audit). |
| `self.debug = None` | 94, 120 | Never read, never set elsewhere. | Drop. (Same pattern as Network #82, Markers #89, Hu #96.) |
| `self._cp` attribute | 80, 209, 215 + ~10 read sites | Redundant — equals `self.xp` when `device_type == "cuda"`, `None` when `"cpu"`. Tracked alongside backend at 4 sites (`_resolve_backend`, `_set_backend`, `_switch_to_cpu`, `_try_import_cupy`). | Drop; use `self.xp` directly when on GPU; replace `self._cp is None` checks with `self.device_type != "cuda"`. ~12 callsites. |
| `_build_tree` inner OOM/error cascade | 237–268. Two `_switch_to_cpu` calls (lines 250, 252) — one for OOM, one for any other GPU exception. **Mutates `self.xp` / `self.device_type` / `self._cp` / `self._gpu_kdtree_cls` in place** via `_switch_to_cpu()`; persists across all later frames. | Same wiki-flagged footgun ("`_switch_to_cpu` calls persist across frames"). | See Pass 8 / Slice 3. |
| `_query_tree` GPU inner OOM cascade | 280–293. `_switch_to_cpu` at line 290 runs on **any** GPU exception (not just OOM — the `if self._is_oom_error(exc)` only gates `_free_gpu_memory()`, not `_switch_to_cpu`). | Latent silent-fallback bug: a non-OOM GPU error (e.g., array dtype mismatch) flips the backend permanently with only a `logger.warning` to show for it. | See Pass 8 / Slice 3. Same characterization gap as Hu Cascade A. |
| `_query_tree` GPU bruteforce inner OOM cascade | 295–312. `_switch_to_cpu` at line 309 (same shape — runs on any exception, not just OOM). | Same footgun as the GPU branch above. | See Pass 8 / Slice 3. |
| `_query_bruteforce_gpu` chunk-level OOM | 354–360. Halves `chunk_size` and retries; raises if `chunk_size <= 1`. **No `self.*` mutation.** | Clean per-call adaptive pattern. | Keep; just route through canonical `adaptive_run.is_oom_error` / `free_gpu_memory`. |
| `_query_bruteforce_cpu` chunk-level OOM | 388–391. Same pattern, CPU `MemoryError`. | Clean. | Keep. |
| `_warned_gpu_fallback` flag | 81, 204–206, 216 | Tied to `_switch_to_cpu`; reset by `_set_backend`. Once-per-instance log rate-limit. | If Slice 3 deletes `_switch_to_cpu`, the flag still has a use (rate-limit per-call fallback warnings). Move the `if not self._warned_gpu_fallback: logger.warning(...); self._warned_gpu_fallback = True` block to a tiny helper (e.g., `_warn_gpu_fallback(reason)`) called from each fallback site. |

The wiki gotchas already flag the cross-frame mutation ("`_switch_to_cpu` calls persist across frames"). The dechaos finding refines this: the cascade has **three layers** — outer (`run()` `mode_candidates`), inner-build (`_build_tree`), inner-query (`_query_tree`). The inner-build and inner-query layers BOTH mutate device state, and `_query_tree` does so on **any** exception (not just OOM), which is bug-adjacent.

## Pass 4 — Dependency Scan

| # | Type | Issue | Impact |
|---|---|---|---|
| 1 | **No constructor dual-API** (matches Hu post-#97) | 8 named kwargs (`num_t`, `viewer`, `store_running_matches`, `max_refine_iterations`, `device`, `low_memory`, `max_query_points`, `max_bruteforce_pairs`) with sensible defaults. Napari widget (`nellie_settings.py:299–312`, `get_reassign_params` at 711–719) maps 1:1 to constructor args. **No `prefer_gpu`-style overlap.** | **No pre-slice decision needed**, unlike Markers PRD #84. |
| 2 | `device` accepts undocumented `"cuda"` alias | Constructor docstring says `{"auto","cpu","gpu"}` (61); `_resolve_backend` (139) silently accepts `"cuda"` too. | `adaptive_run.normalize_device` is the canonical source; `_set_backend` already routes through it (213). Backend hoist (Slice 3) makes the constructor go through it too, fixing the gap. |
| 3 | Constructor stores raw `self.device` before normalization | line 71 `self.device = device`; `_resolve_backend` re-normalizes inside (138). Markers (post-#90) and Hu (post-#97) both normalize once via `adaptive_run.normalize_device` at constructor entry. | Slice 3 normalizes via `adaptive_run.normalize_device(device)` at constructor entry, matching Network/Markers/Hu. |
| 4 | `FlowInterpolator` is constructed unconditionally even on no_t | 86–87: in `no_t` path, `self.flow_interpolator_fw = None`. So actually it's NOT constructed on no_t — that's correct. False alarm. | None. |
| 5 | `low_memory` re-clamps `max_query_points` and `max_bruteforce_pairs` in TWO places | constructor (77–79) and `_set_low_memory` (218–224) | Acceptable — `_base_*` attributes hold the originals so re-clamping is idempotent. Pin the clamping behavior in tests. |
| 6 | Outer `run()` re-normalizes device but `_set_backend` re-normalizes again | `run()` line 1077 (`adaptive_run.normalize_device(self.device)`); `_set_backend` line 213 also normalizes the per-iteration `dev` from `mode_candidates`. | Belt-and-suspenders; not a bug. |
| 7 | Hidden FlowInterpolator dependency on `flow_vector_array.npy` | `FlowInterpolator._initialize` (305) → `_allocate_memory` (113–125) → `np.load` of `flow_vector_array.npy`. **VoxelReassigner instantiates two FlowInterpolators in `__init__` (108–109)**, which means upstream Hu must have completed BEFORE `VoxelReassigner(im_info)` is even called — not just before `.run()`. | Tests must materialize `flow_vector_array.npy` before constructing. Affects fixture ordering. Document in test docstring. |
| 8 | `match_coord_dtype = None` in `no_t` early-return + populated by `_select_match_coord_dtype()` | The `no_t` path (98) sets it to None and returns; the main path defers to `_allocate_memory` (871). Internal callers (`_run_reassignment` line 1041) defensively `or np.uint16` it. | Acceptable — defensive `or np.uint16` covers the gap. Pin in test. |

## Pass 5 — Contract Scan

Pinnable invariants for Slice 1 tests (no test pins any of these today):

**Output schema**

- `im_branch_label_reassigned`: int32 memmap, same shape as `im_skel_relabelled`. Values ⊆ {0, …, max_branch_id}.
- `im_obj_label_reassigned`: int32 memmap, same shape as `im_instance_label`. Values ⊆ {0, …, max_obj_id}.
- `voxel_matches.npy`: object-dtype numpy array of length `num_t - 1` when `store_running_matches=True`; each entry is `[best_prev, best_next]` where each is a `(K, D)` array of `match_coord_dtype` (uint16/uint32/uint64 based on `max(spatial_shape)`).
- File NOT written when `store_running_matches=False`. **Pin both branches** so refactors don't silently start writing.

**Initialization (t=0)**

- `reassigned_branch_memmap[0] == branch_label_memmap[0]` and `reassigned_obj_memmap[0] == obj_label_memmap[0]` only at non-background voxels (`> 0`). Background stays 0. Pin both label types.

**Vote semantics**

- `_vote_assign_labels_for_frame` (907–988): only target voxels with non-zero label in `label_memmap[t+1]` ever get a reassigned label (940). Reassigned voxels at t+1 inherit a non-zero label from a non-zero source at t. Pin: a target voxel that's background in `label_memmap[t+1]` stays 0 in the reassigned output regardless of how many candidate matches point to it.
- `max_refine_iterations` controls how many vote rounds happen per frame pair (953). Setting `max_refine_iterations=1` produces fewer assignments than the default 3 on dense overlapping flows. Pin one per-iteration count change.

**4-tier tree backend**

- `_build_tree` returns a `_TreeHandle` whose `.backend` is one of `{"cpu", "cpu_bruteforce", "gpu", "gpu_bruteforce"}`.
- Empty input → `backend="cpu"`, `tree=None`, `coords_real_scaled=None` (line 239). Pin: `_query_tree` on this handle returns `(empty float32, empty int64)`.
- CPU normal path (`device_type == "cpu"`): builds `cKDTree(coords_real_scaled)` (261); falls to `cpu_bruteforce` on `MemoryError` (262–268). **Pin both paths** by monkeypatching `cKDTree`.
- `_can_use_bruteforce` (319–324): rejects when `n_real * n_query > max_bruteforce_pairs` — falls through to CPU KDTree.

**Distance & weighting**

- `_compute_error_distance` (405–410): physical units (`coords * self.flow_interpolator_fw.scaling`).
- `_distance_threshold` (720–756): drops matches with physical distance ≥ `flow_interpolator_fw.max_distance_um` (744–745).
- `_vote_targets` (429–467): weight is `1.0 / (distance + 1e-6)` (438) — **inverse-distance with epsilon floor**. Two-stage `lexsort` over `ravel_multi_index`-flattened coords. Pin one minimal numeric example so refactors of the lexsort don't silently change winner selection.

**`match_coord_dtype` selection** (`_select_match_coord_dtype` 395–403)

- `max(spatial_shape) <= 65 536` → `uint16`.
- `<= 4_294_967_296` → `uint32`.
- else → `uint64`.
- Wiki gotcha: bumping image size past 65 535 in any axis silently widens the saved `.npy`. Pin the threshold in test (synthetic shape with axis = 65 537 → uint32 saved).

**Empty-input edge cases**

- Frame pair where either `master_mask_prev` or `master_mask_next` is empty → loop breaks (1027–1029); all subsequent frame pairs unreassigned. Pin: a single empty frame stops the run; later frames stay zero.
- Frame pair with no valid matches → loop breaks (1032–1034). Same termination semantics.
- `match_voxels` returns empty if either `vox_prev` or `vox_next` is empty (778–781). Pin shape `(0, D)` int64 + `(0,)` float64.

**Backend semantics**

- `device='cpu'` start: `_TreeHandle.backend ∈ {"cpu", "cpu_bruteforce"}`. `self.device_type == "cpu"` throughout. **No fallback events.** Pin.
- `device='gpu'` start (skip-if-no-cupy): `self.device_type == "cuda"`, `_TreeHandle.backend ∈ {"gpu", "gpu_bruteforce", "cpu", "cpu_bruteforce"}` depending on cascade. **Pin** the post-run `self.device_type` (today: mutates to `"cpu"` after first GPU OOM; under Option A2 below: stays `"cuda"`).

**`no_t` short-circuit**

- `im_info.no_t == True` → `__init__` (84–101) sets `flow_interpolator_fw/bw = None`, `running_matches = []`, `branch_label_memmap = None`, etc. `run()` (1074) early-returns with a logger info; no files written. Pin both: file-not-created and run-returns-cleanly.

**Inputs not mutated**

- `im_skel_relabelled`, `im_instance_label`, `flow_vector_array.npy` unchanged after `run()` (compare hashes pre/post). Same pattern as Hu Slice 1.

## Pass 6 — Composability Scan

Mostly **already factored** — `nellie.utils.adaptive_run` is the canonical backend primitive. Findings:

- The 4-tier tree-backend pattern (`_TreeHandle` + `_build_tree` + `_query_tree` + `_can_use_bruteforce`) is unique to this stage; no extraction warranted. The wiki documents the cascade clearly.
- `_vote_targets` (429–467), `_assign_unique_matches` (662–718), `_select_best_pairs` (412–427) are clean reusable lexsort-over-`ravel_multi_index` primitives. Single consumer today; **no extraction warranted** — defer until a second consumer materializes (the Hierarchy stage may want similar patterns; revisit then).
- `_match_forward` / `_match_backward` (473–620) share enough structure to potentially fold into one helper with a `direction` flag. **Don't** — readability of forward-vs-backward reads better as two named methods, even with some duplication. Defer indefinitely.
- The outer `run()` cascade pattern (resolve device order → `mode_candidates` → try / `is_gpu_unavailable_error` / `is_oom_error` / log / continue) is **now repeated nearly verbatim across Filter/Label/Network/Markers/Hu/VoxelReassigner** — about 25 lines of structural sameness per stage. Cross-stage extraction candidate, deferred per queue policy until all 4 untested stages reach the same shape (Hierarchy is the last one).
- The three inner OOM cascades (`_build_tree`, `_query_tree` GPU, `_query_tree` GPU bruteforce) duplicate the outer cascade's error-classification logic. See Pass 3 / Pass 8.

## Pass 7 — Testability Scan

**Current state**: zero tests (test file deleted in scaffold rebuild). Slice 1 fills this gap before Slices 2–3 touch structure.

**Fixture cascade extension** (`tests/conftest.py`): the cascade is currently Filter (session) → Label (session) → Network/Markers (per-test factories) → Hu (per-test factory + Markers session cache from Hu's PR #95). VoxelReassigner needs the **outputs of BOTH Network AND Hu** (`im_skel_relabelled` from Network, `flow_vector_array.npy` from Hu) PLUS the existing Label memmap. **Add**:

- `network_3d_path` / `network_2d_path` session-scoped fixtures (run Filter+Label+Network once per session, return `Path` to `im_skel_relabelled` memmap). Pattern mirrors `label_*_path` (254–281) and `markers_*_paths` (525–559).
- `hu_outputs_3d_path` / `hu_outputs_2d_path` session-scoped fixtures (run Filter+Label+Markers+Hu once per session, return `Path` to `flow_vector_array.npy`). Heaviest new fixture — full upstream pipeline.
- `_make_voxel_reassign_imageinfo_factory` that copies in 3 memmap files: `im_instance_label` (from label cache), `im_skel_relabelled` (from network cache, NEW), and `flow_vector_array.npy` (from hu cache, NEW). Mirrors `_make_hu_imageinfo_factory` (568–616) shape, just with different keys.
- Per-test (`make_voxel_reassign_imageinfo_2d/3d`) and module-scoped (`_module` variants) factories.

Estimated ~150–180 added lines to conftest. **Heavier than Hu's Slice 1** because Hu only added a Markers session cache (one new upstream stage); VoxelReassigner adds Network AND Hu session caches (two new upstream stages). The Hu session cache already exists upstream-wise (markers cache) but doesn't currently materialize the `flow_vector_array.npy` output.

**Test plan for Slice 1** (target ~20 tests, mirroring `test_hu_tracking.py` and `test_mocap_marking.py`'s shape):

```text
Smoke / shape:
- 2D + 3D: run end-to-end on the yeast fixtures with device='cpu', assert
  im_branch_label_reassigned and im_obj_label_reassigned exist with
  expected shapes.
- num_t default vs explicit override.
- store_running_matches=True writes voxel_matches.npy; False does not.
- no_t fixture → run() early-returns; no files written.

Output contract:
- 2D + 3D: reassigned memmaps have dtype int32, shape == input label shape.
- voxel_matches.npy is an object array of length (num_t - 1) when written;
  each entry is [best_prev (K,D), best_next (K,D)] of match_coord_dtype.
- match_coord_dtype: synthetic shape with max axis 65_537 → uint32 saved.
- Initialization invariant: reassigned[0] == input_label[0] at non-zero
  voxels (both branch and obj label types).

Vote / assignment behavior:
- Reassigned voxels at t+1 are a strict subset of label_memmap[t+1] > 0.
  No "phantom labels" at background voxels.
- max_refine_iterations=1 vs default 3: assert iter=1 produces fewer
  assignments on overlapping flows.
- Empty mask at frame N: loop breaks; frames N..end stay all-zero.

4-tier tree backend (CPU only — GPU paths skipped if cupy unavailable):
- device='cpu' end-to-end: post-run `self.device_type == "cpu"` always.
- Monkeypatch cKDTree to raise MemoryError → cpu_bruteforce path engages
  (assert via tree-handle backend on _build_tree return); end-to-end
  result equivalence with default path (within tolerance).
- Empty input → tree_handle.backend == "cpu", tree=None;
  _query_tree returns empty arrays.

Distance / weighting characterization:
- _vote_targets minimal numeric example: 3 sources voting on 1 target,
  one weight strictly highest → that source label wins.
- _distance_threshold drops matches ≥ flow_interpolator.max_distance_um.
- _select_best_pairs returns 1-best per target (lexsort behavior).

Architecture characterization (pin BEFORE Slice 3 changes them):
- Inner _build_tree GPU OOM cascade: monkeypatch _gpu_kdtree_cls to
  raise MemoryError → assert post-run `self.device_type == "cpu"`
  (cross-frame mutation). [Skip if no cupy; characterize via fault
  injection on the resolved-backend state instead.]
- Inner _query_tree GPU OOM cascade: monkeypatch GPU query to raise
  MemoryError → assert post-run `self.device_type == "cpu"`.
- _query_tree silent fallback on non-OOM exceptions: monkeypatch GPU
  query to raise ValueError → assert post-run `self.device_type == "cpu"`
  (current bug-adjacent behavior; Slice 3 will change this).

Inputs not mutated:
- raw + im_skel_relabelled + im_instance_label + flow_vector_array.npy
  unchanged after run() (compare hashes pre/post).

Viewer status callback:
- viewer=None: no-op.
- viewer with .status property: assert per-frame status string written
  ("Reassigning voxels. Frame: N of M.").
```

**Hard-to-test on CPU-only CI**: the inner GPU OOM cascades require either a GPU runtime or fault injection on the post-resolve `_gpu_kdtree_cls` / `_TreeHandle` state. Keep characterization minimal: focus on assertions that work on CPU (e.g., `self.device_type` after a forced CPU-side `_switch_to_cpu` simulation), so Slice 3 can intentionally change the contract and the test fails loudly OR passes with updated assertion.

**Heaviest part of Slice 1**: the `hu_outputs_*_path` session fixture. Filter + Label + Markers + Hu on a 2-frame yeast volume should run in ~60–90s combined; session caching mitigates per-test cost. Risk: if Hu runs slower than expected on CPU, sliding scope to a smaller subsample is an option. The Hu session cache may be the single longest pre-test setup in the suite once this lands.

## Pass 8 — Refactor Sequencing

Mirror PRDs #77 (Network), #84 (Markers), #91 (Hu) — three slices, one per PR. **No pre-slice decision needed** (constructor is clean — no `prefer_gpu` overlap).

### Slice 1 — Characterization tests (mirror PR #81 Network, PR #88 Markers, PR #95 Hu)

**Clarify** + **Protect**.

- Add `tests/test_voxel_reassignment.py` (~20 tests, ~700 lines).
- Extend `tests/conftest.py`:
  - `network_3d_path` / `network_2d_path` session fixtures (Filter+Label+Network).
  - `hu_outputs_3d_path` / `hu_outputs_2d_path` session fixtures (Filter+Label+Markers+Hu, return `Path` to `flow_vector_array.npy`).
  - `_make_voxel_reassign_imageinfo_factory` (3 file copies: `im_instance_label`, `im_skel_relabelled`, `flow_vector_array.npy`).
  - `make_voxel_reassign_imageinfo_2d/3d` per-test + `_module` variants.
- Pin all contracts from Pass 5 + the inner-OOM characterizations from Pass 7.
- No production-code changes.

**Risk**: low. Pure additive. **Heavier than Hu's Slice 1** — adds two new session caches (Network and Hu output) instead of one (Markers output). Estimated 700 lines test + 180 lines conftest, vs Hu's 865 + 211.

### Slice 2 — Structural cleanups (mirror PR #82 Network, PR #89 Markers, PR #96 Hu)

**Clarify** + **Stabilize**.

- Delete `_get_t` (dead).
- Delete `self.shape = None`, `self.spatial_shape = None`, `self.match_coord_dtype = None`, `self.debug = None` from main init path (and from `no_t` path if not externally read — audit).
- Drop `self._cp` attribute; replace ~12 callsites with `self.xp` / `self.device_type == "cuda"` checks.
- Delete `__main__` block.
- Optional cosmetic: deduplicate the `no_t` early-return init block (84–101) against the main path (103–131) — defer if Slice 2 gets too heavy.
- Wiki touch-ups in [[voxel-reassignment]]: drop the `self._cp` indirection note (if any); update gotcha "Half-hoisted backend code" to flag that the backend hoist is queued for Slice 3.

**Risk**: medium. Constructor signature unchanged (no user-visible widget changes). The `self._cp` drop touches ~12 sites — Slice 1 tests guard the behavior contract.

### Slice 3 — Backend hoist (mirror PR #83 Network, PR #90 Markers, PR #97 Hu)

**Stabilize** + **Separate**.

- Delete `_resolve_backend`, `_try_import_cupy`, `_is_oom_error`, `_free_gpu_memory`, `_switch_to_cpu`.
- Constructor + `_set_backend` call `adaptive_run.resolve_backend` directly. Add a small local helper `_get_gpu_kdtree_cls()` that returns `cupyx.scipy.spatial.cKDTree` when on GPU and the class is importable (irreducible — `adaptive_run.try_import_cupy` returns the ndimage module, not the spatial KDTree class, and only this stage needs it; not worth widening the canonical surface).
- Replace `try/except Exception` patterns at `_build_tree` (247), `_query_tree` (287, 306), `_query_bruteforce_gpu` (354), `_query_bruteforce_cpu` (388 — already `MemoryError`-only) with explicit `if not adaptive_run.is_oom_error(exc): raise` (matches Hu `_match_frames` post-#97 lines 1050–1055 and `_get_frame_features` post-#97 lines 580–583).

- **Inner OOM cascades — architectural decision** (3 cross-frame mutation sites: `_build_tree` 250+252, `_query_tree` 290, 309). Same shape-of-decision as Hu's resolved decision #1, with one additional layer:

  **Option A1 (fully delete inner cascades, like Hu Cascade A)**:
  - On GPU OOM in `_build_tree`: raise → outer `mode_candidates` cascade restarts whole stage from frame 0 with the next `(dev, low)` candidate.
  - On GPU OOM in `_query_tree` (both `gpu` and `gpu_bruteforce` branches): raise → outer cascade.
  - **Tradeoff**: outer retry restarts from frame 0 (re-doing tree builds + queries for early frames). For a long time-series, this is expensive. But it eliminates ALL cross-frame backend mutation.
  - Same shape as Hu Cascade A (per-frame OOM → outer retry).

  **Option A2 (recommended — mixed: keep local algorithmic fallbacks, drop `_switch_to_cpu` calls)**:
  - On GPU OOM in `_build_tree`: drop the `_switch_to_cpu` call. Fall through to the existing CPU KDTree path **within this single call** (lines 260–268). Subsequent `_build_tree` calls in later frames will try GPU again; if GPU is genuinely full, every frame will OOM-and-fallback, but each fallback is fast (cupy throws OOM nearly instantly when memory is exhausted).
  - On GPU OOM in `_query_tree` (gpu branch): drop the `_switch_to_cpu` call. Keep the local rebuild-CPU-KDTree-and-query fallback (lines 291–293) — it's a per-call rebuild, not a backend mutation. Subsequent frames' queries will try GPU again.
  - On GPU OOM in `_query_tree` (gpu_bruteforce branch): same — drop `_switch_to_cpu`, keep local rebuild.
  - Replace `_warned_gpu_fallback` rate-limit with a small `_warn_gpu_fallback(reason)` helper called at each fallback site (preserves the once-per-instance log behavior).
  - **Tradeoff**: every frame after the first OOM pays a fast "try GPU first, OOM, fall back to CPU" cost (vs current code which switches once and stays on CPU). Performance cost negligible (per-frame OOM check is ~µs); architectural cleanliness eliminates the cross-frame mutation footgun.
  - Same shape as Hu's resolved decision for Cascade B (`_match_frames` dense→sparse: keep algorithmic fallback, drop `_switch_to_cpu`).

  **Option B (keep all cascades, route through canonical helpers)**:
  - Replace `self._is_oom_error(exc)` → `adaptive_run.is_oom_error(exc)`, `self._free_gpu_memory()` → `adaptive_run.free_gpu_memory(self.xp)`, `self._switch_to_cpu(reason)` → `self._set_backend("cpu")` + `_warn_gpu_fallback(reason)`.
  - Document the cross-frame mutation explicitly in the wiki (preserves the footgun but at least pinned).

  - Either Option A1/A2: Slice 1's inner-OOM characterization tests get rewritten to pin the new contract (no cross-frame `self.device_type` mutation from inner cascades). The current "silent fallback on non-OOM exceptions" bug (the `_query_tree` GPU branch swallows non-OOM exceptions and switches to CPU) is also fixed — explicit `if not adaptive_run.is_oom_error(exc): raise` propagates non-OOM errors.

**Risk**: medium-high under Option A1 (real behavior change on OOM — full restart), medium under Option A2 (algorithmic fallback preserved, only cross-frame mutation removed), medium under Option B. Slice 1 tests guard the non-OOM contract; the OOM contract is intentionally being changed under A1/A2.

### Deferred (per queue policy)

- `VoxelReassignerConfig` dataclass extraction. Wait until all 4 untested stages (Markers ✓, HuMomentTracking ✓, VoxelReassigner, Hierarchy) have completed test+hoist so the cross-stage Config slice can land as one consistent pass.
- Extending `adaptive_run.try_import_cupy` to return the spatial KDTree class — only one consumer; not worth widening the canonical surface.

---

## Open questions for the user

1. **Slice 3 inner cascades** — Hu had two inner cascades and chose Split Option A (delete one, mixed-modify the other). VoxelReassigner has THREE cross-frame mutation sites across `_build_tree` (250, 252) and `_query_tree` (290, 309) — all four are `_switch_to_cpu` calls that persist across frames. The recommendation is **Option A2 (mixed)**:
   - **All three sites**: keep the local algorithmic fallback (CPU KDTree rebuild within the same call) but drop `_switch_to_cpu`. Each subsequent frame's `_build_tree`/`_query_tree` will try GPU again; if GPU is genuinely full, every frame pays a fast OOM-and-fallback cost (microseconds).
   - **Bonus fix**: the `_query_tree` GPU branches currently `_switch_to_cpu` on **any** exception (not just OOM). Option A2 adds explicit `if not adaptive_run.is_oom_error(exc): raise` so non-OOM errors (e.g., dtype mismatches, dimension errors) propagate instead of silently flipping the backend.
   - Confirm Option A2, or pick A1 (delete the local fallbacks too — outer cascade restarts from frame 0) or B (preserve cross-frame mutation, route through canonical)?

2. **`self._cp` drop in Slice 2** — recommended action is to drop the attribute entirely and use `self.xp` directly when `self.device_type == "cuda"`. Same shape as Markers' `self.use_gpu` cleanup (PR #89) and Hu's `self._on_gpu` cleanup (PR #96). ~12 callsites. Confirm, or keep `self._cp` as a readability shorthand?

3. **Wiki fold timing** — same as Hu PRD #91: defer to during/after the slices (Slice 2 would be the natural point for the `self._cp` cleanup note + half-hoisted-backend-code gotcha update; Slice 3 would rewrite the "OOM at any tier triggers `_free_gpu_memory()` + `_switch_to_cpu()`" gotcha to reflect the new behavior under Option A2)? Or fold the durable findings before Slice 1?
