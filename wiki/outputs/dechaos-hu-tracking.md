---
created: 2026-05-08
modified: 2026-05-08
---

# Dechaos scan — `nellie/tracking/hu_tracking.py` (`HuMomentTracking`)

One-shot review of the next pipeline stage to receive the test+hoist treatment. Findings are intended to feed an upcoming three-slice PRD that mirrors PRDs #77 (Network) and #84 (Markers). Durable items should be folded into [[hu-tracking]] gotchas after the PRD is open.

**Reference templates**: Network (`nellie/segmentation/networking.py`, post-#83) and Markers (`nellie/segmentation/mocap_marking.py`, post-#90) are the closest mirrors — same backend pattern, same outer `adaptive_run.mode_candidates` cascade. Use them as the after-picture.

---

## Resolved decisions (2026-05-08)

1. **Slice 3 inner OOM cascades** → **Split Option A**.
   - **Cascade A (`_get_frame_features` per-frame OOM, lines 572–583)**: delete entirely. Rename `_get_frame_features_impl` → `_get_frame_features`. On per-frame OOM, raise → outer `adaptive_run.mode_candidates` cascade in `run()` retries the whole stage with the next `(dev, low_memory)` candidate. Eliminates the cross-frame backend-mutation footgun. Same shape as Markers' resolved decision #2 (PRD #84).
   - **Cascade B (`_match_frames` dense-OOM → sparse, lines 1140–1146)**: keep the dense→sparse algorithmic fallback but **drop the `_switch_to_cpu()` call**. Sparse is CPU-only on the matching axis, but later frames' feature extraction can stay on GPU. Replace `self._is_oom_error(exc)` → `adaptive_run.is_oom_error(exc)` and `self._free_gpu_memory()` → `adaptive_run.free_gpu_memory(self.xp)`; remove `self._switch_to_cpu()` so frames N+1..end keep their original backend.
   - Net effect: zero cross-frame `self.*` mutation from inner cascades; outer cascade remains the only place the device flips for the rest of the run.

2. **`cost_cutoff = 1.0` lift target** → **Module constant**. Add `_COST_CUTOFF = 1.0` near the top of `nellie/tracking/hu_tracking.py`. Both `_find_best_matches` (909) and `_match_frames_sparse` (1033) reference `_COST_CUTOFF`. Smallest possible dedup; no public API change; no instance state. Slice 1's "dense ↔ sparse equivalence" test pins the value. Lifting to a constructor arg is deferred to the cross-stage `HuMomentTrackingConfig` slice — not done in Slice 2 because there's no documented tuning use case today, and the broader queue.md item ("cost weighting is unjustified in code") should resolve first.

3. **Wiki fold timing** → **Defer entirely**, fold during the slice PRs. Slice 1 adds a "tests pin..." note to the gotchas section. Slice 2 rewrites gotcha #4 (cost_cutoff lift) and adds the dense/sparse equivalence note. Slice 3 rewrites the "adaptive degradation has two layers" gotcha to reflect the new behavior (no cross-frame mutation). Durable items (`.item()` perf footgun, empty-result dtype asymmetry) get folded in whichever slice touches the relevant code, or via a separate `/repo-wiki update` after all 3 slices land.

---

## Pass 1 — System Map

- **Stage 5 of 7** in `nellie/run.py` (line 97). Two in-tree call sites: `run.py:97` (`HuMomentTracking(im_info, device=device, low_memory=low_memory)`) and `nellie_napari/nellie_processor.py:446` (`**step_kwargs` from the settings widget's `get_tracking_params()`).
- **Class**: `HuMomentTracking`, single class in `nellie/tracking/hu_tracking.py`, 1290 lines.
- **Inputs (memmaps via `ImInfo.pipeline_paths`)**: `im_instance_label` (from Label), raw image (from `im_info.im_path`), `im_preprocessed` (from Filter), `im_marker` (from Markers), `im_distance` (from Markers). All five allocated in `_allocate_memory` (508–512).
- **Outputs**: a single `np.save`'d numpy array file at `flow_vector_array_path` (NOT a memmap; written via `np.save` line 1234 inside `_run_hu_tracking`).
  - 2D: 6 columns `[t, y, x, dy, dx, cost]`
  - 3D: 8 columns `[t, z, y, x, dz, dy, dx, cost]`
  - Empty result: `np.empty((0, 6 or 8), dtype=np.float32)` (1230–1232)
- **External deps**: `nellie.utils.adaptive_run` (canonical backend helpers), `numpy`, `scipy.ndimage`, `scipy.spatial.distance.cdist`, `scipy.spatial.cKDTree`, optionally `cupyx.scipy.ndimage` via `cupy`, `nellie.im_info.verifier.ImInfo`.
- **Tests**: none. `tests/test_hu_tracking.py` does not exist; deleted in the May 2026 scaffold rebuild and not restored. Wiki ([[now]] watch list) confirms tracking modules are unpinned.
- **Architectural shape**: per-frame outer loop (`_run_hu_tracking`, 1162–1235) → per-frame feature extraction (`_get_frame_features` wrapping `_get_frame_features_impl`) → per-frame matching (`_match_frames` dispatching to `_get_cost_matrix` + `_find_best_matches` (dense) or `_match_frames_sparse` (KDTree)). Outer `run()` uses `adaptive_run.mode_candidates` for `(dev, low_memory)` retries. **Three layers of inner adaptive degradation on top of that** — see Pass 3.

## Pass 2 — Boundary Scan

| # | Mixed concern | Where | Suggested seam |
|---|---|---|---|
| 1 | Local backend resolution duplicates `adaptive_run` | `_resolve_backend` (143–158), `_try_import_cupy` (160–181), `_is_oom_error` (183–192), `_free_gpu_memory` (194–200), `_switch_to_cpu` (202–206) | Delete; call `adaptive_run.resolve_backend` / `try_import_cupy` / `is_oom_error` / `free_gpu_memory(self.xp)` / `self._set_backend("cpu")`. Same change Network's PR #83 made and Markers' PR #90 made. |
| 2 | Local `GPU_OOM_ERRORS` tuple | 19–23 | `adaptive_run.is_oom_error` already covers `cupy.cuda.memory.OutOfMemoryError`. The `try/except GPU_OOM_ERRORS + (MemoryError,)` clauses (575, 668, 1140) can be replaced with `try/except Exception` + `if not adaptive_run.is_oom_error(exc): raise` — same shape Network's `_get_pixel_class` (484–491) uses today. |
| 3 | Dev driver in module body | `__main__` block (1285–1289) hard-codes a Windows path | Delete (Network did the same in PR #82, Markers in PR #89). |
| 4 | Per-frame inner OOM cascade with cross-frame backend mutation | `_get_frame_features` (572–583) calls `self._switch_to_cpu()` on GPU OOM; mutation persists across all later frames | See Pass 3 / Pass 8. |
| 5 | Inner matching OOM cascade also mutates backend | `_match_frames` (1140–1146) calls `self._switch_to_cpu()` on dense-matching OOM before falling through to sparse | See Pass 3 / Pass 8. |
| 6 | I/O mixed with dispatch in main loop | `_run_hu_tracking` (1162–1235) accumulates `frame_vectors` in memory, concatenates, and writes one `.npy` file at the end (line 1234) | Keep — single-write stage; differs from Network/Markers (per-frame memmap writes) but the boundary is fine. The `.npy` write is at the stage boundary, not mixed in. |

## Pass 3 — Responsibility Scan

The class wears multiple hats — backend management, image-moment math, ROI extraction, KDTree gating, dense/sparse matching, pipeline orchestration. Most hats are inherent to the algorithm and the "fat class" pattern is consistent with Filter/Label/Network/Markers. Cleanups below are **dead code, redundant attributes, and architectural duplication — not split candidates**.

| Item | Location | Status | Action |
|---|---|---|---|
| `_get_t` | 495–501 | Dead. `__init__` (100–102) already resolves `self.num_t`; the `if self.num_t is None` guard inside `_get_t` is unreachable because both `__init__` and `run()` early-return on `self.im_info.no_t`. | Delete. (Network deleted its analogue in #82; Markers in #89.) |
| `self.shape = ()` | 116 | Overwritten in `_allocate_memory` (513). | Drop the init line. |
| `self.debug = None` | 124 | Never read, never set elsewhere. | Drop. (Network deleted same in #82; Markers in #89.) |
| `self._on_gpu` | 130, 173, 186, 195, 575–581, 638–639, 669, 773, 1106, 1140–1146 | Computable from `self.device_type == "cuda"`. Tracked and updated alongside backend at multiple sites (130, 206, 212). Network and Markers both use `self.device_type == "cuda"` directly without a duplicate boolean. | Drop attribute; replace ~10 callsites with `self.device_type == "cuda"`. |
| `_concatenate_hu_matrices` | 548–550 | Single-line wrapper around `xp.concatenate(_, axis=1)`. Misnamed (used for stats too at line 662, not just hu). | Inline at the two call sites (662, 545); the wrapper adds nothing. |
| `_to_cpu_array` | 217–222 | Helper duplicates Network's `_to_cpu` (143–149) shape. | Keep but rename to `_to_cpu` to align with Network (no functional change). Used in `_match_frames` (1112–1115). |
| `_get_frame_features` inner OOM cascade | 572–583 | **Mutates `self.xp`/`self.ndi`/`self.device_type`/`self._on_gpu` in place** (via `_switch_to_cpu()`); persists across all later frames in the same `run()` call. Architectural overlap with outer `mode_candidates` cascade. Same footgun the Markers wiki flagged + Markers PRD #84 resolved by deletion. | See Pass 8 / Slice 3. |
| `_match_frames` inner dense-OOM cascade | 1140–1146 | Same — catches OOM, calls `self._free_gpu_memory()` + `self._switch_to_cpu()`, then falls through to sparse (CPU-only). The backend mutation persists across all later frames. | See Pass 8 / Slice 3. |
| `_get_frame_features_impl` "use_dense → streaming" fallback | 668–674 | Local-only mutation: sets `use_dense = False` (a local variable), no `self.*` mutation, no cross-frame state. **Not a footgun** — the dense→streaming fallback is per-frame and self-contained. | Keep as-is. |

The wiki gotchas already note "adaptive degradation has two layers, both silent" — the dechaos finding refines this: the *outer* (whole-stage) and *inner per-frame* layers BOTH mutate device state, so a run that GPU-OOMs once on frame N silently stays on CPU for frames N+1…end with only `logger.warning` to show for it.

## Pass 4 — Dependency Scan

| # | Type | Issue | Impact |
|---|---|---|---|
| 1 | **No constructor dual-API** (unlike Markers) | 9 named kwargs (`num_t`, `max_distance_um`, `viewer`, `device`, `mode`, `max_dense_pairs`, `max_dense_roi_voxels_cpu`, `max_dense_roi_voxels_gpu`, `low_memory`) with sensible defaults. The napari widget (`nellie_settings.py:281–297`) maps 1:1 to constructor args via `get_tracking_params()` (line 915+). **No `prefer_gpu`-style overlap.** | **No pre-slice decision needed**, unlike Markers PRD #84. |
| 2 | Hardcoded `cost_cutoff = 1.0` in TWO places | `_find_best_matches` (909) AND `_match_frames_sparse` (1033) | Already in [[hu-tracking]] gotchas. **Risk**: changing one without the other silently shifts acceptance rates between dense and sparse paths. Slice 2 candidate: lift to a module/class constant. |
| 3 | Silent fallback on missing T resolution | `dt = self.im_info.dim_res.get('T') or 1.0` (111). Logs a warning at 113. | Acceptable — has a warning. Pin both branches in tests. |
| 4 | Silent input flooring on `max_distance_um` | `self.max_distance_um = max(max_distance_um * dt, 0.5)` (114). Coerces user input upward. Documented in [[tracking/index]]. | Pin in test (assert effective `max_distance_um` for sub-0.5 input). |
| 5 | `device` accepts undocumented `"cuda"` alias | Constructor docstring says `{"auto","cpu","gpu"}` (48); `_resolve_backend` (145) silently accepts `"cuda"`. | `adaptive_run.normalize_device` is the canonical source; backend hoist (Slice 3) makes the constructor go through it, fixing the gap. |
| 6 | `self.device` set raw before `_resolve_backend` | line 128 stores raw `device` string; `_resolve_backend` (line 144) re-normalizes inside. Compare to Network's `__init__` (62): `self.device = device` then `self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(device)`. Same minor shape divergence. | Slice 3 normalizes via `adaptive_run.normalize_device(device)` at constructor entry, matching Network/Markers. |
| 7 | `_get_distance_mask` mixed-type semantics | Args `coords_post_phys`, `coords_pre_phys` are explicitly converted to numpy at 767–768 regardless of backend; `distance_matrix` is computed via `xp` arithmetic on GPU but `cdist` (CPU) on CPU; always returned as `xp.ndarray`. | Document; not a refactor target. |
| 8 | Sparse path is CPU-only but doesn't switch backend | `_match_frames_sparse` (947) uses `cKDTree` (CPU only). `_match_frames` (1106–1115) explicitly converts arrays to CPU when dispatching to sparse, **but `self.device_type` and `self._on_gpu` are NOT updated** when forced into sparse mode (only when dense-OOM triggers `_switch_to_cpu`). | Inconsistent with the dense-OOM cascade. Acceptable but confusing — Slice 1 should pin "forcing `mode='sparse'` does not mutate `self.device_type`". |
| 9 | `.item()` synchronous device→host transfer in tight loop | `_get_frame_features_impl` line 634: `int(xp.ceil(xp.max(distance_max_frame[marker_mask])).item()) * 2 + 1`. Per-frame on GPU, this forces a sync. | Performance footgun on GPU. Document in wiki, don't fix in dechaos slices. |

## Pass 5 — Contract Scan

Pinnable invariants for Slice 1 tests (no test pins any of these today):

**Output schema (`flow_vector_array.npy`)**
- 2D: shape `(N, 6)`, columns `[t, y, x, dy, dx, cost]`. Column dtypes from `np.column_stack` of mixed int64/float32 inputs → final array dtype is **float64** (numpy upcasts; verify in test).
- 3D: shape `(N, 8)`, columns `[t, z, y, x, dz, dy, dx, cost]`. Same dtype upcast behavior.
- Empty result: `np.empty((0, 6 or 8), dtype=np.float32)` (1230–1232) — **dtype mismatch** with the populated path. Pin both contracts so refactors don't silently unify them.
- Cost column values ∈ `[0.0, 1.0]` (cost_cutoff = 1.0).
- All `t` values ∈ `[0, num_t - 2]` (last frame has no successor).

**Per-frame returns (`_FrameFeatures`)**
- `coords_voxel`: numpy array, int dtype, shape `(N, 2)` 2D / `(N, 3)` 3D.
- `coords_phys`: numpy array, float dtype, shape `(N, 2)` 2D / `(N, 3)` 3D, scaled by `self.scaling`.
- `stats`: `xp.ndarray`, shape `(N, 4)` (intensity_mean, intensity_var, frangi_mean, frangi_var).
- `hu`: `xp.ndarray`, shape `(N, 6)` 2D / `(N, 18)` 3D (3-axis max projection × 6 features each).

**Matching contract**
- Hardcoded `cost_cutoff = 1.0` in BOTH dense (`_find_best_matches` line 909) and sparse (`_match_frames_sparse` line 1033) paths. Pin both so divergent updates fail loudly.
- Match acceptance is **union of row-min and col-min candidates** (lines 919–940 dense, 1077–1093 sparse) → may produce duplicates. Pin so a future Hungarian/dedup change is intentional.
- Hu moment 7 (mirror invariance) is intentionally omitted (`_calculate_hu_moments` line 316 comment; returns only the first 6 eta-based moments).
- 3D: `_get_hu_moments` projects sub-volumes along z/y/x then concatenates `(6, 6, 6)` → 18 features (line 545). Not a true 3D moment.

**Adaptive degradation thresholds**
- Dense vs sparse matching: `num_pairs <= max_dense_pairs` (1130) where `num_pairs = N_post * N_pre`.
- Dense vs streaming ROI: `total_voxels <= dense_limit` (640) where `dense_limit = max_dense_roi_voxels_gpu if on_gpu else max_dense_roi_voxels_cpu`.
- `low_memory=True` forces streaming ROI extraction (642).
- KDTree path is CPU-only — converts inputs to numpy regardless of backend.

**Empty-input edge cases**
- Marker frame with zero markers: `_get_frame_features_impl` (612–618) returns `_FrameFeatures` with empty arrays (shape `(0, dims)`).
- Empty stats/hu: `_get_cost_matrix` (864) and `_match_frames_sparse` (968) both short-circuit with `[], [], []`.
- First frame: skipped (1177–1179); never matches against itself.
- All frames empty → `frame_vectors=[]` → `np.empty((0, N), dtype=np.float32)` saved.

**`_log_hu` finiteness**
- `eps = xp.finfo(hu.dtype).tiny` floor (line 325) — `log(0)` becomes `log(eps)`, not `-inf`.
- `xp.where(xp.isfinite(log_hu), log_hu, 0.0)` (328) — replaces NaN/inf with 0. Pin: feed all-zero hu input → output is finite, not NaN.

**Cost weighting (queue.md item)**
- Dense path: `_get_cost_matrix` divides z-scored stats by `stats_matrix.shape[2]` (876) and z-scored hu by `hu_matrix.shape[2]` (882) BEFORE `xp.nansum` (888). Distance feature is NOT divided.
- Sparse path: `_match_frames_sparse` uses `np.mean(z_stats, axis=1)` and `np.mean(z_hu, axis=1)` (1053). Mathematically equivalent to dense's `sum/count`.
- Pin one minimal numeric example so dense and sparse produce ~identical scores on a tiny synthetic input — protects the "z-score sum + 1.0 cutoff" heuristic against silent drift.

## Pass 6 — Composability Scan

Mostly **already factored** — `nellie.utils.adaptive_run` is the canonical backend primitive. Findings:

- `_calculate_normalized_moments` (228–276), `_calculate_hu_moments` (278–317), `_log_hu` (319–329) are clean reusable image-statistics primitives. Single consumer today; **no extraction warranted** — defer until a second consumer materializes. (PRD #196 Slice 2 (#198) deletes `_zscore_normalize` since the cost-matrix path no longer needs it; the second-consumer deferral never resolved. If a future caller wants z-score normalization, write a fresh primitive sized for that use.)
- `_get_distance_mask` is stage-specific (tied to `self.max_distance_um` and `self.xp`). `_get_difference_matrix` is also being deleted by PRD #196 Slice 2 (#198) — the per-feature streaming refactor in `_get_cost_matrix` doesn't need a `(N, N, F)` tensor.
- The outer `run()` cascade pattern (resolve device order → `mode_candidates` → try / `is_gpu_unavailable_error` / `is_oom_error` / log / continue) is **now repeated nearly verbatim across Filter/Label/Network/Markers/Hu** — about 25 lines of structural sameness per stage. Cross-stage extraction candidate, deferred until all 4 untested stages reach the same shape (queue policy: per-stage Config dataclass extraction is the planned cross-stage pass).
- The two inner OOM cascades duplicate the outer cascade's error-classification logic (`_is_oom_error`, `_free_gpu_memory`, `_switch_to_cpu` are the local equivalents of canonical `adaptive_run` helpers). See Pass 3 / Pass 8.
- **Hu's two-axis dense/sparse split** (ROI extraction + matching, each with its own threshold and fallback) is unique to this stage; no extraction warranted.

## Pass 7 — Testability Scan

**Current state**: zero tests (test file deleted in scaffold rebuild). Slice 1 fills this gap before Slices 2–3 touch structure.

**Fixture cascade extension** (`tests/conftest.py`): the cascade is currently Filter (session) → Label (session) → Network/Markers (per-test factories cascading off both). HuMomentTracking's inputs need the **outputs of Markers** (`im_marker`, `im_distance`) PLUS the existing Frangi + Label memmaps. **Add**:

- `markers_2d_paths` / `markers_3d_paths` session-scoped fixtures that run Markers once per session and expose paths to BOTH `im_marker` and `im_distance` (return a `dict[str, Path]` or named tuple).
- `_make_hu_imageinfo_factory` that copies in 4 memmaps: Frangi (`im_preprocessed`), Label (`im_instance_label`), `im_marker`, `im_distance`. Mirrors `_make_markers_imageinfo_factory` (373–413) and `_make_network_imageinfo_factory` (275–308).
- Per-test (`make_hu_imageinfo_2d/3d`) and module-scoped (`_module` variants) factories.

Estimated ~80–100 added lines to conftest. **Heavier than Markers' Slice 1** because Markers piggybacked on existing Frangi+Label session caches; Hu requires a brand-new Markers session cache.

**Test plan for Slice 1** (target ~17 tests, mirroring `test_networking.py` and `test_mocap_marking.py`'s shape):

```text
Smoke / shape:
- 2D + 3D: run end-to-end on the yeast fixtures with device='cpu', assert
  flow_vector_array.npy exists at expected path with non-empty content.
- num_t default vs explicit override.

Output contract:
- 2D file shape (N, 6), 3D file shape (N, 8). N > 0 on yeast fixtures.
- Populated dtype: float64 (np.column_stack upcast); empty dtype: float32.
  Pin both — do not silently unify.
- Cost column values ∈ [0.0, 1.0] inclusive.
- Vector columns are integer-valued.
- All `t` values ∈ [0, num_t - 2].
- No T axis → run() early-returns; no file written / file is empty array.

Behavior:
- mode='dense' forces dense matching path.
- mode='sparse' forces KDTree path regardless of size; assert `self.device_type`
  is unchanged when starting on CPU (sparse is CPU-only but does not mutate
  self.device_type).
- mode='auto' switches based on N_post * N_pre vs max_dense_pairs.
- low_memory=True forces streaming ROI extraction.
- Empty marker frame → no flow vectors written for that pair.
- First frame is never matched against itself.

_log_hu finiteness:
- All-zero hu input → output is finite (no NaN/inf).

Cost weighting equivalence (queue.md item):
- On a tiny synthetic input where dense and sparse both kick in,
  match-set produced by both paths is identical (or near-identical
  with documented tolerance).

Architecture characterization (pin BEFORE Slice 3 changes them):
- Inner _get_frame_features OOM cascade: monkeypatch
  _get_frame_features_impl to raise MemoryError once on GPU; assert
  `self.device_type == "cpu"` after recovery (cross-frame mutation).
- Inner _match_frames dense-OOM cascade: monkeypatch _get_cost_matrix
  to raise MemoryError once on GPU; assert `self.device_type == "cpu"`
  after recovery.

Inputs not mutated:
- raw + Frangi + Label + im_marker + im_distance memmaps unchanged
  after run() (compare hashes pre/post).
```

**Hard-to-test**: the inner OOM cascades require fault injection (monkeypatch `_get_frame_features_impl` or `_get_cost_matrix` to raise `MemoryError` on first call). Keep characterization minimal: assert `self.device_type == "cpu"` after injection, so Slice 3 can intentionally change the contract (under Option A) and the test fails loudly — OR pass under Option B with updated assertion.

**Markers session fixture is the heaviest part of Slice 1**. Markers on a 2-frame 17×192×279 yeast volume should run in <30s (similar order to Label); session caching mitigates per-test cost. Risk: if Markers runs slower than expected, sliding scope to a smaller subsample is an option.

## Pass 8 — Refactor Sequencing

Mirror PRDs #77 (Network) and #84 (Markers) — three slices, one per PR. **No pre-slice decision needed** (constructor is clean — no `prefer_gpu` overlap).

### Slice 1 — Characterization tests (mirror PR #81 Network, PR #88 Markers)

**Clarify** + **Protect**.

- Add `tests/test_hu_tracking.py` (~17 tests, ~600 lines).
- Extend `tests/conftest.py`:
  - `markers_3d_paths` / `markers_2d_paths` session fixtures (run Markers once per session, return `{'im_marker': Path, 'im_distance': Path}`).
  - `_make_hu_imageinfo_factory` (4 memmaps: Frangi + Label + Marker + Distance).
  - `make_hu_imageinfo_2d/3d` per-test + `_module` variants.
- Pin all contracts from Pass 5 + the inner-OOM characterizations from Pass 7.
- No production-code changes.

**Risk**: low. Pure additive. **Heavier than Markers' Slice 1** because of the Markers session fixture extension (Markers piggybacked on existing Frangi+Label caches).

### Slice 2 — Structural cleanups (mirror PR #82 Network, PR #89 Markers)

**Clarify** + **Stabilize**.

- Delete `_get_t` (dead).
- Delete `self.shape = ()`, `self.debug = None` initial state.
- Drop `self._on_gpu` attribute; replace ~10 callsites with `self.device_type == "cuda"`.
- Inline `_concatenate_hu_matrices` (single-line wrapper).
- Rename `_to_cpu_array` → `_to_cpu` for cross-stage naming consistency with Network.
- **Lift `cost_cutoff = 1.0` to a module/class constant** (e.g. `_COST_CUTOFF`) so dense and sparse paths reference the same constant. Optional: only lift to constructor arg in the deferred Config slice.
- Delete `__main__` block.
- Wiki touch-ups in [[hu-tracking]]: add the `cost_cutoff` deduplication note; update gotcha #4 if the hardcoded value moves to a class constant.

**Risk**: medium. Constructor signature unchanged (no user-visible widget changes). The `cost_cutoff` lift is a meaningful refactor — Slice 1 tests guard the value and the dense/sparse equivalence.

### Slice 3 — Backend hoist (mirror PR #83 Network, PR #90 Markers)

**Stabilize** + **Separate**.

- Delete `_resolve_backend`, `_try_import_cupy`, `_is_oom_error`, `_free_gpu_memory`, `_switch_to_cpu`.
- Drop the local `GPU_OOM_ERRORS` tuple (lines 19–23).
- Constructor + `_set_backend` call `adaptive_run.resolve_backend` directly (matching Network's `__init__` line 62 / Markers' post-#90).
- Replace `try/except GPU_OOM_ERRORS + (MemoryError,)` patterns at 575, 668, 1140 with `try/except Exception` + `if not adaptive_run.is_oom_error(exc): raise` (matches Network's `_get_pixel_class` 484–491 pattern).
- **Inner OOM cascades — architectural decision** (same shape as Markers' resolved decision #2):

  **Option A (recommended for the per-frame cascade): delete `_get_frame_features` wrapper entirely.**
  - Rename `_get_frame_features_impl` → `_get_frame_features` and remove the OOM try/except. Outer `mode_candidates` cascade in `run()` handles OOM by retrying the whole stage.
  - **Tradeoff**: outer retry restarts from frame 0. Eliminates the cross-frame backend-mutation footgun (wiki gotcha alignment with Markers' resolved decision #2).
  - **Keep** the `use_dense → streaming` fallback inside `_get_frame_features_impl` (668–674) — that's per-feature, not cross-frame, no `self.*` mutation. Just route `_is_oom_error` → `adaptive_run.is_oom_error`.

  **Option A (recommended for the matching cascade, with caveat): keep the dense → sparse fallback BUT remove the `_switch_to_cpu()` call.**
  - The dense-OOM `_match_frames` cascade currently mutates backend before falling through to sparse (CPU-only). Sparse-only doesn't need GPU; the backend mutation is gratuitous and persists across frames.
  - Remove the `self._free_gpu_memory()` + `self._switch_to_cpu()` lines; just log + fall through. Later frames keep their original backend.
  - Replace `self._is_oom_error(exc)` → `adaptive_run.is_oom_error(exc)`.

  **Option B: keep both inner cascades but route through canonical helpers.**
  - Replace `self._is_oom_error(exc)` → `adaptive_run.is_oom_error(exc)`, `self._free_gpu_memory()` → `adaptive_run.free_gpu_memory(self.xp)`, `self._switch_to_cpu()` → `self._set_backend("cpu")`.
  - Document the cross-frame mutation explicitly in the wiki.

- Either option: delete the local helpers; use canonical `adaptive_run` helpers throughout.
- Slice 1's inner-OOM characterization tests get updated to match the chosen contract (deleted/loosened under Option A; preserved under Option B).

**Risk**: medium-high under Option A (real behavior change on OOM), medium under Option B. Slice 1 tests guard the non-OOM contract; the OOM contract is intentionally being changed.

### Deferred (per queue policy)

- `HuMomentTrackingConfig` dataclass extraction. Wait until all 4 untested stages (Markers ✓, HuMomentTracking, VoxelReassigner, Hierarchy) have completed test+hoist so the cross-stage Config slice can land as one consistent pass.
- Lift `cost_cutoff` to a constructor arg (rather than module constant) — only if a user wants to tune it. No use case today.

---

## Open questions for the user

1. **Slice 3 inner cascades** — Markers had ONE inner cascade and chose Option A (delete entirely). Hu has TWO inner cascades. The recommendation here is **mixed Option A**:
   - **Per-frame cascade (`_get_frame_features`)**: delete entirely, like Markers. Cross-frame backend mutation is the same footgun.
   - **Matching cascade (`_match_frames` dense→sparse)**: keep the dense→sparse fallback (different path, valid local-only fallback) but remove the gratuitous `_switch_to_cpu()` call. Sparse is CPU-only by design; backend mutation is incidental, not necessary.
   - Confirm this split, or pick uniform Option A (delete both) or Option B (keep both, route through canonical)?

2. **`cost_cutoff = 1.0` duplication** in Slice 2 — lift to a module-level constant (e.g. `_COST_CUTOFF`), a class constant (`HuMomentTracking.COST_CUTOFF`), or a constructor arg?
   - Recommendation: module-level constant. Class constant is fine but adds no value over module-level. Constructor arg deferred to Config slice unless a tuning use case exists today.

3. **Wiki fold timing** — same as Markers PRD #84: defer to during/after the slices (Slice 2 would be the natural point for the `cost_cutoff` documentation update + cross-frame mutation gotcha rewrite if Option A lands), or fold the durable findings before Slice 1?
