---
created: 2026-05-08
modified: 2026-05-08
---

# Dechaos scan — `nellie/feature_extraction/hierarchical.py` (`Hierarchy`)

One-shot review of the **last untested stage** in the nellie pipeline. Findings are intended to feed an upcoming three-slice PRD that mirrors PRDs #77 (Network), #84 (Markers), #91 (Hu), and #98 (VoxelReassigner). Durable items should be folded into [[feature-extraction]] gotchas during/after the slice PRs.

**Reference templates**: VoxelReassigner (`nellie/tracking/voxel_reassignment.py`, post-#104) is the most recent mirror. Markers (`nellie/segmentation/mocap_marking.py`, post-#90) is the closest *structural* mirror — it had the same `prefer_gpu`/`device` constructor overlap that Hierarchy has now (`use_gpu`/`device`), resolved via Slice 2 Option A (drop `prefer_gpu`). Hu (`nellie/tracking/hu_tracking.py`, post-#97) shows the canonical `adaptive_run.is_oom_error` + `if not …: raise` pattern at every fallback site. The wiki [[feature-extraction]] article already documents the four-tier feature pipeline and the per-frame retry semantics; this scan operationalizes the test+hoist work.

**Headline differences vs prior scans**:
- File is **2131 lines** — almost 2× VoxelReassigner (1117) and Hu (1117). Five sub-classes (`Voxels`, `Nodes`, `Branches`, `Components`, `Image`) + 3 module-level helpers + the `Hierarchy` orchestrator.
- **No inner cross-frame mutation cascade.** `Branches._compute_branch_lengths_and_degrees` has a per-call GPU→CPU fallback but does NOT mutate `self.use_gpu` across frames. So Slice 3 is **simpler** than VoxelReassigner's Slice 3 — no Option A1/A2/B architectural decision to make.
- **Pre-slice constructor overlap to resolve.** `use_gpu: bool` + `device: str | None` overlap (same shape as Markers' pre-Slice 2 `prefer_gpu`/`device` overlap). Per [[queue|wiki/queue.md]], this needs a pre-slice decision — recommended **Option A: drop `use_gpu`, keep only `device`**, mirroring Markers' Slice 2 resolution.
- **Slice 1 is the heaviest yet** — Hierarchy consumes outputs from EVERY upstream stage (Filter, Label, Network, Markers, Hu, VoxelReassigner). Conftest cascade extension materializes a new `voxel_reassign_outputs_*_paths` session cache (Filter+Label+Network+Markers+Hu+VoxelReassigner), and the per-test factory copies in 9–10 files vs VoxelReassigner's 3.

---

## Pass 1 — System Map

- **Stage 7 of 7** in `nellie/run.py:113` — `Hierarchy(im_info, skip_nodes=False, device=device, low_memory=low_memory)`. Two in-tree call sites: `run.py:113` (overrides `skip_nodes=False`) and `nellie_napari/nellie_processor.py:524` (`Hierarchy(**base_kwargs, **step_kwargs)` from `get_feature_params()` at `nellie_settings.py:944`). The napari widget's `skip_nodes` is conditional on the global `analyze_node_level` checkbox (`nellie_processor.py:553–556`).
- **Class**: `Hierarchy` (lines 53–608) + 5 colocated sub-classes (`Voxels` 683–1162, `Nodes` 1275–1429, `Branches` 1444–1877, `Components` 1880–2043, `Image` 2046–2116) + 3 module-level helpers (`append_to_array` 611, `create_feature_array` 628 — **dead, legacy**, `aggregate_stats_for_class` 1165, `distance_check` 1432). 2131 lines.
- **Inputs (memmaps via `ImInfo.pipeline_paths`)** — heaviest of any stage:
  - Always loaded: `im_raw`, `im_preprocessed`, `im_distance`, `im_skel`, `im_pixel_class`, `im_instance_label`, `im_skel_relabelled`, `im_border` (8 files).
  - Loaded only when `not no_t` AND VoxelReassigner ran: `im_obj_label_reassigned`, `im_branch_label_reassigned` (2 files; both must exist on disk for either to be assigned — `_allocate_memory` 217–233 has both-or-neither guard).
  - Loaded indirectly via `FlowInterpolator` when `enable_motility and not no_t and num_t > 1`: `flow_vector_array.npy` (`flow_interpolation.py:121`) plus the raw image again.
- **Outputs**: 5 per-level CSVs (`features_voxels`, `features_nodes`, `features_branches`, `features_organelles`, `features_image`) streamed one frame at a time + 1 pickle (`adjacency_maps`) when `enable_adjacency=True`.
- **External deps**: `nellie.utils.adaptive_run` (canonical backend helpers — already used for `normalize_device`, `gpu_available`, `should_use_low_memory`, `mode_candidates`, `is_gpu_unavailable_error`, `is_oom_error`), `numpy`, `pandas` (CSV), `pickle` (adjacency maps), `scipy.spatial.cKDTree` (used in `distance_check`), `skimage.measure.regionprops`, optionally `cupy` (used directly in `Branches._compute_branch_lengths_and_degrees`), `nellie.tracking.flow_interpolation.FlowInterpolator`, `nellie.im_info.verifier.ImInfo`.
- **Tests**: none. `tests/test_hierarchical.py` does not exist; this is the last untested stage per [[queue|wiki/queue.md]].
- **Architectural shape**: outer `run()` (567–608) uses `adaptive_run.mode_candidates` for `(dev, low_memory)` retries → `_run_hierarchy` (538–565) wires up: `_get_t` → lazy `FlowInterpolator` init → `_allocate_memory` → `_get_hierarchies` (runs Voxels → Nodes → Branches → Components → Image with timing logs) → `_save_dfs` (streams 5 CSVs) → optional `_save_adjacency_maps` (builds 6 edge lists, pickles). The five sub-classes share a `hierarchy.X` fat pointer back to the orchestrator and accumulate per-frame state in instance lists indexed by `t`. There is **no inner OOM cascade** that mutates cross-frame state — the only inner GPU/CPU dispatch (`Branches._compute_branch_lengths_and_degrees` line 1630) is per-call and self-contained.

## Pass 2 — Boundary Scan

| # | Mixed concern | Where | Suggested seam |
|---|---|---|---|
| 1 | Local backend resolution duplicates `adaptive_run` | `_cupy_available` (144–150), `_resolve_device` (152–161) | Delete; call `adaptive_run.gpu_available()` and `adaptive_run.resolve_backend(...)` directly. Same change Network's PR #83, Markers' PR #90, Hu's PR #97, VoxelReassigner's PR #104 made. **`_resolve_device` returns a bool (`use_gpu`)** instead of the canonical `(device_type, xp, ndi)` 3-tuple — the bool API exists because the rest of the class threads `self.use_gpu` (bool) through `Branches._compute_branch_lengths_and_degrees` line 1634. Slice 3 needs to either (a) keep `self.use_gpu` as a derived `self.device_type == "cuda"` property, or (b) drop `self.use_gpu` and inline `self.device_type == "cuda"` checks at the 1 callsite. (b) is cleaner; same shape as Markers' `self.use_gpu` cleanup (PR #89). |
| 2 | Module-level `try: import cupy as cp` + `_HAS_CUPY` flag | 43–50 | Replace internal usage with `adaptive_run.try_import_cupy()` calls or hide behind `_get_cupy_module()` local helper. The `Branches._compute_branch_lengths_and_degrees_backend` (1515–1628) takes an `xp` arg — it doesn't actually need module-level `cp`/`_HAS_CUPY` if the dispatcher (`_compute_branch_lengths_and_degrees` 1630–1639) routes through `adaptive_run.try_import_cupy()` at the dispatch site. **Hu and VoxelReassigner have already deleted equivalent module-level cupy imports** in their post-hoist state. |
| 3 | Local OOM detection in `Branches._compute_branch_lengths_and_degrees` | 1637 — `except cp.cuda.memory.OutOfMemoryError` only | Route through `adaptive_run.is_oom_error(exc)` to catch the broader OOM family (same widening that PRD #70 PR3 added for Network's `"OutOfMemory"` substring). Currently brittle — a different cupy/CUDA version raising a renamed OOM class would slip through and crash the run. |
| 4 | `use_gpu: bool` + `device: str \| None` constructor overlap | `__init__` 63 + 67. `_resolve_device` 152–161 — when `device == "auto"`, `use_gpu` controls; when `device` is explicit `"gpu"`/`"cpu"`, `use_gpu` is ignored. Napari widget passes BOTH (`get_feature_params` at `nellie_settings.py:944` always sends `use_gpu=...` AND `device=...`). | **THE pre-slice decision.** Same shape as Markers' resolved `prefer_gpu`/`device` overlap (PRD #84 Slice 2 picked Option A: drop `prefer_gpu`). Recommended Option A here: drop `use_gpu`; widget stops passing it; `device` alone determines backend (`auto`/`cpu`/`gpu`). One-touch at 4 surfaces: constructor signature + napari widget kwargs assembly + napari widget UI checkbox + the `SettingsConfig` dataclass field at `nellie_settings.py:92`. **Land BEFORE Slice 2 cleanups** so Slice 2 doesn't end up touching the constructor twice. |
| 5 | Dev driver in module body | `__main__` block (2119–2131) — instantiates `Hierarchy` against a hardcoded Windows-only path | Delete (Network/Markers/Hu/VoxelReassigner all did the same in their cleanup slices). |
| 6 | Legacy unused helper | `create_feature_array` (628–680) — docstring explicitly says "Original non-streaming implementation kept for backwards compatibility. Not used inside Hierarchy anymore." | Delete. No grep hits outside the function definition. |
| 7 | I/O mixed with orchestration in `_run_hierarchy` | 538–565 — runs both `_save_dfs` and `_save_adjacency_maps` directly. `_save_dfs` (339–431) repeats the same 11-line CSV-write block 5 times (one per level), differing only in `path` and `labels` arg. | Acceptable cohesion (small enough that DRY-ing the loop adds opacity). `_iter_feature_arrays` already factored out the per-frame streaming. Keep. |
| 8 | Adjacency map computation mixed with persistence | `_save_adjacency_maps` (433–536) — 100 lines of edge-list construction + 1 line of `pickle.dump` | Acceptable cohesion. Single consumer + single output file. Keep. |

## Pass 3 — Responsibility Scan

The class wears multiple hats — backend management, lazy `FlowInterpolator` init, memmap allocation, sub-class orchestration (5 levels), CSV streaming, adjacency map pickling, outer OOM/low-memory cascade. Most hats are inherent to the algorithm and the "fat orchestrator" pattern is consistent with Filter/Label/Network/Markers/Hu/VoxelReassigner. Cleanups below are **dead code, redundant attributes, legacy helpers — not split candidates**.

| Item | Location | Status | Action |
|---|---|---|---|
| `_get_t` | 182–188 | **Dead.** `__init__` line 98 (`self.num_t = self.im_info.shape[0]`) always sets it before `_get_t` could read `None`. The `if self.num_t is None and not self.im_info.no_t:` guard is unreachable. Called once from `_run_hierarchy` line 539, where `self.num_t` is already set. | Delete. (Network deleted analogue in #82; Markers in #89; Hu in #96; VoxelReassigner in #103.) |
| `create_feature_array` | 628–680 | **Dead.** Docstring says so. No grep hits outside the function. | Delete. |
| `__main__` block | 2119–2131 | Dev driver against hardcoded `F:\` path | Delete. |
| `self.im_raw = None`, `self.im_struct = None`, `self.im_distance = None`, `self.im_skel = None`, `self.im_pixel_class = None`, `self.label_components = None`, `self.label_branches = None`, `self.im_border_mask = None` | 122–129 | All 8 overwritten by `_allocate_memory` (194–215). | Drop the init lines — `_allocate_memory` is unconditionally called from `_run_hierarchy` (554) which is unconditionally called from `run()` (593). |
| `self.im_obj_reassigned = None`, `self.im_branch_reassigned = None` | 130–131 | Conditionally overwritten by `_allocate_memory` (217–233). The `None` is the sentinel for "VoxelReassigner output not on disk". | Keep (these init values are load-bearing — referenced as `is None` checks in `Branches._get_branch_stats` line 1770 and `Components._get_component_stats` line 1965). |
| `self.flow_interpolator_fw/bw: FlowInterpolator \| None = None` | 134–135 | Overwritten in `_run_hierarchy` (548–552). The `None` is the sentinel for "motility disabled or single frame". | Keep (referenced as `is None` checks in `Voxels._get_motility_stats` line 968–969). |
| `self.voxels = None`, `self.nodes = None`, `self.branches = None`, `self.components = None`, `self.image = None` | 138–142 | Overwritten by `_get_hierarchies` (239–263). | Keep (used as in-progress sentinels by `_save_dfs` indirection — though all sub-class calls happen from inside `_run_hierarchy` so the sentinels are technically dead. Defer to optional cleanup; not load-bearing). |
| `self._cupy_available()` (instance method) | 144–150, called at `_resolve_device` 156, 161 | Duplicates `adaptive_run.gpu_available()` exactly. | Slice 3 deletes; replace 2 callsites with `adaptive_run.gpu_available()`. |
| `self._resolve_device(device, use_gpu)` (instance method) | 152–161, called at `__init__` 117, `_set_backend` 166 | Duplicates `adaptive_run.resolve_backend` *partially* — returns a bool, not a 3-tuple, and the boolean derives from the device-string + use_gpu flag. | Slice 3 deletes after Slice 2 drops `self.use_gpu`. Constructor + `_set_backend` call `adaptive_run.resolve_backend(device, prefer_gpu=…)` directly; backend stored as `self.device_type` (str). |
| `self.use_gpu` (bool attribute) | 117, set; read at `Branches._compute_branch_lengths_and_degrees` 1634 | Derived from `_resolve_device`; redundant once we have `self.device_type`. | Slice 2 (with the `use_gpu` constructor decision): drop the attribute; replace 1 read site with `self.device_type == "cuda"`. |
| `self._prefer_gpu` (bool attribute) | 116, set; read at `_resolve_device` (called from `_set_backend`) and `run()` line 575 | Stores the constructor `use_gpu` flag for later re-resolution. | Disposition tied to the `use_gpu` decision: dropped along with `use_gpu` if Option A; kept as a normalized field if a different option wins. |
| `self.device` raw storage before normalization | 115 (`self.device = (device or "auto").lower()`) | `adaptive_run.normalize_device` at `run()` 571 and `_set_backend` 164 normalizes again later. | Slice 3: normalize once at constructor entry via `adaptive_run.normalize_device(device)`, matching Network/Markers/Hu/VoxelReassigner post-hoist. |
| `Branches._compute_branch_lengths_and_degrees` per-call OOM fallback | 1630–1639 — try GPU backend, catch `cp.cuda.memory.OutOfMemoryError`, fall back to CPU | **Clean per-call adaptive pattern. No `self.*` mutation.** | Keep; route `cp.cuda.memory.OutOfMemoryError` → `adaptive_run.is_oom_error(exc)` for cross-version robustness. **Same shape as VoxelReassigner's `_query_bruteforce_gpu` chunk-level OOM** (clean local fallback, no cross-frame state). |

The wiki [[feature-extraction]] article already flags the per-region adaptive chunk halving in `Voxels._get_node_info` (842–853, `_process_chunks` MemoryError retry) — that is a third clean local-only OOM fallback in the file. **Pure CPU MemoryError, no cupy involvement** — leave entirely as-is. No `adaptive_run` routing needed since CPU `MemoryError` is caught directly.

## Pass 4 — Dependency Scan

| # | Type | Issue | Impact |
|---|---|---|---|
| 1 | **Constructor dual-API: `use_gpu` + `device`** (matches Markers pre-#84) | 9 named kwargs; `use_gpu` (bool, default True) and `device` (str \| None, default None → "auto") both reachable from the napari widget. When `device == "auto"`, `use_gpu` controls; when `device` is explicit, `use_gpu` is ignored. | **Pre-slice decision required.** Per [[queue|wiki/queue.md]]. Same shape as Markers' `prefer_gpu`/`device` overlap; Markers picked Option A (drop `prefer_gpu`) in PRD #84 Slice 2. Recommended Option A here too; touches constructor + napari widget + `SettingsConfig` dataclass + (optional) hide checkbox in widget UI. |
| 2 | `device` accepts undocumented `"cuda"` alias | Constructor doesn't validate; `_resolve_device` (153) silently accepts `"cuda"` along with `"auto"`/`"cpu"`/`"gpu"`. The constructor docstring (89) lists only `{"auto", "cpu", "gpu"}`. | `adaptive_run.normalize_device` is the canonical source; `_set_backend` already routes through it (164). Backend hoist (Slice 3) makes the constructor go through it too, fixing the gap. |
| 3 | Constructor stores raw `self.device` before normalization | line 115 — `(device or "auto").lower()`. `run()` (571) and `_set_backend` (164) re-normalize via `adaptive_run.normalize_device`. | Slice 3 normalizes via `adaptive_run.normalize_device(device)` at constructor entry, matching Network/Markers/Hu/VoxelReassigner. |
| 4 | `FlowInterpolator` lazy init in `_run_hierarchy` | 542–552. Built only if `enable_motility and not no_t and num_t > 1`. Single-frame stacks skip flow loading even when motility is "enabled". | Documented in wiki gotchas. Pin in test (single-frame fixture → `flow_interpolator_fw is None` post-run). |
| 5 | `low_memory` re-clamps `_resolve_node_chunk_size` ceiling | 176–177 — `max_mask_elems = max(1, max_mask_elems // 4)` when low_memory. | Acceptable. Pin chunk-size derivation in test (synthetic `num_nodes`, `num_voxels` → expected chunk size). |
| 6 | Outer `run()` re-normalizes device but `_set_backend` re-normalizes again | `run()` line 571 and `_set_backend` line 164 both call `adaptive_run.normalize_device(self.device)`. | Belt-and-suspenders; not a bug. Keep. |
| 7 | Module-level `try: import cupy` + `_HAS_CUPY` flag | 43–50; `_HAS_CUPY` used by `Hierarchy._cupy_available` (145) and `Branches._compute_branch_lengths_and_degrees` (1634). | Inside-function `adaptive_run.try_import_cupy()` checks would mask the module-level state. Keep import at module level for `cp.cuda.memory.OutOfMemoryError` reference (1637), but route the runtime check through `adaptive_run.gpu_available()` and the OOM detection through `adaptive_run.is_oom_error(exc)`. |
| 8 | `_save_adjacency_maps` builds 6 edge lists in memory before pickling | 433–536. Could be heavy for large datasets (millions of voxels per frame × num_t). | Acceptable boundary — single-write `pickle.dump`. Same shape as VoxelReassigner's `np.save` of `voxel_matches`. Pin output schema in test. |
| 9 | `Hierarchy.skip_nodes` defaults differ between callers | Constructor default `True`; `run.py:113` hard-codes `False`; `nellie_settings.py:316` defaults checkbox checked (`False`); napari `_run_feature_export` derives from `analyze_node_level` checkbox (`nellie_processor.py:553–554`). | Already flagged as open question in [[queue|wiki/queue.md]]. **Out of scope for the test+hoist PRD** — leave for a follow-up reconciliation pass. |

## Pass 5 — Contract Scan

Pinnable invariants for Slice 1 tests (no test pins any of these today):

**Output schema**

- `features_voxels.csv`: header `t,label,<feature>_<stat>,…` where `<feature> ∈ {linear_vel, angular_vel, linear_acc, angular_acc, rel_linear_vel, rel_angular_vel, rel_linear_acc, rel_angular_acc, rel_directionality, structure, intensity, x, y, z}` and `<stat> ∈ {raw}` (since voxels are leaves, no aggregation). Wait — actually `_iter_feature_arrays` (303–311) treats each `features_to_save` entry as an aggregate-shaped dict `{feature: feature_vals[t]}`, then `append_to_array` (611–625) flattens with `_raw` suffix. Pin both header order AND row count == sum-of-frame-voxel-counts.
- `features_nodes.csv` (only if `not skip_nodes`): same shape, with `<feature> ∈ {divergence, convergence, vergere, node_thickness, x, y, z}` plus aggregated voxel metrics (`<voxel_feature>_mean/std_dev/min/max/sum`).
- `features_branches.csv`: with `<feature> ∈ {branch_length, branch_thickness, branch_aspect_ratio, branch_tortuosity, branch_area, branch_axis_length_maj, branch_axis_length_min, branch_extent, branch_solidity, reassigned_label, x, y, z}` plus aggregated voxel + node (if not skip) metrics.
- `features_organelles.csv`: `<feature> ∈ {organelle_area, organelle_axis_length_maj, organelle_axis_length_min, organelle_extent, organelle_solidity, reassigned_label, x, y, z}` plus aggregated voxel + node + branch metrics.
- `features_image.csv`: only aggregated voxel + node + branch + component metrics. One row per frame.
- `adjacency_maps` (pickle): dict with keys `{"v_b", "v_n", "v_o", "n_b", "n_o", "b_o"}`; each value is a list of `(N_t, 2)` int64 ndarrays per frame. **`v_n` and `n_b`/`n_o` are populated only when `not skip_nodes`** (440 + 478). Pin both branches.
- File NOT written when `enable_adjacency=False`. **Pin both branches** so refactors don't silently start writing.

**Initialization (t=0)**

- Voxel `vec01` at t=0 is full-NaN (998–999 — backward flow undefined for first frame).
- Voxel `vec12` at t=num_t-1 is full-NaN (1006 — forward flow undefined for last frame).
- Single-frame stack (`num_t=1`): all motility features full-NaN regardless of `enable_motility`.

**Aggregation semantics** (`aggregate_stats_for_class` 1165–1272)

- Two implementations: vectorized fast path (1228–1272) and `low_memory=True` path (1183–1219). **Both must produce identical mean/std/min/max/sum** (NaN-equal) for the same `(child_class, t, list_of_idxs)` — already documented as wiki invariant `test_aggregate_stats_low_memory_parity` (which doesn't exist yet — wiki is aspirational). Pin **first**.
- `reassigned_label` is **excluded** from aggregation in both paths (1180, 1186, 1224, 1231) — only the raw value passes through to CSV.
- Multi-dimensional stats are silently skipped (1192–1193, 1236–1238). Pin: a per-frame stat that ends up shape `(N, 3)` (e.g., `linear_vel_vector`) does NOT appear as `<name>_mean/std/…` columns in aggregate consumers.
- Empty-group case (1196–1208): `low_memory` path appends NaN per stat; vectorized path uses `largest_idx == 0` → constructs `(N, 1)` NaN array then reduces. **Both paths must agree** on empty-group → NaN.
- NaN propagation: `nanmean`/`nanstd`/`nanmin`/`nanmax`/`nansum` (1204–1208 + 1255–1259). Empty-of-all-NaN slice triggers `RuntimeWarning` — silenced at module top (24–26, 30–31).

**Vote / assignment behavior** (Voxels-level)

- `_get_min_euc_dist` (861–887): per-branch label, returns the index of the voxel with **minimum-magnitude flow vector** in that branch. NaN entries excluded. Returns shape `(max_label + 1,)` float, NaN where no valid voxel. Pin one minimal numeric example.
- `_get_ref_coords` (889–913): clips branch labels to `[0, max_label]`, looks up `idxmin`, returns `(coords[idxmin], coords[idxmin])`. NaN preserved through `vals_a/b` mask. Pin shape contract.
- Per-branch `rel_*` motility uses the single `_get_min_euc_dist` representative — wiki gotcha already documents this. Pin via assertion that `rel_linear_vel` for voxels in a branch equals their offset from the min-flow representative.

**Node assignment chunking** (`Voxels._get_node_info` 743–860)

- `_resolve_node_chunk_size` (171–180): default chunk `node_chunk_size or 10000`; if `num_nodes * chunk > max_node_mask_elems`, chunk shrinks to `max(1, max_node_mask_elems // num_nodes)`; low_memory mode quarters `max_node_mask_elems`. Final clamped to `[1, num_voxels]`.
- Adaptive chunk halving on `MemoryError` (842–853): clean local-only fallback. Pin: monkeypatch `_process_chunks` to raise on first call → assert chunk halved on retry.

**Branch length / thickness** (`Branches._get_branch_stats` 1641–1804)

- `_compute_branch_lengths_and_degrees` (1630–1639): GPU backend on `self.use_gpu and _HAS_CUPY`, fall back to CPU on `cp.cuda.memory.OutOfMemoryError` (1637 — narrow). **Pin** the per-call fallback by monkeypatching `_compute_branch_lengths_and_degrees_backend` to raise `cp.cuda.memory.OutOfMemoryError` on first call → assert second call uses CPU backend, result equivalent to direct-CPU run.
- Tip-aware length: `lone_tips` (degree 0) get `2 * radius` added; `tips` (degree 1) get `radius` added (1699–1706). Pin one minimal numeric example (a 3-voxel branch with degree pattern `[1, 2, 1]` and known radii).
- Length/thickness swap (1719–1722): if `median_thickness > base_length`, swap. Wiki gotcha already documents this. Pin: synthetic blobby branch where thickness exceeds length → swap occurs.
- Tortuosity uses **first two tips** per label only (1733–1750). Pin: branch with 3+ tips → tortuosity computed from first two tip pairs only.

**Reassigned label** (`Branches._get_branch_stats` 1768–1774, `Components._get_component_stats` 1963–1969)

- Set to `np.nan` if `no_t` OR `im_branch_reassigned/im_obj_reassigned is None` (VoxelReassigner outputs missing on disk).
- When set, computed as `argmax(bincount(region_reassigned_labels))` — most common reassigned label in the region's voxels. Pin: 5 voxels with reassigned labels `[1,1,1,2,2]` → reassigned_label == 1.

**`enable_motility=False`**

- `Voxels._get_motility_stats` (956–989): early returns with NaN-fills for all motility outputs. Wiki gotcha already documents this. Pin: `enable_motility=False` → `linear_vel_raw` column all-NaN AND no flow files loaded.

**`enable_adjacency=False`**

- `_save_adjacency_maps` not called. Pin: `enable_adjacency=False` → adjacency_maps file does not exist on disk.

**`skip_nodes=True`**

- `Nodes.run()` (1421–1429) early-returns; `Voxels._run_frame` (1149) skips `_get_node_info`; `_save_dfs` (363) skips writing `features_nodes.csv`; `_save_adjacency_maps` (446) skips `v_n` edges and `n_b`/`n_o` blocks. Pin all four.

**Backend semantics**

- `device='cpu'` start: `self.use_gpu == False` throughout. Branches' `_compute_branch_lengths_and_degrees` takes the np-only path. Pin.
- `device='gpu'` start (skip-if-no-cupy): `self.use_gpu == True`. Branches' `_compute_branch_lengths_and_degrees` takes the cp path; falls to np per-call on OOM **without mutating `self.use_gpu`**. Pin post-run `self.use_gpu == True` even if all per-call fallbacks fired (current behavior — pin BEFORE Slice 3 changes anything else, in case Slice 3 inadvertently changes this).
- Outer `run()` (567–608): on `is_gpu_unavailable_error` → swap to CPU; on `is_oom_error` → retry with `low_memory=True`. Same `mode_candidates` cascade as other stages. Pin via fault injection on `_run_hierarchy` first-frame raise.

**`no_t` short-circuit**

- `im_info.no_t == True` → `_run_hierarchy` (542–552) does NOT init flow interpolators (motility branch fails the `not im_info.no_t` guard). `_allocate_memory` (217–233) skips reassigned-label loading. `_save_dfs` runs all 5 CSVs (single-row each since `num_t == 1`). `_save_adjacency_maps` runs over 1 frame.
- Wait — `_get_t` line 187 is the only place `self.num_t` would be set from `im_info.shape[axes.index("T")]` for non-no_t case. For `no_t` case, `__init__` line 98 already sets `self.num_t = self.im_info.shape[0]`. **Latent bug?**: for a `no_t` image with shape e.g. `(1, Z, Y, X)`, `self.num_t = 1`. The Voxels/Branches/etc. `run()` methods loop `for t in range(self.hierarchy.num_t)` — so they loop ONCE. That's correct behavior. False alarm. But pin `no_t=True → num_t=1` and end-to-end run completes.

**Inputs not mutated**

- `im_raw`, `im_preprocessed`, `im_distance`, `im_skel`, `im_pixel_class`, `im_instance_label`, `im_skel_relabelled`, `im_border`, `im_obj_label_reassigned`, `im_branch_label_reassigned`, `flow_vector_array.npy` all unchanged after `run()` (compare hashes pre/post). Same pattern as VoxelReassigner Slice 1.

## Pass 6 — Composability Scan

Mostly **already factored** at the right level — `nellie.utils.adaptive_run` is the canonical backend primitive (and Hierarchy already uses `mode_candidates`, `is_gpu_unavailable_error`, `is_oom_error`, `should_use_low_memory`). Findings:

- The 5 sub-classes (`Voxels`, `Nodes`, `Branches`, `Components`, `Image`) share a common shape: `__init__(self, hierarchy)` + per-frame state lists + `_run_frame(t)` + `run()` looping over `num_t`. **Could** introduce a `Level(ABC)` base class with `run()` template, but the duplicated `for t in range(...)` + viewer status update pattern is only 4–5 lines and the per-class state diverges enough that the abstraction would force a `dict[str, list]` indirection. **Don't extract**; defer indefinitely.
- `_iter_feature_arrays` (279–337) is already the streaming primitive that replaced `create_feature_array` (628–680). Delete the legacy version (it's annotated as such).
- The 5-block CSV-write pattern in `_save_dfs` (339–431) — same 11-line block repeated with `path` + `level` + `labels` differences. **Could** factor into `_save_level_csv(level, path, labels)`. Marginal — saves ~30 lines but adds one indirection. Defer to Slice 2 if the cleanup is light, otherwise leave; explicit unrolled is arguably more readable when there are exactly 5 levels and the reader wants to find a specific one. **Recommendation: leave**.
- The 3-block `regionprops` morphology block in `Branches._get_branch_stats` (1768–1804) and `Components._get_component_stats` (1963–1997) is nearly identical — same fields collected, same NaN handling, same axis-length try/except. Could factor into `_collect_regionprops(regions, no_z, im_reassigned_t)`. Worth doing in a future pass; not a Slice 2 priority.
- The outer `run()` cascade pattern (resolve device order → `mode_candidates` → try / `is_gpu_unavailable_error` / `is_oom_error` / log / continue) is **now repeated nearly verbatim across Filter/Label/Network/Markers/Hu/VoxelReassigner/Hierarchy** — about 25 lines of structural sameness per stage. **Cross-stage extraction candidate**, queued (per [[queue|wiki/queue.md]]) until the cross-stage Config dataclass slice. Hierarchy is the LAST stage to reach the same shape — once Slice 3 lands, this extraction becomes eligible.

## Pass 7 — Testability Scan

**Current state**: zero tests. Slice 1 fills this gap before Slices 2–3 touch structure. **This is the heaviest Slice 1 in the queue** — Hierarchy consumes outputs from every upstream stage.

**Fixture cascade extension** (`tests/conftest.py`): the cascade is currently Filter (session) → Label (session) → Network (session, added by VoxelReassigner Slice 1) → Markers (session) → Hu output (session, added by VoxelReassigner Slice 1) → VoxelReassigner (per-test factories). Hierarchy needs **all of the above plus VoxelReassigner outputs as a session cache**. Add:

- `voxel_reassign_outputs_3d_paths` / `voxel_reassign_outputs_2d_paths` session-scoped fixtures: run Filter+Label+Network+Markers+Hu+VoxelReassigner once per session, return a dict with `Path`s to `im_obj_label_reassigned`, `im_branch_label_reassigned` memmaps. Pattern mirrors `markers_*_paths` (543) and `hu_outputs_*_path` (added in VoxelReassigner Slice 1). **Heaviest new fixture in the whole suite** — full upstream pipeline through stage 6.
- `_make_hierarchical_imageinfo_factory` that copies in 9–11 files: `im_preprocessed`, `im_distance`, `im_skel`, `im_pixel_class`, `im_instance_label`, `im_skel_relabelled`, `im_border`, `im_obj_label_reassigned` (optional, only if VoxelReassigner ran), `im_branch_label_reassigned` (optional, ditto), `flow_vector_array.npy` (optional, only if motility), and the raw image is read directly from `im_info.im_path`. Mirrors `_make_voxel_reassign_imageinfo_factory` (added in VoxelReassigner Slice 1) shape, just with way more files.
- Per-test (`make_hierarchical_imageinfo_2d/3d`) and module-scoped (`_module` variants) factories.
- Possibly a **lightweight variant** that skips the `voxel_reassign_outputs_*` cache for tests that don't need reassigned labels (e.g., `no_t` tests) — saves ~30s of pipeline runtime per such test.

Estimated **~250–300 added lines to conftest** — the heaviest delta yet. **~3× heavier than Hu's Slice 1 conftest** (which added one Markers session cache) and **~1.5× heavier than VoxelReassigner's Slice 1 conftest** (which added Network + Hu output session caches). Risk: full upstream-pipeline session fixture might run 90–120s on CPU before any Hierarchy test executes.

**Test plan for Slice 1** (target ~25 tests, mirroring `test_voxel_reassignment.py`'s 27-test shape):

```text
Smoke / shape:
- 2D + 3D: run end-to-end on the yeast fixtures with device='cpu', assert
  features_voxels.csv, features_nodes.csv, features_branches.csv,
  features_organelles.csv, features_image.csv all exist with non-zero rows.
- skip_nodes=True (default): features_nodes.csv NOT created; v_n / n_b /
  n_o keys empty in adjacency_maps.pkl.
- skip_nodes=False: all 5 CSVs created; adjacency_maps populated.
- no_t fixture → all 5 CSVs created with single-row content; flow
  interpolators not initialized.
- enable_motility=False: motility columns (linear_vel_*, angular_vel_*,
  linear_acc_*, angular_acc_*, rel_*) all-NaN in features_voxels;
  FlowInterpolator never instantiated.
- enable_adjacency=False: adjacency_maps.pkl does not exist.

Output contract:
- 2D + 3D: features_voxels row count == sum(num_foreground_voxels per t).
- features_branches row count == sum(num_unique_branch_labels per t).
- features_organelles row count == sum(num_unique_component_labels per t).
- features_image row count == num_t.
- CSV header order is stable across frames (set on first frame, reused
  in append mode).
- adjacency_maps.pkl: dict with keys {v_b, v_n, v_o, n_b, n_o, b_o};
  each value is a list of (N_t, 2) int64 ndarrays per frame.

Aggregation parity:
- aggregate_stats_for_class with low_memory=False vs low_memory=True
  produces NaN-equal mean/std/min/max/sum on the same (child_class, t,
  list_of_idxs). [The wiki invariant test_aggregate_stats_low_memory_parity
  exists in the wiki but not in the test suite — pin first.]
- reassigned_label excluded from aggregate columns regardless of low_memory.
- Empty-group case → NaN in both paths.

Vote / motility characterization:
- _get_min_euc_dist minimal numeric example: 3 voxels in one branch with
  flow vector magnitudes [1.0, 0.5, 2.0] → returns idx 1 for that branch.
- vec01 at t=0 is full-NaN; vec12 at t=num_t-1 is full-NaN.
- Single-frame stack (num_t=1): all motility full-NaN.

Branch length / thickness characterization:
- Synthetic 3-voxel branch with degree pattern [1, 2, 1] and known radii:
  pin tip-radius adjustment.
- Length/thickness swap: synthetic blobby segment where thickness > length
  → values swap.
- Tortuosity: synthetic branch with 3 tips → tortuosity computed from
  first two tip pairs only.

Reassigned label characterization:
- no_t fixture: reassigned_label all-NaN in branches/components CSVs.
- VoxelReassigner outputs missing on disk: reassigned_label all-NaN.
- Region with [1,1,1,2,2] reassigned voxels → reassigned_label == 1.

Backend characterization (CPU only — GPU paths skipped if cupy unavailable):
- device='cpu' end-to-end: post-run self.use_gpu == False.
- Branches._compute_branch_lengths_and_degrees: monkeypatch
  _compute_branch_lengths_and_degrees_backend to raise
  cp.cuda.memory.OutOfMemoryError on first GPU call → assert second call
  uses CPU backend, result equivalence with direct-CPU run.
- Outer cascade fault injection: monkeypatch _run_hierarchy to raise
  cupy OOM on first call → assert retry with low_memory=True succeeds.

Node assignment chunking:
- _resolve_node_chunk_size synthetic table: pin formula across
  num_nodes/num_voxels/low_memory combinations.
- Monkeypatch _process_chunks to raise MemoryError on first call →
  assert chunk size halved on retry.

Inputs not mutated:
- raw + im_preprocessed + im_distance + im_skel + im_pixel_class +
  im_instance_label + im_skel_relabelled + im_border +
  im_obj_label_reassigned + im_branch_label_reassigned +
  flow_vector_array.npy unchanged after run() (compare hashes pre/post).

Viewer status callback:
- viewer=None: no-op.
- viewer with .status property: assert per-frame status string written
  for each level (5 levels × num_t writes plus the "Saving features to
  csv files." and "Done!" boundary writes).
```

**Hard-to-test on CPU-only CI**: the GPU branch in `Branches._compute_branch_lengths_and_degrees` requires either a GPU runtime or fault injection on `cp.cuda.memory.OutOfMemoryError`. Strategy: monkeypatch `_HAS_CUPY = True` and a stub `cp` module on the test process (skip if real `cp` is importable to avoid double-bookkeeping). Same fault-injection strategy as VoxelReassigner Slice 1 used for inner-OOM characterization.

**Heaviest part of Slice 1**: the `voxel_reassign_outputs_*_paths` session fixture. Filter + Label + Network + Markers + Hu + VoxelReassigner on a 2-frame yeast volume should run in ~120–180s combined; session caching mitigates per-test cost. **Risk**: this is the longest pre-test setup in the suite. Sliding scope to a smaller subsample (e.g., 1-frame "no_t" only) would skip the VoxelReassigner step (which requires `num_t > 1`) but loses coverage of the reassigned-label code path. Recommendation: keep the 2-frame fixture; cache aggressively.

## Pass 8 — Refactor Sequencing

Mirror PRDs #77 (Network), #84 (Markers), #91 (Hu), #98 (VoxelReassigner) — three slices, one per PR. **Pre-slice decision REQUIRED** on `use_gpu`/`device` overlap (see Pass 4 #1 + open question #1).

### Pre-slice 0 — Constructor overlap decision (mirror Markers' Slice 2 in PRD #84)

**One micro-PR.** Resolve `use_gpu` + `device` overlap. Recommended **Option A (drop `use_gpu`)**:

- Remove `use_gpu: bool = True` from `Hierarchy.__init__` signature.
- Stop reading `self._prefer_gpu` (or replace with a local during constructor — dropped along with `use_gpu`).
- `_resolve_device(device, use_gpu)` becomes `_resolve_device(device)` (one read site at line 117 + one at `_set_backend` line 166).
- `Branches._compute_branch_lengths_and_degrees` line 1634 reads `self.use_gpu` — **needs to become `self.device_type == "cuda"` after Slice 3**, but in this pre-slice it can stay `self.use_gpu` since `self.use_gpu = self._resolve_device(self.device)` is still set.
- Drop `feature_use_gpu` from `nellie_settings.py`: line 92 (dataclass field), line 321 (UI checkbox), line 594 (form row), line 725 (config write), line 834 (config read), and most importantly line 946 (`get_feature_params` returns `use_gpu=...`).
- Drop the `Use GPU` checkbox from the Feature Export form in the napari UI.

**Risk**: low. Single-purpose decision PR. Can be merged BEFORE Slice 1 if user wants Slice 1 to write tests against the cleaned-up constructor signature. **Alternative**: fold into Slice 2 (Markers' approach in PRD #84). Recommended approach: **fold into Slice 2** so Slice 1 tests can pin both the OLD signature (deprecation pin) and the new signature won't churn the test file. Same as Markers post-#89.

### Slice 1 — Characterization tests (mirror PR #81 Network, PR #88 Markers, PR #95 Hu, PR #102 VoxelReassigner)

**Clarify** + **Protect**.

- Add `tests/test_hierarchical.py` (~25 tests, ~900–1000 lines).
- Extend `tests/conftest.py`:
  - `voxel_reassign_outputs_3d_paths` / `voxel_reassign_outputs_2d_paths` session fixtures (Filter+Label+Network+Markers+Hu+VoxelReassigner).
  - `_make_hierarchical_imageinfo_factory` (9–11 file copies).
  - `make_hierarchical_imageinfo_2d/3d` per-test + `_module` variants.
- Pin all contracts from Pass 5 + the `_compute_branch_lengths_and_degrees` GPU-fallback characterization from Pass 7.
- Test against the OLD constructor signature (with `use_gpu`) so the deprecation lands in Slice 2, not Slice 1.
- No production-code changes.

**Risk**: medium-low. Pure additive but **heaviest Slice 1 conftest delta yet** (~250–300 conftest lines vs VoxelReassigner's ~328 — actually similar size, but with one more upstream stage materialized). Estimated 900 lines test + 250 conftest, vs VoxelReassigner's 1145 + 328. Risk lever: `voxel_reassign_outputs_*` session fixture might run 90–180s before any test executes; if too slow, scope to 1-frame `no_t` for half the tests and 2-frame for the rest.

### Slice 2 — Structural cleanups + `use_gpu` overlap resolution (mirror PR #82 Network, PR #89 Markers, PR #96 Hu, PR #103 VoxelReassigner)

**Clarify** + **Stabilize**.

- **Resolve `use_gpu`/`device` overlap (Option A — drop `use_gpu`)**. See Pre-slice 0 for the diff. Folded into Slice 2 here.
- Delete `_get_t` (dead — unreachable guard).
- Delete `create_feature_array` (628–680 — explicitly dead per docstring).
- Delete `__main__` block (2119–2131).
- Delete dead `self.X = None` init lines: `self.im_raw`, `self.im_struct`, `self.im_distance`, `self.im_skel`, `self.im_pixel_class`, `self.label_components`, `self.label_branches`, `self.im_border_mask` (122–129, 8 lines). `self.im_obj_reassigned`, `self.im_branch_reassigned`, `self.flow_interpolator_*`, `self.voxels/nodes/branches/components/image` are load-bearing as None sentinels — **keep**.
- Drop `self.use_gpu` attribute (set at 117, read at 1634). Replace 1 callsite with `self.device_type == "cuda"` check (or leave as a derived `@property` if cleaner).
- Drop `self._prefer_gpu` attribute (set at 116, read at `_resolve_device` and `run()` 575). Without `use_gpu`, the `auto` branch becomes "use GPU if available" unconditionally — same as Markers/Hu/VoxelReassigner post-hoist.
- Optional cosmetic: deduplicate the `_save_dfs` 5-block CSV-write pattern (~30 lines saved). **Defer if Slice 2 gets too heavy** — explicit unrolled is arguably more readable.
- Wiki touch-ups in [[feature-extraction]]: drop the `use_gpu` mention from gotchas if present; flag the backend hoist as queued for Slice 3.

**Risk**: medium. Constructor signature CHANGES (no more `use_gpu` kwarg). Napari widget UI form changes. The `self.use_gpu` drop touches 1 site (low impact) but the napari widget + dataclass + UI form changes touch ~6 sites. Slice 1 tests guard the runtime contract. **Validate the napari widget change in a real napari session** (`/code` Phase 4 mandatory step for UI changes).

### Slice 3 — Backend hoist (mirror PR #83 Network, PR #90 Markers, PR #97 Hu, PR #104 VoxelReassigner)

**Stabilize** + **Separate**.

- Delete `_cupy_available` (144–150) and `_resolve_device` (152–161).
- Constructor + `_set_backend` call `adaptive_run.resolve_backend(device)` directly. Backend stored as `self.device_type` (str, `"cpu"` or `"cuda"`) and `self.xp` (numpy or cupy module).
- `_set_backend` becomes ~3 lines: `device = adaptive_run.normalize_device(device); self.device = device; self.device_type, self.xp, _ = adaptive_run.resolve_backend(device)`.
- Constructor normalizes via `adaptive_run.normalize_device(device)` at entry (matching Network/Markers/Hu/VoxelReassigner post-hoist).
- Delete the module-level `try: import cupy as cp / _HAS_CUPY = True` block (43–50). Replace 2 callsites:
  - `Hierarchy._cupy_available` → already deleted (line 145 was the only consumer).
  - `Branches._compute_branch_lengths_and_degrees` line 1634 (`if self.hierarchy.use_gpu and _HAS_CUPY`) → `if self.hierarchy.device_type == "cuda"` (since Slice 2 already dropped `use_gpu`).
  - `Branches._compute_branch_lengths_and_degrees` line 1636 (`return self._compute_branch_lengths_and_degrees_backend(t, cp)`) → use `self.hierarchy.xp` instead of module-level `cp`.
  - `Branches._compute_branch_lengths_and_degrees` line 1637 (`except cp.cuda.memory.OutOfMemoryError`) → `except Exception as exc: if not adaptive_run.is_oom_error(exc): raise; logger.warning(...)`. **Same explicit `if not …: raise` pattern as Hu post-#97 + VoxelReassigner post-#104.**
- **No `_switch_to_cpu` to delete** — Hierarchy never had one. **No inner cross-frame mutation cascade to refactor.** Slice 3 is structurally the SIMPLEST of the four hoist slices.
- Wiki touch-ups in [[feature-extraction]]: rewrite the "GPU OOM falls back to CPU per call without mutating self.use_gpu" gotcha to reflect the new behavior (now via `adaptive_run.is_oom_error`).

**Risk**: medium. Real behavior change: the `_compute_branch_lengths_and_degrees` fallback now catches the broader OOM family (was: only `cp.cuda.memory.OutOfMemoryError`). Slice 1 tests need to be **rewritten to pin the new contract** — the architecture-characterization test that uses `cp.cuda.memory.OutOfMemoryError` directly becomes one that uses an arbitrary OOM-family exception. Same shape as VoxelReassigner Slice 3's 3-test inversion.

### Deferred (per queue policy)

- `HierarchyConfig` dataclass extraction. Wait until **all stages have completed test+hoist** so the cross-stage Config slice can land as one consistent pass. With Hierarchy as the LAST stage, this becomes eligible immediately after Slice 3 lands.
- `Level(ABC)` base class for the 5 sub-classes. Marginal value; defer indefinitely.
- `_save_level_csv(level, path, labels)` extraction in `_save_dfs`. Marginal; defer or fold into Slice 2 if it's light.
- `_collect_regionprops(regions, no_z, im_reassigned_t)` extraction across `Branches._get_branch_stats` and `Components._get_component_stats`. Worth doing in a follow-up cleanup PR; not Slice 2 priority.
- `Hierarchy.skip_nodes` cross-caller default reconciliation (queue.md item). **Out of scope** for the test+hoist PRD.

---

## Open questions for the user

1. **`use_gpu` / `device` constructor overlap (Pre-slice 0)** — Markers had the analogous `prefer_gpu` / `device` overlap and resolved it via Option A in Slice 2 of PRD #84 (drop `prefer_gpu`, keep only `device`). Same recommendation here: **drop `use_gpu`, keep only `device`**. Confirm Option A — and confirm whether to fold into Slice 2 (Markers' approach) or land as a separate Pre-slice 0 micro-PR before Slice 1?

2. **`_save_dfs` 5-block CSV-write deduplication in Slice 2** — saves ~30 lines but adds one indirection (`_save_level_csv(level, path, labels)`). Markers/Hu/VoxelReassigner Slice 2s did some optional cosmetic cleanups, others deferred. Fold the dedup into Slice 2, or defer to a follow-up cleanup PR?

3. **Wiki fold timing** — same as VoxelReassigner PRD #98: defer to during/after the slices (Slice 2 would be the natural point for the `use_gpu` removal note + half-hoisted-backend-code gotcha update; Slice 3 would rewrite the GPU OOM gotcha for the new `adaptive_run.is_oom_error` behavior). Or fold the durable findings before Slice 1?

4. **`voxel_reassign_outputs_*_paths` session fixture cost** — full upstream-pipeline session fixture might run 90–180s before any Hierarchy test executes, making this the longest pre-test setup in the suite. **Acceptable** (caches across all tests in the session), or scope down to a 1-frame `no_t` fixture for half the tests and 2-frame for the rest (saving ~60s but losing coverage of the reassigned-label code path)?
