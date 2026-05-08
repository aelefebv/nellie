---
created: 2026-05-06
modified: 2026-05-08
---

# Queue — open architectural questions

Unresolved decisions, design questions, or known unknowns surfaced by the wiki seeding pass. Add items as they arise; remove them when resolved (with a pointer to the decision article or commit that resolved them).

## Open

- **macOS hard-pinned to CPU** in `nellie/__init__.py` — the MPS branch is commented out. Will Apple Silicon GPU support be revived? See [[gpu-runtime]].
- **`_clean_junctions` in `segmentation/networking.py` is dead code** — defined but never called from the main path. **Scoped to Slice 2 (cleanups) of the upcoming Network PRD**, which mirrors PRD #70's three-slice shape (tests → cleanups → backend hoist). Bundled with the other Network dead-code findings from the recent dechaos pass: `_local_max_peak` + the LoG/sigma machinery, the `__main__` block, and the unreachable `force_cpu=False` branch of `_remove_connected_label_pixels`. See [[networking]].
- **`SettingsConfig` round-trip in the [[settings|napari settings widget]] is dead code today** — preset save/load not wired. Ship or remove?
- **Hu-moment cost weighting is unjustified in code** — z-scored sum of distance + stats + Hu blocks with no learned weights or rationale committed. Document the heuristic or replace? See [[hu-tracking]].
- **`flow_interpolation.py __main__` block has a `self.im_info` typo bug** — uses `self` outside a class. Remove or fix. See [[flow-interpolation]].
- **Plugin-menu injection touches napari private API** (`viewer.window._qt_window` + literal `"&Plugins"` string). Will silently break on napari ≥ 0.5 menu changes. See [[napari-plugin/index|napari plugin]] and [[loader]].
- **`Hierarchy.skip_nodes` defaults differ between callers.** [[pipeline|`run.py`]] hard-codes `False`; the [[settings|napari settings widget]] exposes it as a flag. Reconcile?
- **OOM detection in `adaptive_run` string-sniffs `"out of memory"`.** Fragile across CuPy/CUDA driver versions — new error wordings silently break the fallback. See [[gpu-runtime]].
- **Per-stage test bootstrap + backend hoist** for the 4 untested stages: [[networking]], [[mocap-marking]], [[hu-tracking]], [[voxel-reassignment]], plus [[feature-extraction|`hierarchical`]]. Each stage's pipeline is a three-in-one PR: (1) characterization tests against subsampled real data (same fixture pattern as PRD #51); (2) backend hoist to the canonical helpers in [[gpu-runtime|`adaptive_run`]] (`resolve_backend`, `try_import_cupy`, `free_gpu_memory`, `is_oom_error`) — the helpers added for [[filtering]] in PRD #59. Recommended order by pipeline-order: Network → Markers → HuMomentTracking → VoxelReassigner → Hierarchy. Rationale: cumulative test coverage — each stage's tests can rely on real upstream outputs from previously-tested stages, and pipeline-order matches [[pipeline|`run.py`]]'s actual sequencing.
- **Verifier (`im_info/verifier.py`) is unpinned by tests.** The legacy `test_verifier_metadata.py` was wiped during the May 2026 scaffold rebuild and not restored. Behaviors that need pinning before any refactor: per-format `dim_res` parsing (OME, ImageJ, ImageJ-fallback to TIFF tags, ND2 axesCalibration + median-of-diffs T, raw TIFF tag unit conversion), `_normalize_time_axis` leading-singleton heuristic, `change_axes` / `change_dim_res` validation gates, `save_ome_tiff` channel-collapse + T-slicing, OME provenance round-trip, `ImInfo._normalize_axes` / `_normalize_memmap` Z-squeeze. See [[im-info]].
- **Per-stage `*Config` dataclass extraction**, deferred until **all** stages have tests + canonical backend in place — then batched as a single cross-stage pass for consistency review (so individual stages like [[labelling]] that become eligible earlier still wait their turn). Each stage gets a frozen `*Config` dataclass colocated above its class, mirroring the [[filtering|`FrangiConfig`]] pattern from PRD #66. Touches the stage file + `nellie/run.py` + `nellie_napari/nellie_processor.py` + tests + wiki. Two stages have an open design question: [[mocap-marking|`Markers`]] has overlapping `prefer_gpu: bool` + `device: str` args; [[feature-extraction|`Hierarchy`]] has overlapping `use_gpu: bool` + `device: str | None`. Resolve which device-flag wins before the batched config PRD.

## Deferred wiki work

- **INTERVIEW pass not yet run.** DERIVE + LINT done on 2026-05-06. INTERVIEW is the next step in the INIT bootstrap chain — captures tribal knowledge that code can't reveal: why decisions were made, footguns from past incidents, the *why* behind designs DERIVE could only guess at (e.g., the unjustified Hu-moment cost weighting; the rationale for skipping nodes by default; the napari plugin's choice not to call `run.py`). Run `/repo-wiki interview` when ready.

## Recently resolved

- **`nellie/run copy.py` deleted** (2026-05-07) — confirmed orphan, no importers. Removed alongside `cli.py` cleanup.
- **`nellie/cli.py` deleted** (2026-05-07) — broken since `run.py`'s signature change; no console-script entry point referenced it; not worth fixing for a stale UI nobody uses.
