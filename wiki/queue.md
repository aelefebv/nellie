---
created: 2026-05-06
modified: 2026-05-07
---

# Queue — open architectural questions

Unresolved decisions, design questions, or known unknowns surfaced by the wiki seeding pass. Add items as they arise; remove them when resolved (with a pointer to the decision article or commit that resolved them).

## Open

- **`run copy.py` is a stale developer scratch fork.** Not imported anywhere; older `run()` signature; Jupyter-style cells against a hard-coded Windows path. Decision needed: delete? See [[pipeline]].
- **CLI is broken / out-of-date with `run.py`.** `nellie/cli.py` calls `run(tif_file, ..., ch=, num_t=, output_dirpath=)` but `run.py` no longer accepts those kwargs. Decision: fix `cli.py`, deprecate it, or delete? See [[pipeline]].
- **macOS hard-pinned to CPU** in `nellie/__init__.py` — the MPS branch is commented out. Will Apple Silicon GPU support be revived? See [[gpu-runtime]].
- **`_clean_junctions` in `segmentation/networking.py` is dead code** — defined but never called from the main path. Remove or wire up? See [[networking]].
- **`SettingsConfig` round-trip in the [[settings|napari settings widget]] is dead code today** — preset save/load not wired. Ship or remove?
- **Hu-moment cost weighting is unjustified in code** — z-scored sum of distance + stats + Hu blocks with no learned weights or rationale committed. Document the heuristic or replace? See [[hu-tracking]].
- **`flow_interpolation.py __main__` block has a `self.im_info` typo bug** — uses `self` outside a class. Remove or fix. See [[flow-interpolation]].
- **Plugin-menu injection touches napari private API** (`viewer.window._qt_window` + literal `"&Plugins"` string). Will silently break on napari ≥ 0.5 menu changes. See [[napari-plugin/index|napari plugin]] and [[loader]].
- **`Hierarchy.skip_nodes` defaults differ between callers.** [[pipeline|`run.py`]] hard-codes `False`; the [[settings|napari settings widget]] exposes it as a flag. Reconcile?
- **OOM detection in `adaptive_run` string-sniffs `"out of memory"`.** Fragile across CuPy/CUDA driver versions — new error wordings silently break the fallback. See [[gpu-runtime]].
- **Per-stage backend code is duplicated across 6 stages.** `_resolve_backend`, `_try_import_cupy`, `_is_oom_error`, `_free_gpu_memory`, `_switch_to_cpu`, `_set_backend` are ~identical in [[filtering]], [[labelling]], [[networking]], [[mocap-marking]], [[hu-tracking]], [[voxel-reassignment]], plus [[feature-extraction|`hierarchical`]]. The next pipeline (Pass 8 Separate, scope B) migrates *only* `Filter` to use canonical helpers in [[gpu-runtime|`adaptive_run`]] — Filter is the only stage with characterization tests today. Migrate each remaining stage when it gains its own tests; one stage per PR, each PR adds tests then migrates. Same pattern as the test bootstrap from PRD #51.

## Deferred wiki work

- **INTERVIEW pass not yet run.** DERIVE + LINT done on 2026-05-06. INTERVIEW is the next step in the INIT bootstrap chain — captures tribal knowledge that code can't reveal: why decisions were made, footguns from past incidents, the *why* behind designs DERIVE could only guess at (e.g., the unjustified Hu-moment cost weighting; the rationale for skipping nodes by default; the napari plugin's choice not to call `run.py`). Run `/repo-wiki interview` when ready.

## Recently resolved

_Empty._
