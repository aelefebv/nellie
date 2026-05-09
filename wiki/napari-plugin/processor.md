---
created: 2026-05-06
modified: 2026-05-09
---

# Processor widget

Pipeline driver. **Re-implements** the same stage sequence as [[pipeline|`run.py`]] one button per stage, all via `napari.qt.threading.thread_worker`. "Run Nellie" sets `pipeline=True` and chains preprocess → segment → mocap → track → (reassign?) → feature export through `_start_worker`.

## Why

The CLI / `run.py` path is synchronous and would freeze the UI. The processor wraps each stage in a worker generator, gates the next step on file existence (`check_file_existence` walks `im_info.pipeline_paths`), and exposes per-stage buttons so users can re-run a single stage without redoing the whole pipeline.

## Interactions

- Reads per-step Configs from [[settings|`Settings.get_*_params()`]] at click time. Each getter returns a typed Config (`FrangiConfig`, `LabelConfig`, `NetworkConfig`, `MarkersConfig`, `HuMomentTrackingConfig`, `VoxelReassignerConfig`, `HierarchyConfig`); 6 of 7 also return `num_t` as the second tuple element (`HierarchyConfig` is asymmetric — no num_t per PRD #112 resolved decision #3). The processor's `_run_*` methods pass the Config straight to the stage constructor — no dict-copy + pop + `Config(**kwargs)` wrap. Two cross-cutting checkboxes were folded into their getter in PR #138: `remove_edges` (into `FrangiConfig` via `get_preprocessing_params`) and `skip_nodes` decision (`analyze_node_level` + per-step override → `HierarchyConfig.skip_nodes` via `get_feature_params`).
- Reads basic-tab checkboxes from [[settings|`Settings`]] directly for non-Config concerns: `voxel_reassign` (next-step branch on the tracking → reassign vs feature_export decision), `remove_intermediates_checkbox` (post-Hierarchy cleanup arg, not a Config field). Pre-PR #138, `remove_edges` and `analyze_node_level` were also read here; both now flow through their getters into the relevant Config.
- Final step calls `analyzer.rewrite_dropdown()` on the main thread to refresh the [[analysis|Analyze tab]].
- `check_file_existence` flips the loader's `analysis_tab` enabled when `features_organelles` exists.

## Gotchas

- **Plugin must mirror [[pipeline|`run.py`]] stage order one-to-one.** If `run.py` adds a stage or changes args, this widget needs the same change or behavior diverges silently.
- **Errors from a worker stop the pipeline** (`pipeline=False`) but leave already-produced artifacts on disk. Reruns silently overwrite.
- **No real progress bar** — only `show_info` notifications and an animated ellipses status label (`QTimer`, 500 ms).
- **Each stage's Config is pulled at click time, not at "Run Nellie" time** — changing settings mid-run can produce mixed-config outputs.

## Invariants

- Workers run via `@thread_worker(ignore_errors=True)`; the main thread is never blocked.
- Stage chaining only proceeds if `self.pipeline is True` and the prior step did not error.
