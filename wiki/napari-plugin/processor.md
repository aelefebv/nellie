---
created: 2026-05-06
modified: 2026-05-07
---

# Processor widget

Pipeline driver. **Re-implements** the same stage sequence as [[pipeline|`run.py`]] one button per stage, all via `napari.qt.threading.thread_worker`. "Run Nellie" sets `pipeline=True` and chains preprocess → segment → mocap → track → (reassign?) → feature export through `_start_worker`.

## Why

The CLI / `run.py` path is synchronous and would freeze the UI. The processor wraps each stage in a worker generator, gates the next step on file existence (`check_file_existence` walks `im_info.pipeline_paths`), and exposes per-stage buttons so users can re-run a single stage without redoing the whole pipeline.

## Interactions

- Reads per-step kwargs from [[settings|`Settings.get_*_params()`]] at click time. The dict returned by `get_preprocessing_params()` is wrapped into a `FrangiConfig` (after popping `num_t`) before being passed to [[filtering|`Filter`]]. Other stages still take kwargs directly until they get their own `*Config` dataclasses.
- Reads basic-tab checkboxes from [[settings|`Settings`]] directly: `remove_edges`, `analyze_node_level`, `voxel_reassign`, `remove_intermediates`. The basic tab is **shared mutable state**, not just a settings store.
- Final step calls `analyzer.rewrite_dropdown()` on the main thread to refresh the [[analysis|Analyze tab]].
- `check_file_existence` flips the loader's `analysis_tab` enabled when `features_organelles` exists.

## Gotchas

- **Plugin must mirror [[pipeline|`run.py`]] stage order one-to-one.** If `run.py` adds a stage or changes args, this widget needs the same change or behavior diverges silently.
- **Errors from a worker stop the pipeline** (`pipeline=False`) but leave already-produced artifacts on disk. Reruns silently overwrite.
- **No real progress bar** — only `show_info` notifications and an animated ellipses status label (`QTimer`, 500 ms).
- **Each stage's kwargs are pulled at click time, not at "Run Nellie" time** — changing settings mid-run can produce mixed-config outputs.

## Invariants

- Workers run via `@thread_worker(ignore_errors=True)`; the main thread is never blocked.
- Stage chaining only proceeds if `self.pipeline is True` and the prior step did not error.
