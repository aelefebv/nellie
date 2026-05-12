---
created: 2026-05-06
modified: 2026-05-12
---

# Pipeline orchestration

A thin sequential driver in `nellie/run.py` that wires the seven pipeline stages together against one shared `ImInfo` handle. The [[processor|napari processor]] is the only in-tree caller; it **re-implements** the same sequence stage-by-stage rather than calling `run()`. (`run()` is also importable for scripted use.)

## Stage order

`run()` instantiates and executes (in order):

1. `Filter` — see [[filtering]] — Frangi vesselness preprocessing → `im_preprocessed`
2. `Label` — see [[labelling]] — instance segmentation → `im_instance_label`
3. `Network` — see [[networking]] — skeleton, pixel-class, branch-relabelled volume
4. `Markers` — see [[mocap-marking]] — motion-anchor seed points → `im_marker`
5. `HuMomentTracking` — see [[hu-tracking]] — frame-to-frame correspondence → `flow_vector_array`
6. `VoxelReassigner` — see [[voxel-reassignment]] — propagates labels through time
7. `Hierarchy` — see [[feature-extraction]] — voxel→node→branch→organelle→image stats

All inter-stage state lives **on disk**, addressed via `ImInfo.pipeline_paths` (a string-keyed dict pre-populated in `_create_output_paths`). Stages communicate by filename convention only — no Python objects pass between them. Internal arrays land under `nellie_necessities_output_path_no_ext` as OME-TIFF (with `.npy` for flow/match arrays); user-facing feature tables are CSV under `user_output_path_no_ext`.

## Interactions

- Built on top of [[im-info|ImInfo]], which must have had `find_metadata()` and `load_metadata()` called by the caller before `run()`.
- Imports `xp`, `ndi`, `is_gpu`, `device_type` from `nellie/__init__.py` (a platform-gated backend shim — see [[gpu-runtime]]).
- The [[processor|napari processor]] mirrors this stage list and order one-to-one. **Any change here must be mirrored there** or the UI silently diverges.

## Gotchas

- **No `try/except` around stages** — first failure aborts the chain and leaves partial outputs on disk; reruns silently overwrite. This is what gives the per-output cleanup mechanism its "only-on-success" semantics for free (see below).
- **Stage args are not uniform.** `device="auto"` and `low_memory` are passed to `Filter`, `Label`, `HuMomentTracking`, `Hierarchy`. `Network`, `Markers`, `VoxelReassigner` only get `device` — no `low_memory` knob from `run()`.
- **`Hierarchy` is invoked with `skip_nodes=False` hard-coded.** The napari [[settings|settings widget]] exposes it as a flag; `run()` doesn't.
- **macOS hard-pinned to CPU** at `nellie/__init__.py` (the MPS branch is commented out — see [[gpu-runtime]]).
- **Per-output cleanup runs at end-of-pipeline, only on success.** `run()` accepts `cleanup_drop_keys: frozenset[str] | None`; when truthy, calls `im_info.remove_marked_intermediates(drop_keys)` after `Hierarchy.run()` returns. Mid-pipeline failures propagate without cleanup, leaving partial state for inspection. `None` (default) and the empty `frozenset` are no-ops. Same trigger point in the napari [[processor]] (read from `SettingsConfig.cleanup_drop_keys`). See [[decisions/0014-intermediates-policy-frozenset]] and the [[glossary|DROPPABLE_KEYS]] entry.

## Invariants

- Caller must run `FileInfo.find_metadata()` then `load_metadata()` before passing to `run()` — `ImInfo` constructor reads `file_info.ome_output_path` and assumes `good_axes` / `good_dims` are resolved.
- Stage order is fixed and load-bearing. Each stage reads outputs of all prior stages by `pipeline_paths` key; reordering breaks file dependencies, not just semantics.
- Exactly one `ImInfo` is shared across the whole run; outputs are scoped per-(file, channel, temporal range) by the path scheme.
- `device` is forwarded uniformly so all stages agree on backend within a run.
