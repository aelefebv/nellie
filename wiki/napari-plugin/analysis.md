---
created: 2026-05-06
modified: 2026-05-06
---

# Analysis widget

Post-pipeline data explorer over the [[feature-extraction|hierarchy CSVs]]. Three-stage dropdown (level → feature → statistic), matplotlib histogram canvas, mean/std vs median/IQR toggle, log scale, per-timepoint vs pooled (driven by `viewer.dims.events.current_step`), CSV/PNG export to `im_info.graph_dir`.

## Why

The hierarchy CSVs are five files with `feature_stat`-style columns; raw CSV inspection is impractical. This widget parses suffixes (via `STAT_SUFFIXES`) to surface "the feature without its stat" as one dropdown step, and the available stats as the next — making the multi-level structure navigable. The `overlay()` action joins stats across levels via `adjacency_maps.pickle` so users can paint, e.g., per-organelle mean velocity onto the voxel grid.

## Interactions

- Loads the five CSVs from [[feature-extraction]] lazily on first need.
- Loads `adjacency_maps.pickle` for `overlay()`.
- Adds layers to the napari viewer — `add_labels` for "reassigned" attributes, `add_image` (turbo colormap, 98th-percentile contrast) otherwise.
- Click callback `get_index` looks up which voxel/node/branch/organelle was clicked and populates a `QTableWidget`.

## Gotchas

- **`feature_map` cache rebuilt on every level change** — fast for small frames, slow for large ones.
- **`label_mask` is mutated in-place across overlays** — calling `overlay()` repeatedly accumulates `nan`-replaced state. Reset between explorations.
- **`reset()` exists but is not wired to [[loader|`NellieLoader.reset()`]]** — the loader recreates the widget instead. If you wire it up, verify it clears the in-place mask state.
- **`STAT_SUFFIXES` is the contract.** Adding a new aggregator in [[feature-extraction|`Hierarchy`]] requires updating this widget too.

## Invariants

- The feature dropdown values map 1:1 to columns in the loaded CSV.
- Overlay layer types are deterministic from the source attribute name (`reassigned*` → labels; everything else → image).
