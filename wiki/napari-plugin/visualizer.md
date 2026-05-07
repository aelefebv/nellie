---
created: 2026-05-06
modified: 2026-05-06
---

# Visualizer widget

Per-stage layer loader (raw, [[filtering|preprocessed]], [[labelling|segmentation]], [[mocap-marking|mocap]], [[voxel-reassignment|reassigned]]) plus track visualization. Each "Open …" button is idempotent: reuses the existing layer if still in `viewer.layers`.

## Why

Maps the on-disk pipeline outputs (memmaps and CSVs) into napari layers without recomputing anything. Track visualization wraps `LabelTracks` (`tracking/all_tracks_for_label.py`) for per-label or all-label trajectories — see [[tracking/index|tracking hub]].

## Interactions

- Reads `settings.skip_vox` and `settings.track_all_frames` from the [[settings|settings widget]].
- Tracks come from `LabelTracks` for the active labels layer's `selected_label` or all labels, optionally across all frames (only meaningful for reassigned label layers).
- All large arrays are loaded via `tifffile.memmap` in mode `"r+"` so napari's painting tools can write back; failed memmap falls back to read-only with a status warning.

## Gotchas

- **`_add_labels_initially_hidden` is a load-bearing napari workaround.** Passing `visible=False` to `add_labels` while `ndisplay=3` crashes the Volume visual; the fix is to add visible, then hide.
- **Layer names are hard-coded strings** ("Pre-processed", "Labels: Branches", "Labels: Organelles", "Mocap Markers", "Reassigned px: …", "Tracks: …"). User-renamed layers break the active-layer lookup.
- **`_get_active_label_layer_and_path` uses `is` for identity, not name** — adding a layer copy via "Duplicate Layer" can confuse it.
- **Scale bar is forced to "um".** Independent of the actual `dim_res` units (which are also microns by convention — but this hardcodes the assumption).

## Invariants

- Layer names are stable and used as identity by other widgets.
- Image layers loaded with `scale=dim_res` so napari's measurements report physical units.
