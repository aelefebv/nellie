---
created: 2026-05-06
modified: 2026-05-06
---

# File select widget

Single-file or folder (batch) input with axis-order combo boxes, T/Z/XY resolution fields with `QDoubleValidator`, and channel/start-end-frame spinners. Backed by [[im-info|`FileInfo`/`ImInfo`]]. The densest user-facing surface in the plugin.

## Why

Validates the **axis/scale contract** that the rest of the [[pipeline]] depends on. Without this, downstream stages would either crash on bad metadata or silently produce wrong physical units.

## Interactions

- Constructs `FileInfo` / `ImInfo`; on confirm calls `save_ome_tiff` (gated on `good_axes` + `good_dims`).
- `on_process` calls `nellie.go_process()` (on the [[loader]]), which copies `self.im_info` up and enables Process + Visualize tabs.
- `on_preview` memmaps the OME-TIFF and adds it as an image layer with µm scale.

## Gotchas

- **Batch mode requires identical axes + shape across files** — `initialize_folder` rejects mismatches. Mixed shapes can't be batched.
- **`_apply_to_each_file_info` swallows the first exception per action** — partial failure may not be visible.
- **Preview switches `viewer.dims.ndisplay` based on Z > 1** — flipping back to 2D requires manual viewer reset.
- **Many `blockSignals` toggles** to avoid cycles between the `dim_order` line-edit and combo-box widgets; tread carefully when adding fields.
- **Channel default is 0** (see [[im-info]] gotchas) — multichannel files quietly become single-channel in the OME-TIFF written here.

## Invariants

- A confirmed file always has `good_axes` and `good_dims` true on its `ImInfo`.
- Batch mode `im_info_list` entries share axes and shape.
