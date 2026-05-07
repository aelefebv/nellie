---
created: 2026-05-06
modified: 2026-05-06
---

# Loader (`NellieLoader`)

The `QTabWidget` shell and lifecycle root for the [[napari-plugin/index|napari plugin]]. Owns the canonical pipeline state (`im_info`, `im_info_list`, `current_version`, `latest_version`) and the six tab widgets.

## Why

Acts as the single source of truth for cross-tab state. Child widgets all reach up to `self.nellie = loader` rather than message each other directly — keeps coupling explicit but means the loader is the de-facto god-object of the plugin.

## Interactions

- Discovered via `napari.yaml` → `nellie.loader` command.
- Constructor calls `add_nellie_plugins_to_menu(self)` from `discover_plugins.py` to inject third-party plugins via `entry_points(group='nellie.plugins')`.
- Hosts the `VersionWorker` (one-shot `QThread`) that fetches the latest PyPI version and writes back to the loader; `Home.set_update_status` reads from it.
- `currentChanged` triggers lazy `post_init` on Visualize / Analyze / Settings the first time each is opened.

## Gotchas

- **`reset()` rebuilds every child widget from scratch but reuses `self.viewer`** — napari layers from prior sessions persist unless cleared manually.
- **Plugin-menu injection touches `viewer.window._qt_window` (private API)** and matches the literal string `"&Plugins"` — silently no-ops on napari versions that rename or restructure that menu.
- Tab-enable/disable is manual: `Process` and `Visualize` enabled by [[fileselect|`file_select.on_process`]]; `Analyze` enabled by [[processor|`processor.check_file_existence`]] when `features_organelles` exists on disk.

## Invariants

- One `NellieLoader` per napari viewer instance.
- `self.im_info` (single-file mode) and `self.im_info_list` (batch mode) are the only canonical references — child widgets read these, never re-derive.
