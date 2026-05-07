---
created: 2026-05-06
modified: 2026-05-06
---

# Napari plugin

A multi-tab napari dock widget that walks users from raw 2D/3D microscopy file through the [[pipeline]] (segmentation, tracking, hierarchical feature extraction), then visualizes and analyzes results in the same viewer.

## Plugin lifecycle

Discovered via napari's manifest at `nellie_napari/napari.yaml`, which declares one command `nellie.loader` pointing to `nellie_napari:NellieLoader` and exposes a single widget named "Nellie". When the user opens the widget, napari instantiates `NellieLoader(viewer)`. The constructor immediately calls `add_nellie_plugins_to_menu(self)` (`nellie_napari/discover_plugins.py`), which uses `importlib.metadata.entry_points(group='nellie.plugins')` to find third-party Nellie sub-plugins, walks `viewer.window._qt_window.menuBar()` to locate "&Plugins", and injects a "Nellie plugins" submenu. Each discovered callable is wired to receive the loader as its argument.

This is a **private-API touch** (`_qt_window`) — fragile across napari versions.

## Widget topology

[[loader|`NellieLoader`]] subclasses `QTabWidget` and is the parent owner. Six tabs in order: Home, File validation, Process, Visualize, Analyze, Settings. The loader holds the canonical pipeline state (`im_info`, `im_info_list`, `current_version`, `latest_version`) and integer `*_tab` indices.

- **Process / Visualize / Analyze start disabled.** [[fileselect|`file_select.on_process()`]] calls `nellie.go_process()`, which copies `file_select.im_info` up to the loader, enables Process + Visualize, and triggers their `post_init`.
- **Analyze is gated separately** by [[processor|`processor.check_file_existence()`]], which flips `analysis_tab` enabled when `features_organelles` exists on disk.
- **`currentChanged` lazily calls `post_init`** on Visualize / Analyze / Settings the first time each is opened.

Every child widget keeps a back-reference `self.nellie = loader` and reaches up for shared state (notably `nellie.settings.<checkbox>`) — widgets are tightly coupled to the loader, not directly to each other.

## Pipeline bridge

The plugin **does not call `nellie/run.py`.** [[processor|`NellieProcessor`]] instantiates each stage class (`Filter`, `Label`, `Network`, `Markers`, `HuMomentTracking`, `VoxelReassigner`, `Hierarchy`) one `ImInfo` at a time inside `@thread_worker(ignore_errors=True)` generators from `napari.qt.threading`. `_start_worker` wires `started` → button-disable + `set_status`, `errored` → `_handle_worker_error`, `finished` → `_on_worker_finished` which re-runs `check_file_existence` and chains the next step iff `self.pipeline` is True and the prior step didn't error.

Per-step kwargs come from [[settings|`Settings.get_*_params()`]] at click time. Progress is reported through `napari.utils.notifications.show_info` plus a 500 ms `QTimer` that animates ellipses on a status `QLabel`. **There is no real progress bar** — pipeline stages are atomic to the UI.

## Subsystem-level gotchas

- **`add_nellie_plugins_to_menu` reaches into `viewer.window._qt_window`** and matches the literal string `"&Plugins"`. Napari ≥ 0.5 renames or restructures menus and this **silently no-ops**.
- **`NellieLoader.reset()` rebuilds every child widget from scratch but reuses `self.viewer`** — napari layers from prior sessions persist unless cleared manually. State-leak risk.
- **`NellieVisualizer._add_labels_initially_hidden`** documents a real napari bug: passing `visible=False` to `add_labels` while `ndisplay=3` crashes the Volume visual. The workaround (add visible, then hide) is load-bearing — see [[visualizer]].
- **All large arrays are loaded via `tifffile.memmap` in mode `"r+"` for label layers** so napari's painting tools can write back; failed memmap falls back to read-only with a status warning.
- **Layer names are hard-coded strings** ("Pre-processed", "Labels: Branches", "Labels: Organelles", "Mocap Markers", "Reassigned px: …", "Tracks: …"). Renaming a layer in the UI breaks identity-based lookups.
- **`Settings` has a parallel `SettingsConfig` dataclass and `to_config` / `apply_config` round-trip but nothing in-tree calls them** — they're for future preset save/load.
- **Version check** runs in a one-shot `QThread` on construction and writes `current_version` / `latest_version` onto the loader; `Home.set_update_status` is invoked from the worker's signal slot on the main thread. Dev / pre-release versions (setuptools-scm `*.devN+g…`) are explicitly grayed out, not flagged outdated.

## Per-widget articles

- [[loader]] — shell + lifecycle + plugin-menu injection
- [[fileselect]] — file/folder picker, axis overrides, OME-TIFF write
- [[processor]] — pipeline driver, threading, chaining
- [[settings]] — per-stage parameter forms
- [[visualizer]] — layer construction and track display
- [[analysis]] — post-pipeline data explorer + adjacency overlays

`nellie_home.py` is a splash tab (logo, links, screenshot button + `Ctrl-Shift-E` keybinding for screenshots to `im_info.screenshot_dir`). Folded into this hub instead of its own article — it's mostly chrome.
