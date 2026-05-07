---
created: 2026-05-06
modified: 2026-05-06
---

# Settings widget

Two-tab (Basic / Advanced) parameter form. Every pipeline stage has its own sub-tab with override-checkbox-gated optional spinboxes (`_make_optional_spinbox`), device combos (auto/cpu/gpu), and low-memory checkboxes.

## Why

Centralizes pipeline knobs so the [[processor]] can pull `get_<step>_params()` at click time. The advanced tab maps almost 1:1 onto stage-class constructor kwargs.

## Interactions

- [[processor]] calls `get_<step>_params()` per stage and reads basic-tab checkboxes (`remove_edges`, `analyze_node_level`, `voxel_reassign`, `remove_intermediates`) directly via `nellie.settings.<checkbox>.isChecked()`.
- Exposes a `to_config` / `apply_config` round-trip plus a `SettingsConfig` dataclass — for future preset save/load. **Nothing in-tree currently calls these.**

## Gotchas

- **Basic-tab checkboxes are shared mutable state**, not just a settings store. The processor reads them through the loader (`nellie.settings.<checkbox>`) at click time, so changing them mid-pipeline affects later stages.
- **Per-stage device combo is independent of the others** — easy to mix `gpu` and `cpu` in one run by accident.
- **`SettingsConfig` round-trip is dead code today.** Don't rely on it; if you wire it up, verify all override checkboxes round-trip too.

## Invariants

- `get_<step>_params()` returns a dict keyed by the stage's constructor kwargs.
- Optional spinboxes return `None` when their override checkbox is unchecked — this is how stages know to use their default.
