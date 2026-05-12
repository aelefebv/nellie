---
created: 2026-05-12
modified: 2026-05-12
---

# Per-output intermediate retention policy is a `frozenset[str]` of `pipeline_paths` keys, not a structured dataclass

The legacy `ImInfo.remove_intermediates()` is all-or-nothing: every non-CSV
entry in `pipeline_paths` plus `im_path` is deleted, gated by the napari
Settings widget's single `remove_intermediates_checkbox`. The new
per-output retention design (see [[pipeline]] for stage→output mapping)
introduces a policy parameter that travels through both `nellie.run.run()`
and `nellie_napari.processor` so the napari widget and scripted callers
share one mechanism.

The decision is **the policy is a `frozenset[str]` of `pipeline_paths` keys
to drop**, validated against a module-level `DROPPABLE_KEYS: frozenset[str]`
constant (the universe of toggleable outputs: the 12 image-like
intermediates + `adjacency_maps` + `im_path`). Presets are module-level
`frozenset[str]` constants derived from `DROPPABLE_KEYS` (`KEEP_EVERYTHING_PRESET = frozenset()`,
`CSVS_ONLY_PRESET = DROPPABLE_KEYS`,
`MASKS_AND_CSVS_PRESET = DROPPABLE_KEYS - {<the three label maps> + im_path}`).
The cleanup method is `ImInfo.remove_marked_intermediates(drop_keys: frozenset[str])`,
which validates `drop_keys <= DROPPABLE_KEYS` and deletes each existing path.
The legacy `remove_intermediates()` becomes a one-line shim:
`self.remove_marked_intermediates(drop_keys=DROPPABLE_KEYS)` — preserving the
existing test contract (CSVs survive; `im_path` deleted).

Status: **Accepted** — to be implemented per the PRD this ADR was written
during.

## Considered Options

- **`frozenset[str]` of drop keys (chosen).** Minimal API surface; presets
  are derivable arithmetic on `DROPPABLE_KEYS`; new pipeline outputs only
  need a single `DROPPABLE_KEYS` addition and the presets re-derive
  automatically; legacy backward-compat shim is one line; immutable so it
  can be a default-arg or module constant safely; validation at the
  `ImInfo.remove_marked_intermediates` boundary catches typos with one
  `assert drop_keys <= DROPPABLE_KEYS` line.
- **`@dataclass IntermediatesPolicy` with one bool field per intermediate
  (rejected).** Type-safe and IDE-discoverable for scripted callers, but
  carries 14+ boolean fields that all need to be threaded through every
  preset constructor and updated whenever a new pipeline output is added.
  The discoverability advantage is real but the napari Settings widget is
  the actual discovery surface for end users; scripted callers reach for
  named preset constants 90% of the time, where the type doesn't matter.
- **`dict[str, bool]` mapping key → keep (rejected).** Redundant: each
  entry carries a boolean but only the "drop" answers matter. Forces a
  policy decision on missing-key semantics (keep, drop, or raise?) that
  `frozenset` sidesteps by construction. Mutable, so it can't safely be a
  module constant for presets without `MappingProxyType` ceremony.
- **Preset enum + override sets (rejected).** Two-layer state (named
  preset + diff sets) is more to think about than the final drop set
  itself; the napari widget would need to track which preset was last
  selected separately from the actual checkbox state. The `frozenset`
  collapses this — selecting a preset just sets the drop set; tweaking a
  checkbox updates the drop set; the widget reverse-derives "is this a
  named preset?" by membership comparison.

## Consequences

- **Public API contract is `frozenset[str]`** for `nellie.run.run(file_info, ..., cleanup_drop_keys: frozenset[str] | None = None)`
  and `SettingsConfig.cleanup_drop_keys: frozenset[str]`. Hard to reverse
  after release: scripted callers will pin the type; the napari widget's
  saved-settings JSON will serialize as a list and round-trip via
  `frozenset(...)`.
- **`DROPPABLE_KEYS` is the source of truth.** Adding a new output to
  `_create_output_paths` that should be droppable also requires adding
  the key to `DROPPABLE_KEYS`; forgetting this is a one-line mistake
  caught by the validation assert (the new key won't accept toggle
  attempts) but won't auto-show up in the napari widget either. Document
  alongside `_create_output_paths`.
- **CSV keys (`features_voxels`, `features_nodes`, `features_branches`,
  `features_organelles`, `features_image`) are NOT in `DROPPABLE_KEYS`.**
  Hard-protected because the napari analyzer reads them back; any user
  intent to delete CSVs is out-of-scope per Q2 of the design grilling.
- **Legacy `remove_intermediates()` is preserved** as a thin wrapper
  calling `remove_marked_intermediates(drop_keys=DROPPABLE_KEYS)`. The
  two existing tests (`test_remove_intermediates_preserves_csv_files`,
  `test_remove_intermediates_deletes_canonical_im_path`) stay green
  unchanged; they characterize the legacy semantic the wrapper preserves.
- **Cleanup fires once, after `Hierarchy.run()` succeeds** — same trigger
  point as today, no `try/finally`. Inherits the existing pipeline
  "no try/except around stages" gotcha (per [[pipeline]]); a mid-pipeline
  failure leaves all on-disk state for inspection.
