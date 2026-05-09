---
created: 2026-05-06
modified: 2026-05-09
---

# Image metadata (`im_info`)

Normalizes heterogeneous microscopy file metadata (axes, pixel sizes, time intervals, channel) into a canonical OME-TIFF that every downstream stage can memory-map without re-parsing. `FileInfo` (in `nellie/im_info/verifier.py`) does parsing and validation; `ImInfo` wraps a verified `FileInfo`, regenerates the canonical OME-TIFF on demand, and exposes `pipeline_paths` — the on-disk contract that connects the rest of the [[pipeline]].

## Inputs

- **First-class:** ND2 (Nikon, via `nd2`), OME-TIFF, ImageJ TIFF when `physicalsizex` is present in `imagej_metadata`.
- **Best-effort:** ImageJ TIFFs missing `physicalsizex` (fallback to raw TIFF tags), and plain TIFFs read entirely from page-0 tags (`XResolution`, `YResolution`, `ResolutionUnit`, optionally `ZResolution`, `FrameRate`).
- Non-`.tif/.tiff/.nd2` extensions raise `ValueError`.

## Metadata extracted

- Axes string and shape from the file's series.
- Per-axis physical sizes into `dim_res = {X, Y, Z, T}`.
- OME pulls `physical_size_{x,y,z}` and `time_increment` directly.
- ND2 pulls XYZ from `volume.axesCalibration` (falling back to `channels[0].volume.axesCalibration`); T is the **median** of `np.diff(recorded_data["Time [s]"])` — the interval is preserved verbatim if regular, smoothed if jittery, and `None` for single-timepoint stacks.
- TIFF tag units are normalized to micrometers (CENTIMETER → ×1e4, INCH → ×25400); missing `ResolutionUnit` is left unscaled (treated as already-microns).

## Interactions

- The whole [[pipeline]] is downstream — every stage takes an `ImInfo` for shape/axes/`dim_res`/path lookup.
- Canonical in-memory layout is `T[Z]YX` with singleton Z squeezed; `no_z` / `no_t` flags drive 2D-vs-3D and single-frame branches throughout [[segmentation/index|segmentation]], [[tracking/index|tracking]], and the [[napari-plugin/index|napari plugin]].
- The [[fileselect|file-select widget]] is the user-facing surface: it constructs `FileInfo` / `ImInfo`, lets users override axes via combo boxes, and writes the OME-TIFF on confirm.

## Gotchas

- **Channel quietly collapses.** Channel `C` is allowed in `FileInfo` but stripped during `save_ome_tiff` after selecting `self.ch` (default 0). Multichannel files silently become single-channel.
- **Singleton Z is squeezed in `ImInfo` but preserved on disk.** `file_axes` and in-memory `axes` can differ; `_normalize_memmap` reconciles per-stage memmaps.
- **TIFF tags without `ResolutionUnit` assume microns.** No warning.
- **ND2 with one timepoint yields `T=None`** (not 0). Downstream code special-cases this.
- **Mis-ordered axes are trusted from the source library** (`tifffile.series[0].axes`, `nd2.sizes`); fix via `change_axes()` if wrong. `_normalize_time_axis` will silently prepend `T` when shape has one extra leading singleton dim.
- **`output_naming="detailed"` bakes resolutions into filenames** (with `.`→`p`), so re-running with edited `dim_res` produces a new file rather than overwriting. The `"stable"` alternative keeps the original `filename_no_ext` — choose it when downstream tooling needs predictable paths across re-runs.
- **`ImInfo` silently regenerates the OME-TIFF on load** if it's missing *or* if the cached file's axes don't include `T` (verifier.py:765–772). Caches written before the T-normalization landed get rewritten on first re-open without warning — usually fine, but means an existing `ome_output_path` mtime can shift just from constructing `ImInfo`.

## Invariants

Post-`load_metadata`:

- `good_axes` requires axes length matches shape, axes drawn only from `{T,Z,C,Y,X}`, no duplicates, both X and Y present.
- `good_dims` requires every axis present in `axes` to have a non-None entry in `dim_res`.

Verifier behavior:

- **Rejects (raises)** only on out-of-bounds temporal range and unsupported file extensions.
- **Flags via `validation_errors` and boolean gates** for axis/dim problems — consumers must check (e.g. `save_ome_tiff` refuses without both gates true).
- **Silently fixes** leading-singleton T axes, ND2 timestamp jitter (median), TIFF unit conversion.

Post-`ImInfo` init: in-memory array is always `T[Z]YX` with `T` first and Z absent if singleton.

Provenance (source axes, output axes, channel, t_start/t_end, `dim_res`) is JSON-serialized into the OME image description on save.

Verifier behaviors are pinned by `tests/test_verifier_fileinfo.py` (76 tests) + `tests/test_verifier_iminfo.py` (46 tests) as of PRD #119 / PR #120 (2026-05-09). 4 of 5 `metadata_type` branches covered via per-format fixtures (`'ome'`, `'imagej'`, `'imagej_tif_tags'`, `None` × 3 RESUNIT cases); `'nd2'` deferred (see [[queue]]). The 8-slice refactor plan is in [[outputs/dechaos-verifier|`wiki/outputs/dechaos-verifier.md`]] Pass 8.

3 design quirks were confirmed intentional during the dechaos grilling (with git-archaeology citations) and are pinned with explicit "design quirk" comments in the test files:

- **Half-commented `change_axes` length gate** (verifier.py:423–426) — disabled by commit `492edfb` (Aug 2024) for napari fileselect partial-typing UX. The widget pre-validates length at `nellie_fileselect.py:868–874`; the verifier-level gate would crash the GUI on intermediate keystrokes.
- **`_validate` asymmetric raise** (verifier.py:537–538) — raises only on time errors; axis/dim errors are silently flagged via `good_axes`/`good_dims` for the napari widget to display. Time-range errors are programmer errors (caller passed bad `t_start`/`t_end`); axis/dim are user-input errors.
- **`_normalize_axes` excludes `C`; `_axis_errors` includes `C`** — layering boundary, not a contract drift. FileInfo accepts multichannel input; `save_ome_tiff` collapses C; ImInfo only ever reads post-collapse OME-TIFFs.

Discovered during test writing (worth flagging for Slice 7 / OME-TIFF writer extraction): **tifffile strips singleton axes on readback** after `save_ome_tiff` writes a `(1, Y, X)` array with `metadata={'axes': 'TYX'}`. The canonical contract for the saved axes is the provenance JSON's `output_axes` field plus OME's `pixels.size_*`, NOT `tifffile.series[0].axes`.
