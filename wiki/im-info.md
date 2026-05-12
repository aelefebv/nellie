---
created: 2026-05-06
modified: 2026-05-12
---

# Image metadata (`im_info`)

Normalizes heterogeneous microscopy file metadata (axes, pixel sizes, time intervals, channel) into a canonical OME-TIFF that every downstream stage can memory-map without re-parsing. The `nellie.im_info` package owns this layer:

- `FileInfo` (in `nellie/im_info/verifier.py`) — source-format reader, validation, and OME-TIFF writer.
- `ImInfo` (in `nellie/im_info/verifier.py`) — wraps a verified `FileInfo`, regenerates the canonical OME-TIFF on demand, and exposes `pipeline_paths` (the on-disk contract that connects the rest of the [[pipeline]]).
- `MetadataExtractor` Protocol + 5 per-format implementations in `nellie/im_info/extractors/` — the strategy seam that handles per-format dispatch (OME, ImageJ, ImageJ-tif-tags, ND2, raw-TIFF). Adding a new format is a single-file addition.
- `load_image(path)` — canonical one-call entry point for programmatic loading.
- `DimRes` TypedDict + `infer_t_axis` (in `nellie/im_info/types.py`) — shared types for the axes/dim layer.

## Canonical entry point

```python
from nellie.im_info import load_image
im_info = load_image("path/to/microscopy.tif")
# im_info.axes, im_info.shape, im_info.dim_res, im_info.pipeline_paths populated
```

`load_image` does the 4-step boot in one call: `FileInfo(path)` → `find_metadata()` → `load_metadata()` → `ImInfo.from_file_info(file_info)`. Use it everywhere except the napari fileselect widget, which intentionally splits the steps so the UI can surface auto-detected axes for user override before triggering ImInfo construction.

The lower-level API (`FileInfo` + `ImInfo.from_file_info`) is still public; reach for it when you need to inspect / mutate axes between metadata extraction and ImInfo construction.

## Inputs

- **First-class:** ND2 (Nikon, via `nd2`), OME-TIFF, ImageJ TIFF when `physicalsizex` is present in `imagej_metadata`.
- **Best-effort:** ImageJ TIFFs missing `physicalsizex` (fallback to raw TIFF tags), and plain TIFFs read entirely from page-0 tags (`XResolution`, `YResolution`, `ResolutionUnit`, optionally `ZResolution`, `FrameRate`).
- Non-`.tif/.tiff/.nd2` extensions raise `ValueError`.

## Metadata extracted

Each format gets its own `MetadataExtractor` implementation (`OmeExtractor`, `ImageJExtractor`, `ImageJTifTagExtractor`, `Nd2Extractor`, `RawTiffTagExtractor`). The factory `detect_extractor(filepath)` opens the file once and returns the right instance.

- Axes string and shape from the file's series.
- Per-axis physical sizes into `dim_res = {X, Y, Z, T}` (a `DimRes` TypedDict).
- OME pulls `physical_size_{x,y,z}` and `time_increment` directly.
- ND2 pulls XYZ from `volume.axesCalibration` (falling back to `channels[0].volume.axesCalibration`); T is the **median** of `np.diff(recorded_data["Time [s]"])` — the interval is preserved verbatim if regular, smoothed if jittery, and `None` for single-timepoint stacks.
- TIFF tag units are normalized to micrometers (CENTIMETER → ×1e4, INCH → ×25400); missing `ResolutionUnit` is left unscaled (treated as already-microns).

## Interactions

- The whole [[pipeline]] is downstream — every stage takes an `ImInfo` for shape/axes/`dim_res`/path lookup.
- Canonical in-memory layout is `T[Z]YX` with singleton Z squeezed; `no_z` / `no_t` flags drive 2D-vs-3D and single-frame branches throughout [[segmentation/index|segmentation]], [[tracking/index|tracking]], and the [[napari-plugin/index|napari plugin]].
- The [[fileselect|file-select widget]] is the user-facing surface: it constructs `FileInfo` directly (not via `load_image`), lets users override axes via combo boxes, and writes the OME-TIFF on confirm.
- `bioio` integration (queued separately) will slot in as a `BioioExtractor` — single-file addition, no FileInfo changes required, thanks to the extractor Protocol.

## Construction lifecycle

`FileInfo` and `ImInfo` constructors are **thin** (no I/O). Use the explicit boot for fine-grained control or `load_image` for one-shot use:

| Step | Method | Side effects |
|---|---|---|
| 1 | `FileInfo(path)` | Pure data — stores path-derived strings, no filesystem access. |
| 2 | `file_info.find_metadata()` | Calls `prepare_output_dirs()` to create output dirs. Opens file once via the factory; populates `self._extractor`, `self.axes`, `self.shape`, `self.metadata_type`. |
| 3 | `file_info.load_metadata()` | Calls `self._extractor.parse_dim_res()` to fill `self.dim_res`; runs `self._validate()` to populate `validation_errors` + `good_axes`/`good_dims`. |
| 4 | `ImInfo.from_file_info(file_info)` | Constructs ImInfo and calls `instance.load()` — regen-on-stale check, memmap creation, `_get_ome_metadata`, `_check_axes_exist`, `_create_output_paths`. |

The thin constructors exist so pure-logic tests can construct objects to inspect path computations without filesystem cost.

## Gotchas

- **Channel quietly collapses.** Channel `C` is allowed in `FileInfo` but stripped during `save_ome_tiff` after selecting `self.ch` (default 0). Multichannel files silently become single-channel. The `ImInfo` extractor seam (post-collapse) explicitly rejects `C` in axes — this is a deliberate layering boundary, not a bug.
- **Singleton Z is squeezed in `ImInfo` but preserved on disk.** `file_axes` and in-memory `axes` can differ; `transform_to_axes` (in `verifier.py`) reconciles per-stage memmaps.
- **TIFF tags without `ResolutionUnit` assume microns.** No warning.
- **ND2 with one timepoint yields `T=None`** (not 0). Downstream code special-cases this.
- **Mis-ordered axes are trusted from the source library** (`tifffile.series[0].axes`, `nd2.sizes`); fix via `file_info.change_axes(...)` if wrong. `infer_t_axis` will silently prepend `T` when shape has one extra leading singleton dim.
- **`output_naming="detailed"` bakes resolutions into filenames** (with `.`→`p`), so re-running with edited `dim_res` produces a new file rather than overwriting. The `"stable"` alternative keeps the original `filename_no_ext` — choose it when downstream tooling needs predictable paths across re-runs.
- **`ImInfo` silently regenerates the OME-TIFF on load** if it's missing *or* if the cached file's axes don't include `T`. Caches written before the T-normalization landed get rewritten on first re-open without warning — usually fine, but means an existing `ome_output_path` mtime can shift just from calling `ImInfo.from_file_info(file_info)`.
- **`tifffile` strips singleton axes on readback** after `save_ome_tiff` writes a `(1, Y, X)` array with `metadata={'axes': 'TYX'}`. The canonical contract for the saved axes is the provenance JSON's `output_axes` field plus OME's `pixels.size_*`, **not** `tifffile.series[0].axes`. Discovered during Slice 1 testing.

## Invariants

Post-`load_metadata`:

- `good_axes` requires axes length matches shape, axes drawn only from `{T,Z,C,Y,X}`, no duplicates, both X and Y present.
- `good_dims` requires every axis present in `axes` to have a non-None entry in `dim_res`.

Verifier behavior:

- **`_validate` is symmetric — never raises.** All errors flow through `validation_errors` + `good_axes`/`good_dims` flags. The user-facing mutators (`change_axes`, `change_dim_res`, `change_selected_channel`, `select_temporal_range`) still raise on invalid input at the entry point.
- **`change_axes` raises on length mismatch** (defensive validation for programmatic callers; napari widget pre-validates so it never triggers the raise).
- **`save_ome_tiff` refuses without both `good_axes` and `good_dims`.**
- **Silently fixes** leading-singleton T axes (`infer_t_axis`), ND2 timestamp jitter (median), TIFF unit conversion.

Post-`ImInfo.from_file_info`: in-memory array is always `T[Z]YX` with `T` first and Z absent if singleton.

Provenance (source axes, output axes, channel, t_start/t_end, `dim_res`) is JSON-serialized into the OME image description on save.

## Module layout

```
nellie/im_info/
├── __init__.py            # Re-exports FileInfo, ImInfo, DimRes, load_image
├── types.py               # DimRes TypedDict + infer_t_axis (the "axes" helper)
├── verifier.py            # FileInfo + ImInfo + transform_to_axes + _write_ome_tiff
└── extractors/
    ├── __init__.py        # Re-exports
    ├── protocol.py        # MetadataExtractor Protocol
    ├── ome.py             # OmeExtractor
    ├── imagej.py          # ImageJExtractor
    ├── imagej_tif_tags.py # ImageJTifTagExtractor (carries imagej_meta + tif_tags as named fields)
    ├── nd2.py             # Nd2Extractor
    ├── raw_tiff.py        # RawTiffTagExtractor
    ├── factory.py         # detect_extractor(filepath) — single file open per detection
    └── _tif_tags.py       # Private helper shared by raw + imagej_tif_tags
```

## Test coverage

Verifier behaviors are pinned by ~150 tests across:

- `tests/test_verifier_fileinfo.py` (~95 tests) — per-format `dim_res` extraction, `_write_ome_tiff` helper, `infer_t_axis`, validation contracts, mutator preconditions, `save_ome_tiff` round-trip.
- `tests/test_verifier_iminfo.py` (~68 tests) — construction (thin + `from_file_info`), `pipeline_paths` 18-key surface, `transform_to_axes`, `get_memmap`, `allocate_memory`, `remove_intermediates` (legacy shim) + `remove_marked_intermediates` + `DROPPABLE_KEYS` / preset constants (per-output retention; section H/H.1; see [[decisions/0014-intermediates-policy-frozenset]]), `load_image` orchestrator.
- `tests/test_extractors.py` (~29 tests) — per-extractor `parse_dim_res()` + `detect_extractor` factory dispatch.

4 of 5 `metadata_type` branches covered via per-format fixtures (`'ome'`, `'imagej'`, `'imagej_tif_tags'`, `None` × 3 RESUNIT cases); `'nd2'` deferred until an ND2 sample is checked in or `bioio` integration lands. See [[queue]].

## History

- The verifier's 8-slice dechaos refactor landed on 2026-05-09 (PRDs #119–#133, PRs #120–#134), driven by [[outputs/dechaos-verifier|`wiki/outputs/dechaos-verifier.md`]]. Net pyright impact: 93 → 38 errors on `verifier.py` (−55, 59% reduction). Test suite grew from 147 → 324 (+177 tests). The polymorphic `metadata` field (Pass 5's headline code smell) was eliminated; per-format extractors replaced the 5-way string dispatch.
- The original `change_axes` length gate was disabled by commit `492edfb` (Aug 2024) for napari fileselect partial-typing UX. Slice 2 restored it after confirming napari pre-validates length, making the verifier-level gate redundant for napari but defensive for programmatic callers.
- `_validate` was asymmetrically raising on time errors only (introduced by commit `5caffe6`, Jan 2026). Slice 5 made it symmetric (never raises) — `select_temporal_range` and the other user-facing mutators still raise directly on invalid input.
