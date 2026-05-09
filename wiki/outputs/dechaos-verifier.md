---
created: 2026-05-09
modified: 2026-05-09
---

# Dechaos scan — `nellie/im_info/verifier.py` (`FileInfo` + `ImInfo`)

One-shot review of the **entry-point module** for every nellie pipeline run. Findings will inform the queued test-pinning slice (paired predecessor) — characterization tests written against the contract list this report surfaces.

**Reference templates**: prior dechaos reports were per-stage (`Filter`/`Label`/`Network`/`Markers`/`Hu`/`VoxelReassigner`/`Hierarchy`) and converged on a stage-shaped pattern: 1 fat orchestrator class + per-frame state + GPU/CPU backend cascade. **The verifier does not fit that mold.** It has no GPU path, no per-frame loop, no `adaptive_run` involvement. The structural problems here are fundamentally different — multi-format dispatch, scattered axes-normalization logic, and constructor-time I/O coupling. The refactor sequencing therefore looks unlike any prior stage's plan.

**Headline shape**:
- 1130 lines, 2 classes (`FileInfo` lines 18–696 ≈ 680 lines, `ImInfo` lines 698–1070 ≈ 370 lines), 1 module-level `__main__` driver (~60 lines).
- `FileInfo` does five jobs poorly separated: directory creation, multi-format metadata extraction (5 dispatch branches), axes normalization, multi-aspect validation (3 error sources), state mutation (channel/time/axes/dims), and OME-TIFF writing with provenance.
- `ImInfo` does four jobs poorly separated: cached-OME-TIFF reading + auto-regen, pipeline-path dict generation, memmap creation/normalization, and new-OME-TIFF allocation.
- **Three near-duplicate axes-normalization implementations**: `FileInfo._normalize_time_axis` (210), `ImInfo._normalize_axes` (889), `ImInfo._normalize_memmap` (931). Each handles "ensure T leading + maybe squeeze singleton Z" slightly differently.
- **Seven validation methods** with overlapping concerns: `_check_axes` (350), `_check_dim_res` (358), `_axis_errors` (366), `_dim_errors` (383), `_time_range_errors` (393), `_validate` (508), `get_validation_errors` (411).
- **Five metadata-extraction branches** dispatched by string `metadata_type` in `load_metadata` (332): `'ome'`, `'imagej'`, `'imagej_tif_tags'`, `'nd2'`, `None`. No shared interface; each `_get_*_metadata` takes a differently-shaped `metadata` arg.
- **Half-commented `change_axes` validation gate** at lines 423–426: `if len(new_axes) != len(self.shape):` is commented but `self.good_axes = False` (its body) runs unconditionally. Net effect: `_validate` later catches the length mismatch via `_check_axes`, but `self.axes` has already been mutated to the bad value and the dead `good_axes = False` is misleading dead-state.
- **Hardcoded path** in the `__main__` driver (line 1074): `/Users/austin/test_files/nellie_all_tests`. Not portable; the driver is the de-facto smoke test.
- **Critical contract zero-tests**: `tests/test_verifier_metadata.py` was wiped during the May 2026 scaffold rebuild and not restored. Verifier behaviors are entirely uncovered today.

---

## Pass 1 — System Map

- **Module location**: `nellie/im_info/verifier.py`. Re-exported as `from .verifier import FileInfo, ImInfo` in `nellie/im_info/__init__.py`.
- **Module call shape**: every pipeline entry point follows the same 4-step boot:
  1. `FileInfo(filepath)` — constructor creates `nellie_output/` and `nellie_necessities/` dirs (eager I/O).
  2. `file_info.find_metadata()` — dispatches on extension to `_find_tif_metadata` / `_find_nd2_metadata`; sets `self.metadata`, `self.metadata_type`, `self.axes`, `self.shape`; calls `_normalize_time_axis()`.
  3. `file_info.load_metadata()` — dispatches on `self.metadata_type` (5-way) into `_get_*_metadata` to fill `self.dim_res`; calls `_validate()`.
  4. `ImInfo(file_info)` — auto-regens OME-TIFF if missing or T-axis-stale; memmaps it; normalizes axes; builds `pipeline_paths` dict.
- **Call sites** (3):
  - `nellie/run.py:174–176` — script entrypoint, hardcoded `test_file` path; calls `FileInfo` → `find_metadata` → `load_metadata` → `ImInfo`. Most of `run.py:185–217` is commented-out debug.
  - `nellie_napari/nellie_fileselect.py:482, 555–556, 590–594` — the user-facing surface. Single-file path: `FileInfo` constructed in `select_file`, `find_metadata`/`load_metadata` called in `initialize_single_file`. Batch path: list of `FileInfo` constructed at `:589–591`, then `find_metadata`/`load_metadata` looped for each. Compatibility check across batch (axes + shape match) at `:597–610`.
  - `tests/conftest.py:90–96` — `_build_iminfo` helper used by `imageinfo_3d` / `imageinfo_2d` session fixtures. Copies fixture into `tmp_path` then runs the 4-step boot.
- **Mutators called from napari widget** (5): `change_axes` (878), `change_dim_res("T"/"Z"/"X"/"Y", value)` (913/933/951/952), `change_selected_channel` (967), `select_temporal_range` (981). All wrapped in `lambda file_info: ...` closures and applied across single-file or batch.
- **`ImInfo` consumers** — every algorithmic stage (`Filter`, `Label`, `Network`, `Markers`, `Hu`, `VoxelReassigner`, `Hierarchy`) takes `im_info: ImInfo` and pulls memmaps via `im_info.get_memmap(im_info.pipeline_paths[key])`, allocates via `im_info.allocate_memory(...)`. The `pipeline_paths` dict is the on-disk contract that connects the entire pipeline.
- **External deps**: `nd2` (Nikon ND2 reader), `tifffile` (TIFF read/write/memmap, ResolutionUnit constants), `ome_types` (OME XML parse/serialize), `numpy`, `nellie.utils.base_logger`. **No `adaptive_run` involvement** — verifier is GPU-agnostic.
- **Tests**: zero. Verifier is exercised transitively by every other test fixture (which depends on `_build_iminfo`), but no test pins verifier-specific behaviors. The legacy `tests/test_verifier_metadata.py` was wiped during the May 2026 scaffold rebuild per [[queue]].
- **Architectural shape**: not a "stage" in the pipeline sense. No backend dispatch, no per-frame loop, no Config dataclass, no `adaptive_run.run()` outer cascade. Constructor-heavy: both classes do real I/O at `__init__` time (FileInfo creates dirs; ImInfo opens `tifffile.TiffFile` and may write a new OME-TIFF). State is built up via mutator-call ordering: `find_metadata` → `load_metadata` → optional `change_*` → optional `save_ome_tiff`.

## Pass 2 — Boundary Scan

| # | Mixed concern | Where | Suggested seam |
|---|---|---|---|
| 1 | I/O at construction time | `FileInfo.__init__` (104–145) — `os.makedirs(self.output_dir)` and `os.makedirs(self.nellie_necessities_dir)` happen before any caller has a chance to inspect/cancel. `ImInfo.__init__` (752–792) opens `tifffile.TiffFile`, may call `file_info.save_ome_tiff()` (a full file write), creates a memmap, then walks the OME XML. | Move dir creation to a `prepare_output_dirs()` method called explicitly by `find_metadata` or by the fileselect widget. Move ImInfo's auto-regen + memmap creation behind a `load()` / classmethod constructor (`ImInfo.from_file_info(file_info)`). Constructor should be cheap; explicit `load()` does the work. **Tests will need this seam to characterize without filesystem side-effects.** |
| 2 | Multi-format metadata extraction lacks a strategy seam | `find_metadata` (194–208) does extension-based dispatch (`.tif/.tiff` vs `.nd2`); `load_metadata` (332–348) does string-key dispatch over `self.metadata_type` into 5 different `_get_*_metadata` methods. Each method takes a differently-shaped `metadata` arg (OME object vs dict vs `[dict, dict]` vs `{root, recorded_data}` dict vs raw TIFF tags). | Define a `MetadataExtractor` Protocol with `read(path) -> (metadata, metadata_type, axes, shape)` and `parse_dim_res(metadata) -> dict[str, float \| None]`. Per-format implementations: `OmeExtractor`, `ImageJExtractor`, `ImageJTifTagExtractor`, `Nd2Extractor`, `RawTiffTagExtractor`. The `imagej_tif_tags` case (line 165–166 wraps as `[dict, dict]`) is the smoking gun — it exists because the fallback needs *both* the imagej dict AND the raw tags, but is shoehorned into a list. A proper extractor class would carry both as fields. **Eligible after the test-pinning slice locks per-format `dim_res` outputs.** |
| 3 | Three near-duplicate axes-normalization implementations | `FileInfo._normalize_time_axis` (210–216): "if axes lacks T and shape has 1 extra leading singleton dim, prepend T to axes". `ImInfo._normalize_axes` (889–929): "ensure T is first (prepend or moveaxis); squeeze singleton Z; validate axes ⊆ {T,Z,Y,X}; transpose to canonical T[Z]YX". `ImInfo._normalize_memmap` (931–965): "ensure T is first (prepend or moveaxis); squeeze Z if `'Z' not in self.axes`; validate axes match `self.axes`; transpose to match". | Extract a single `normalize_to_canonical(data, source_axes, target_axes_pattern) -> (data, axes)` function. The 3 callers differ in: (a) does `data` exist or are we just rewriting axes, (b) what's the target pattern (T+rest vs T[Z]YX), (c) what to do with Z (always-squeeze-singleton vs squeeze-only-if-target-lacks-Z). All three can express their behavior as a parameterization of one normalizer. **Highest-value extraction in this file**, but needs characterization tests first because each existing path has subtle behavior (`_normalize_time_axis` only touches axes string and silently no-ops if T already present; `_normalize_axes` raises on missing Y/X; `_normalize_memmap` raises on Z>1 when target lacks Z). |
| 4 | Validation cascade conflates check, mutate, and write-paths | `_validate` (508–539) calls `_check_axes` → `_check_dim_res` → conditionally mutates `t_start`/`t_end` to defaults → calls `_time_range_errors` → raises only on time errors → calls `_get_output_path()`. Net: validation sets multiple flags, mutates state, writes paths, and may raise on a *subset* of errors. The `_check_*` methods themselves dual-write `good_axes`/`good_dims` and return error lists. `get_validation_errors` (411–412) recomputes the same error lists without setting flags — second source of truth. | Split into: (a) `compute_errors() -> ValidationReport` pure function (uses `_axis_errors` + `_dim_errors` + `_time_range_errors`), (b) `apply_defaults()` separate mutator for the `t_start`/`t_end` default-fill, (c) `_get_output_path()` called on demand. The `_check_*` methods can keep their dual-write, but `_validate` should not raise asymmetrically (raises on time, silent on axes/dims). **Resolve the asymmetric-raise contract first** — see Pass 5. |
| 5 | Half-commented `change_axes` validation gate | 423–426 (lines numbered as 423–426 in the file — Read offsets to 421–424 in 0-indexed view): the `if len(new_axes) != len(self.shape):` is commented but its body `self.good_axes = False` runs unconditionally. The `return` and `raise ValueError(...)` are also commented. Net: `self.axes = new_axes` mutates to a bad value; `_validate` later catches it via `_check_axes`'s "Axes length does not match data shape" error and resets `good_axes` correctly. End state is internally consistent but `self.axes` is left bad. | Either restore the gate (re-enable the `if`/`return`) OR delete the dead `self.good_axes = False` and let `_validate` own the flagging. Picking up from PR-history-style: was probably commented because callers were passing valid lengths, the gate fired unexpectedly, and someone disabled it. **Decide via git-archaeology before the test-pinning slice — what was the original failing case?** |
| 6 | `_get_tif_tags_metadata` reads `self.axes` mid-extraction | 270, 273 — `if 'Z' in self.axes` and `if 'T' in self.axes` gates within an extraction method. Means the method has an implicit precondition on extraction order. | Refactor signature to `extract(metadata, axes) -> dict`. Pure function; explicit dependency. |
| 7 | Logging mixed with file I/O | `read_file` (541–572) logs error messages on `tifffile.imread` failure. `save_ome_tiff` (620–695) logs error message on shape mismatch. Logging is structured against `nellie.utils.base_logger` — not unreasonable, but mixed with the `raise ValueError` pattern. | Acceptable cohesion. Boundary-only logging at a true edge of the system. Keep. |
| 8 | Module-level `__main__` driver | 1073–1130 — instantiates `FileInfo(test_file)`, calls `find_metadata`/`load_metadata`, prints a debug dump, calls `change_axes('TZYX')`, `change_dim_res('T', 0.5)`/`('Z', 0.2)`, `select_temporal_range(1, 3)`, then constructs `ImInfo(file_info)`. Hardcoded path `/Users/austin/test_files/nellie_all_tests`. | **Delete** — same disposition as Network/Markers/Hu/VoxelReassigner/Hierarchy `__main__` blocks. Replace with characterization tests in the paired test-pinning slice. |
| 9 | `ImInfo` auto-regen at construction | 765–772 — checks if `im_path` exists; if not (or if cached file lacks T-axis but `file_info.axes` has T), calls `file_info.save_ome_tiff()`. Silent file write. Mtime shifts on any first re-open of pre-T-normalization caches (per [[im-info]] gotchas). | Move to explicit method call (`ImInfo.ensure_canonical()`); make the constructor pure-construction. Same as #1. |
| 10 | `ImInfo._check_axes_exist` only sets `no_z`/`no_t` to False | 794–803 — never resets to True. If you reuse the object (e.g. after axes change), flags are stale. | The flags are computed once at `__init__` and then read by every downstream stage; reuse-after-mutation is not a supported workflow, but the asymmetric set is a footgun. Reset both to `True` at the top of the method, then conditionally set `False`. |

## Pass 3 — Responsibility Scan

`FileInfo` and `ImInfo` are both overloaded but in different ways. The split between them is **not** along clean responsibility lines.

### `FileInfo` (~680 lines)

| Responsibility | Methods | Notes |
|---|---|---|
| Path/output-dir management | `__init__` (104–145), `_get_output_path` (574–618) | Dir creation should move out of constructor (Pass 2 #1). Path generation is pure-ish but reads `self.output_naming`, `self.axes`, `self.dim_res`, `self.t_start`, `self.t_end`, `self.ch`. |
| Multi-format metadata extraction | `_find_tif_metadata` (147), `_find_nd2_metadata` (179), `find_metadata` (194), `_get_imagej_metadata` (218), `_get_ome_metadata` (232), `_get_tif_tags_metadata` (246), `_get_nd2_metadata` (277), `load_metadata` (332) | 5 extractor methods + 2 dispatch methods. Per-format strategies are not interchangeable (different arg shapes). Pass 2 #2. |
| Axes normalization | `_normalize_time_axis` (210–216) | Too small to count as its own class but is one of three near-duplicates. Pass 2 #3. |
| Validation | `_check_axes` (350), `_check_dim_res` (358), `_axis_errors` (366), `_dim_errors` (383), `_time_range_errors` (393), `_validate` (508), `get_validation_errors` (411) | 7 methods, overlapping concerns. Pass 2 #4. |
| User-facing state mutators | `change_axes` (414), `change_dim_res` (430), `change_selected_channel` (448), `select_temporal_range` (475) | All trigger `_validate()`. `change_axes` has the half-commented gate. `select_temporal_range` raises directly (different from `change_axes`/`change_dim_res` which just flag). |
| File reading | `read_file` (541–572) | Memmap-then-fallback-to-imread for TIFF; `nd2.imread` for ND2. Side-effect: sets `self.dtype`. |
| OME-TIFF writing with provenance | `save_ome_tiff` (620–695) | 75 lines. Reads file, manipulates axes (slice channel, slice time, ensure T leading), writes OME-TIFF, then re-opens to inject `dim_res` + provenance JSON. The provenance schema is implicit (Pass 5). |

**Single-summary-sentence test**: "FileInfo extracts metadata from microscopy files and writes a canonical OME-TIFF." That sentence elides validation, state-mutation, and dir-management. Three subclasses suggested:
- `MetadataReader` — extraction strategies + dispatch.
- `MetadataValidator` — error sources + report.
- `OmeTiffWriter` — provenance + axes normalization for write.
- (Path management stays as a small helper.)

### `ImInfo` (~370 lines)

| Responsibility | Methods | Notes |
|---|---|---|
| Cached OME-TIFF reading + auto-regen | `__init__` (752–792) | Auto-regen is a constructor side-effect. Pass 2 #1 #9. |
| Pipeline-path dict | `create_output_path` (805–828), `_create_output_paths` (830–854) | Hardcoded list of 18 keys. The on-disk contract for the whole pipeline. |
| Memmap creation/normalization | `get_memmap` (967–990), `_normalize_memmap` (931–965) | One of three near-duplicate normalizers (Pass 2 #3). |
| New OME-TIFF allocation | `allocate_memory` (992–1070) | 80 lines. Writes empty file or actual data, then re-opens to inject metadata + description + provenance. Significant duplication with `FileInfo.save_ome_tiff` (Pass 6 #2). |
| Axes normalization at load | `_normalize_axes` (889–929), `_get_ome_metadata` (870–887), `_check_axes_exist` (794–803) | Three small methods, all called from `__init__`. |
| Intermediate cleanup | `remove_intermediates` (856–868) | 13 lines, single-purpose. Reads from `pipeline_paths` dict. |

**Single-summary-sentence test**: "ImInfo wraps a verified FileInfo, owns the canonical memmap, and provides per-stage path/memmap accessors." Reasonable. Suggested split is lighter than FileInfo:
- Constructor cleanup (no I/O at `__init__`, see Pass 2 #1 #9).
- `_normalize_memmap` joins the unified normalizer.
- `allocate_memory` and `FileInfo.save_ome_tiff` share an `OmeTiffWriter` (or at least a `_write_ome_tiff_with_metadata(path, data, axes, dim_res, description) -> None` helper).

## Pass 4 — Dependency Scan

| # | Problem | Where | Impact | Suggested change |
|---|---|---|---|---|
| 1 | Hardcoded path | `__main__` line 1074 — `/Users/austin/test_files/nellie_all_tests` | Driver only runs on author's machine. Not portable. Per [[queue]], several other modules had this and were deleted. | Delete `__main__` block (Pass 2 #8). |
| 2 | Implicit call-order | `change_axes`, `change_dim_res`, `change_selected_channel`, `select_temporal_range` all assume `find_metadata`/`load_metadata` have run. `change_selected_channel` raises if `not self.good_dims or not self.good_axes`. The others rely on `self.axes`/`self.shape` being set. | Callers must remember the boot sequence. Tests will need to reproduce it. | Document the precondition in each method's docstring (low-risk, adds a Clarify item). Or introduce a `loaded` state flag and assert at each mutator entry (medium-risk). |
| 3 | I/O at construction | Both constructors. See Pass 2 #1. | Cannot test without filesystem. | Move I/O to explicit methods. **High-value for testability — pair with characterization tests.** |
| 4 | `_get_tif_tags_metadata` depends on `self.axes` mid-call | 270 — `if 'Z' in self.axes:` | Implicit ordering: this method must run AFTER `find_metadata` (which sets axes). | Make `axes` an explicit parameter (Pass 2 #6). |
| 5 | `ImInfo._normalize_memmap` depends on `self.axes` | 949 — `if 'Z' in axes_list and 'Z' not in self.axes:` | Method called from `get_memmap` which is called by every downstream stage. The `self.axes` field is set in `_get_ome_metadata` (called from `__init__`) — so as long as `ImInfo` was constructed, this is fine. But it means `_normalize_memmap` is not pure. | Make `target_axes` an explicit parameter; call sites pass `self.axes`. |
| 6 | `_check_axes_exist` only sets `no_z`/`no_t` to False | 800–803 — never resets to True. Initialized to True at `__init__` line 787–788. | Stale flags if reused after mutation. | See Pass 2 #10. |
| 7 | Hardcoded pipeline-path keys | `_create_output_paths` (830–854) — 18 hardcoded keys. Every stage knows them. | Adding a new pipeline stage requires editing this method AND every consumer that reads `pipeline_paths[key]`. | Acceptable for now (consumers already need to know the key). Long-term: a registry where each stage registers its inputs/outputs. **Don't tackle in dechaos sequencing — out of scope.** |
| 8 | Logger as global | `from nellie.utils.base_logger import logger` (15) — used at module level in `read_file` and `save_ome_tiff` | Standard practice in this codebase. | Keep. |
| 9 | `ImInfo` reads `file_info.ome_output_path` at construction | 764 — `self.im_path = file_info.ome_output_path`. `ome_output_path` is set as a side-effect of `_get_output_path()` which is called from `_validate()`. | Means `ImInfo(file_info)` requires `file_info.load_metadata()` to have run successfully. | Already implicit in the boot sequence; doc-string Clarify. |

## Pass 5 — Contract Scan

The **highest-leverage** dechaos pass for this file. Most of the testability/maintainability issues trace back to fuzzy contracts.

### `FileInfo.metadata` is type-polymorphic by `metadata_type`

| `metadata_type` | `metadata` shape |
|---|---|
| `'ome'` | `ome_types.OME` object |
| `'imagej'` | `dict` of imagej tags |
| `'imagej_tif_tags'` | `[dict_imagej, dict_tif_tags]` (list of two dicts) |
| `'nd2'` | `{'root': nd2.metadata, 'recorded_data': nd2.events_dict}` |
| `None` | `dict` of raw tifffile tags |

The `_get_*_metadata` methods know which shape they expect; callers route by string-tag dispatch. Adding a new format requires (a) a new string tag, (b) a new dispatch branch in `find_metadata`, (c) a new dispatch branch in `load_metadata`, (d) a new `_get_*_metadata` method. **No type for `metadata` field; no shared base class for the extractors.**

### `_get_*_metadata` methods all mutate `self.dim_res` in-place

All four methods write `self.dim_res['X']`, `['Y']`, `['Z']`, `['T']` directly. None return a value. `dim_res` is initialized to `{X: None, Y: None, Z: None, T: None}` in `load_metadata` (336) before dispatch; methods conditionally fill it in. Cannot be tested without instance state.

### `change_axes` documented to validate; in practice does not (much)

```python
def change_axes(self, new_axes):
    """Changes the axes string and revalidates the metadata."""
    # if len(new_axes) != len(self.shape):
    self.good_axes = False
        # return
        # raise ValueError('New axes must have the same length as the shape of the data')
    self.axes = new_axes
    self._validate()
```

Docstring says "revalidates". What it actually does: silently sets `good_axes = False`, mutates `self.axes` (even if invalid), runs `_validate`. Net: bad axes leave `self.axes` in a bad state but `good_axes` correctly flags it. Docstring + indentation suggest the original intent was a guard clause. **Either restore the guard or delete the dead `good_axes = False` and update the docstring.** Pass 2 #5.

### `_validate` raises asymmetrically

```python
if time_errors:
    raise ValueError(time_errors[0])
self._get_output_path()
```

Time errors raise. Axis and dim errors are stored in `self.validation_errors` and flagged via `good_axes`/`good_dims` gates — silent. `_get_output_path()` runs even when axes/dims are invalid, which means `ome_output_path` may be set against a half-baked filename. Three contracts blur:
1. "Validation flags errors via boolean gates" (axes, dims).
2. "Validation raises on errors" (time).
3. "Validation always succeeds in writing the path" (regardless of errors).

**Pick one model.** Recommend: never raise from `_validate`; let callers check `validation_errors`. Time-range mutators (`select_temporal_range`) already raise on bad input directly — that's the right place.

### `read_file` return type

Documented as `np.ndarray`. Actually returns:
- `tifffile.memmap(path)` for TIFF (memory-mapped — semantically a `numpy.memmap`).
- `tifffile.imread(path)` fallback (true `np.ndarray`).
- `nd2.imread(path)` for ND2 (numpy-compatible array).

All three implement the `np.ndarray` interface, but the memmap-vs-array distinction matters for memory and write semantics. Caller may try to mutate and find writes don't persist (or find writes DO persist if mode is `r+`). **Clarify in docstring; type annotation as `np.ndarray | np.memmap`.**

### `pipeline_paths` is a string-keyed dict with 18 magic keys

Keys: `'im_preprocessed'`, `'im_instance_label'`, `'im_skel'`, `'im_skel_relabelled'`, `'im_pixel_class'`, `'im_marker'`, `'im_distance'`, `'im_border'`, `'flow_vector_array'`, `'voxel_matches'`, `'im_branch_label_reassigned'`, `'im_obj_label_reassigned'`, `'features_voxels'`, `'features_nodes'`, `'features_branches'`, `'features_organelles'`, `'features_image'`, `'adjacency_maps'`. No type, no enum. Every stage hard-codes its keys. Already covered by [[pipeline_paths]] glossary entry.

Acceptable today (cohesion is reasonable — every consumer already knows the key it needs). Don't tackle in dechaos sequencing.

### Provenance JSON in OME image description

```python
provenance = {
    "source_axes": self.axes,
    "output_axes": axes,
    "dim_res": {key: _normalize_value(val) for key, val in self.dim_res.items()},
    "channel": self.ch,
    "t_start": self.t_start,
    "t_end": self.t_end,
}
ome.images[0].description = json.dumps(provenance, sort_keys=True)
```

Implicit schema — no version, no validator. Future readers must trust the keys. **Document the provenance schema in [[im-info]]; consider versioning (`{"version": 1, ...}`) before any field additions.** Reading provenance back is not implemented anywhere (no tests, no callers reading `ome.images[0].description`).

### `dim_res` keys are uppercase string axes; `axes` is a string

`dim_res = {'X': None, 'Y': None, 'Z': None, 'T': None}`. `axes = 'TZYX'`. Membership test is `dim in self.axes`. Convention works but is uncodified. **TypedDict candidate**: `class DimRes(TypedDict): X: float | None; Y: float | None; Z: float | None; T: float | None`. Eligible immediately, low-risk.

### Errors from each `_*_errors` method are bare strings

```python
errors.append('Axes length does not match data shape')
errors.append('Axes must only use T, Z, C, Y, X')
errors.append('Axes must not contain duplicates')
errors.append('Axes must include both X and Y')
errors.append(f"Missing {dim} resolution")
errors.append('Temporal range must be >= 0')
errors.append('Start frame must be <= end frame')
errors.append('Temporal range out of bounds')
```

UI may want to localize, programmatic callers may want to dispatch on error kind. Today: regex-match the string. **Low priority** — but a `ValidationError` enum or namedtuple would be a small Stabilize win.

### `axes` and `axes_list` swap representation

`self.axes` is a string; inside `_normalize_axes`/`_normalize_memmap` it becomes `axes_list = list(axes)`. Some methods take a string, some return a string, some operate on lists internally. **Pick one canonical representation in the unified normalizer.**

## Pass 6 — Composability Scan

| # | Trapped logic | Where | Extractable as |
|---|---|---|---|
| 1 | Per-format metadata extraction trapped in 5-way dispatch | `load_metadata` (332), `_get_*_metadata` (218/232/246/277) | `MetadataExtractor` Protocol with one impl per format. See Pass 2 #2. |
| 2 | OME-TIFF write-with-metadata pattern duplicated | `FileInfo.save_ome_tiff` (620) writes file + re-opens to inject `dim_res` + provenance JSON. `ImInfo.allocate_memory` (992) does the same: writes file + re-opens to inject `dim_res` + description. ~30 lines of overlap. | `_write_ome_tiff(path, data, axes, dim_res, description, provenance=None) -> None`. Both callers reduce to the call + their format-specific data prep. |
| 3 | Three near-duplicate axes-normalization implementations | `_normalize_time_axis`, `_normalize_axes`, `_normalize_memmap` | One `normalize_to_canonical(data, source_axes, target_pattern, squeeze_singleton_z=...) -> (data, axes)`. See Pass 2 #3. |
| 4 | Axes manipulation in `save_ome_tiff` | 632–658 — channel-take, time-take, T-prepend, T-as-first reordering | Could share with #3's normalizer if generalized. But probably keep separate (write-side concerns include channel collapse, which read-side doesn't). |
| 5 | Path generation with stringification/round/dot-replace | `_get_output_path` (574–618) | Already pure-ish; could be a free function `build_output_name(filename, axes, dim_res, ch, t_start, t_end, naming) -> str`. Easy extraction; saves ~30 lines from FileInfo. |
| 6 | The 4-step boot sequence (`FileInfo` → `find_metadata` → `load_metadata` → `ImInfo`) | Repeated at each call site | A `load_image(path) -> ImInfo` orchestrator function. Each call site reduces from 4 lines to 1. **Eligible immediately; no internal refactor needed.** |
| 7 | Error-list pattern repeats | 7 validation methods all `errors = []; if ...: errors.append(...); return errors` | A `Validator` class that accumulates errors via `validator.check(condition, message)` would DRY this, but the current pattern is fine for 7 methods. Don't tackle. |

## Pass 7 — Testability Scan

**Current state**: zero tests. `tests/test_verifier_metadata.py` was wiped during the May 2026 scaffold rebuild and not restored. Verifier behaviors are exercised transitively by every other test fixture (which depends on `_build_iminfo` in `tests/conftest.py:90–96`), but no test pins verifier-specific behaviors.

**Why testing is hard today**:
- Constructors do I/O (Pass 2 #1, #9).
- Multi-format extraction needs real files in 5 different formats.
- Validation is interleaved with state mutation.
- The 3 normalization methods have non-obvious overlaps and edge cases.

**What characterization tests need to pin** (informs the paired test-pinning slice in [[queue]]):

### Format-specific `dim_res` extraction (5 paths)

Need fixture files for each of:
- `'ome'` — proper OME-TIFF with `physical_size_{x,y,z}` + `time_increment` set.
- `'imagej'` — ImageJ TIFF with `physicalsizex` in `imagej_metadata` (and `physicalsizey`, `spacing`, `finterval`).
- `'imagej_tif_tags'` — ImageJ TIFF *missing* `physicalsizex` so the fallback to TIFF tags fires (line 164–166). Verify both extractors run and combine.
- `'nd2'` — ND2 with `volume.axesCalibration` set (test the fallback to `channels[0].volume.axesCalibration` separately if possible).
- `None` — raw TIFF with `XResolution`/`YResolution`/`ResolutionUnit` tags. Test all three units (none/CENTIMETER/INCH).

Per-format edge cases to pin:
- ND2 with single timepoint → `T=None` (not 0). Pin against [[im-info]] gotcha.
- ND2 with jittery timestamps → median used. Pin the median calc.
- ImageJ TIFF without `spacing` → `Z=None`.
- Raw TIFF without `ResolutionUnit` → no scaling applied (assumes microns).
- Raw TIFF with `ZResolution` but no `Z` axis → no `Z` set (gated by `'Z' in self.axes`).
- Raw TIFF with `FrameRate` but no `T` axis → no `T` set (gated by `'T' in self.axes`).

### Axes normalization

**`_normalize_time_axis` (FileInfo)**:
- axes='ZYX', shape=(1,16,512,512) → axes='TZYX' (leading singleton triggers prepend).
- axes='ZYX', shape=(16,512,512) → axes='ZYX' unchanged (lengths match, no T injected).
- axes='TZYX', any shape → unchanged (T already present).
- axes=None → unchanged (early return).

**`_normalize_axes` (ImInfo)**:
- axes='YX', data.shape=(512,512) → expand T → axes='TYX', data.shape=(1,512,512).
- axes='ZYX', data.shape=(1,512,512) → expand T + squeeze Z → axes='TYX', data.shape=(1,512,512). (Z is squeezed because singleton.)
- axes='ZYX', data.shape=(16,512,512) → expand T → axes='TZYX', data.shape=(1,16,512,512). (Z preserved because not singleton.)
- axes='ZTYX', data.shape=(16,2,512,512) → moveaxis T to 0, then transpose to TZYX → data.shape=(2,16,512,512).
- axes='YX', no Y → raises `ValueError("Axes must include both Y and X")`.
- axes='CZYX' (with C) → raises `ValueError("Unsupported axes found: ['C']")` because C is not in `_normalize_axes`'s allowed set (but IS in `FileInfo._axis_errors`'s allowed set — the contracts diverge!).

**`_normalize_memmap` (ImInfo)**:
- target axes 'TYX', file_axes 'YX' → expand T.
- target axes 'TYX', file_axes 'ZYX', data.shape=(1,512,512) → expand T + squeeze Z.
- target axes 'TYX', file_axes 'ZYX', data.shape=(16,512,512) → raises `ValueError("Z axis present with size > 1, but ImInfo expects no Z axis")`.
- target axes 'TZYX', file_axes 'TZYX' → no-op.
- file_axes=None → returns memmap as-is.

### Validation contracts

**`_axis_errors`**: 5 distinct error messages, each triggered by a specific input shape. Need 1 test per error path.

**`_dim_errors`**: missing dimension errors (returns 0–4 errors depending on `axes` content vs `dim_res` content).

**`_time_range_errors`**: 4 distinct error paths: t_start < 0, t_end < 0, t_start > t_end, range out of bounds. Plus 3 silent-no-op paths: axes=None, no T axis, t_start/t_end is None.

**`_validate`**: pin the asymmetric-raise (raises only on time errors, silent on axis/dim errors). Pin the `t_start`/`t_end` default-fill when good_axes && T present.

**`get_validation_errors`**: pin that it returns concatenated `_axis_errors + _dim_errors + _time_range_errors` (no flag-mutation, no raise).

### Mutator contracts

**`change_axes`**: pin the half-commented gate behavior. After `change_axes('XY')` on a `shape=(16,512,512)` file, `self.axes='XY'` (mutated) BUT `self.good_axes=False` (flagged). Same with `change_axes('TZYX')` on `shape=(16,512,512)` → `self.axes='TZYX'` (mutated to invalid length) BUT `self.good_axes=False`.

**`change_dim_res`**: invalid dim raises. `dim_res=None` (before `load_metadata`) raises.

**`change_selected_channel`**: requires `good_dims` AND `good_axes`. Requires `'C' in axes`. Requires valid index.

**`select_temporal_range`**: 7 validation paths (all raise).

### `save_ome_tiff` round-trip

- Source TIFF `axes='ZYX'`, channel=0, no time → output OME-TIFF `axes='TZYX'` with T as first dim, single timepoint.
- Source TIFF `axes='TZCYX'`, ch=0, t_start=0, t_end=0 → output `axes='TZYX'` with C dropped.
- Provenance JSON in description has the 6 documented keys.
- `ome.images[0].pixels.physical_size_{x,y,z}` and `time_increment` round-trip.

### `ImInfo` construction

- Re-construction with existing OME-TIFF → no `save_ome_tiff` call (skip the I/O).
- Re-construction with cached file lacking T-axis but file_info has T → triggers regen (Pass 2 #9).
- `pipeline_paths` dict has 18 keys, with the documented suffixes.
- `no_z`/`no_t` correctly inferred from `self.axes` + `self.shape`.

### `get_memmap` + `_normalize_memmap`

- Memmap pulled with file_axes != ImInfo's axes → transposed/squeezed correctly.
- `read_mode='r'` returns read-only.

### `allocate_memory`

- Empty allocation (data=None) writes empty OME-TIFF with shape=`self.shape`.
- Data allocation writes provided data.
- `dim_res` round-trips into pixel metadata.
- Description round-trips.

### `read_file` fallback

- TIFF that supports memmap → returns memmap.
- TIFF that fails memmap (e.g., compressed) → falls back to `imread`.
- TIFF that fails both → raises `ValueError`.
- ND2 → uses `nd2.imread`.

**Mocking complexity**: tests against fixtures (real files) will be cleaner than mocking `tifffile`/`nd2`/`ome_types`. The existing `tests/fixtures/_generate.py` script can extend. **Generate per-format fixtures: 1 OME-TIFF, 1 ImageJ-with-physicalsize, 1 ImageJ-without-physicalsize, 1 raw-TIFF with each unit, 1 ND2 (synthetic).** Skip ND2 generation if `nd2` cannot be written from Python (ND2 is read-only in the `nd2` library) — fall back to a checked-in tiny ND2 fixture.

## Pass 8 — Refactor Sequencing

The verifier doesn't fit the "test → cleanup → backend hoist" 3-slice template that prior stages used. The natural sequence here is **test → extract write/read I/O → unify normalization → split validation → split metadata extraction**. Some slices can land independently; others have hard dependencies.

**Pre-slice clarification: resolve the half-commented `change_axes` gate before any test-writing.** Was the gate disabled because callers were passing valid lengths and the gate fired unexpectedly? Or because the original raise was undesired UX? Git-archaeology + a 2-sentence decision answers this. The test-pinning slice freezes whichever behavior is correct; otherwise you bake the bug.

### Slice 1 — Tests (paired with this dechaos)

Per [[queue]] this is already queued. Scope:
- Generate per-format fixtures (or reuse synthetic OME-TIFF + add 4 more formats).
- Write characterization tests for each Pass 7 contract (~50–80 tests).
- Touch zero production code. Pin existing behavior, including the half-commented gate, the asymmetric `_validate` raise, the `_get_tif_tags_metadata`'s `self.axes` dependency.
- Recommended location: `tests/test_verifier.py` (mirrors other per-module tests).

**Cannot start Slice 2+ until Slice 1 lands.** Verifier touches every other test fixture; structural changes without tests will silently break the entire suite.

### Slice 2 — Clarify (low-risk, no behavior change)

Eligible immediately after Slice 1:
- Delete `__main__` block (line 1073–1130).
- Resolve half-commented `change_axes` gate (Pass 2 #5) — restore the guard OR delete the dead `good_axes = False`.
- Type hints: `metadata: Any` → annotate per-format (`Union[ome_types.OME, dict, list[dict]]` is unsatisfying but truthful; or `Any` with a comment).
- TypedDict `DimRes` for `dim_res` (Pass 5).
- Docstring updates for implicit call-order preconditions (Pass 4 #2).
- Reset `no_z`/`no_t` symmetrically in `_check_axes_exist` (Pass 2 #10).
- `_get_tif_tags_metadata`: `axes` as explicit parameter (Pass 2 #6).

### Slice 3 — Constructor I/O extraction (medium-risk)

After Slice 2:
- Move `os.makedirs` out of `FileInfo.__init__` — into a `prepare_output_dirs()` method called from `find_metadata`.
- Move `ImInfo` auto-regen + memmap creation into `ImInfo.load()` or `classmethod ImInfo.from_file_info(file_info)`. Keep a thin `__init__` that just stores `file_info`.
- Update 3 call sites (`run.py`, `nellie_fileselect.py` × 2, `conftest.py`).
- Tests need an update — characterization tests will need to call the new methods.

**This unlocks pure-logic testing for everything below.** Without this slice, every test still needs filesystem.

### Slice 4 — Unified axes normalizer (medium-risk)

After Slice 3:
- Extract `normalize_to_canonical(data, source_axes, target_pattern, squeeze_singleton_z=...) -> (data, axes)` to module level.
- Replace `_normalize_time_axis`, `_normalize_axes`, `_normalize_memmap` with calls to the unified normalizer.
- `_normalize_axes` and `_normalize_memmap` differ on Z-squeeze policy (always vs target-conditional) — parameterize.
- `_normalize_time_axis` is a degenerate case (no data, only axes string).

**Highest-value extraction.** Removes the largest cohort of subtly-different duplicates in the file.

### Slice 5 — Validation split (medium-risk)

After Slice 3, can land in parallel with Slice 4:
- Resolve asymmetric-raise contract (Pass 5) — recommend never raise from `_validate`.
- Split `_validate` into `compute_errors() -> ValidationReport` (pure) and `apply_defaults()` (mutator).
- `_get_output_path()` called explicitly by mutators after their state mutation.
- Consolidate `_check_*` and `_*_errors` (each pair currently does the same thing twice).

### Slice 6 — Metadata extractor strategy (high-risk)

After Slice 5:
- Define `MetadataExtractor` Protocol.
- 5 implementations: `OmeExtractor`, `ImageJExtractor`, `ImageJTifTagExtractor`, `Nd2Extractor`, `RawTiffTagExtractor`.
- `find_metadata` becomes a 2-line dispatch (extension → factory).
- `load_metadata` becomes a 1-line `dim_res = extractor.parse_dim_res(metadata)`.
- Extractors live in `nellie/im_info/extractors/` (new subpackage).

**This is the structural payoff.** But it depends on Slice 5 having sorted out the validation contract, because extractors will write to `dim_res` and validation runs after.

### Slice 7 — OME-TIFF writer extraction (low-risk)

After Slice 6:
- `_write_ome_tiff(path, data, axes, dim_res, description, provenance=None) -> None` shared by `save_ome_tiff` and `allocate_memory`.
- Saves ~30 lines, removes near-duplicate XML-injection logic.

### Slice 8 — `load_image(path) -> ImInfo` orchestrator (trivial)

Anytime after Slice 3:
- A free function in `nellie/im_info/__init__.py` that does the 4-step boot.
- Updates to `run.py`, `nellie_fileselect.py`, `conftest.py` reduce by ~3 lines each.

---

## Output suggested for the wiki

- Update [[im-info]] to point to this dechaos report (under "Refactor candidates" section).
- Update [[queue]] to link Slice 1 (test-pinning) and Slice 2 (clarify) here. Ladder of subsequent slices remains queue items, eligible after their dependencies.
- Possibly: add an [[im-info|im-info]] gotcha for the half-commented gate, the asymmetric `_validate` raise, and the `_normalize_axes` vs `_axis_errors` allowed-set divergence (allowed_axes excludes C in `_normalize_axes` but includes C in `_axis_errors`).

## Open questions

1. **`change_axes` gate**: restore the guard (raise on length mismatch) or delete the dead `good_axes = False` and trust `_validate`? Need git-archaeology on why it was commented.
2. **Asymmetric `_validate` raise**: keep raising on time errors only, or never raise from `_validate` (callers check `validation_errors`)?
3. **`_normalize_axes` excludes C; `_axis_errors` includes C in allowed set.** Which is correct? `save_ome_tiff` does collapse C into channel-take, so the file written through `_get_ome_metadata` → `_normalize_axes` should never have C. But `FileInfo` validation accepts C. Document the asymmetry or fix one side.
4. **Constructor I/O**: are `nellie_fileselect.py`'s call patterns OK with a `load()` method instead of constructor-side I/O? The widget threads `FileInfo` through closures (e.g. `lambda file_info: file_info.change_axes(text)`) — those would not change, but `select_file`'s `self.file_info = FileInfo(self.filepath, output_naming="detailed")` would need a follow-up `self.file_info.find_metadata()` call.
5. **Per-format extractor subpackage**: new `nellie/im_info/extractors/` directory or single `extractors.py` module? 5 small files vs 1 medium file. Codebase precedent: `nellie/segmentation/frangi_math.py` (single module) and `nellie/segmentation/frangi_filter.py` (single module) — flat-module convention so far.
