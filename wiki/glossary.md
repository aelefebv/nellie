---
created: 2026-05-06
modified: 2026-05-12
---

# Glossary

Project-specific vocabulary, abbreviations, and domain terms. Each entry: term, one-line definition, optional `[[article]]` pointer for deeper context.

## Terms

- **adaptive_run** — Runtime device/memory mode arbitrator in `nellie/utils/adaptive_run.py`. Cascades through `(device, low_memory)` candidates on OOM / GPU-unavailable. See [[gpu-runtime]].
- **branch (skeleton)** — A connected segment of the [[networking|skeleton]] between junctions/tips. One of the five [[feature-extraction|hierarchy levels]].
- **component / organelle** — A connected instance label (one mitochondrion, one ER cluster). A [[feature-extraction|hierarchy level]].
- **dim_res** — Per-axis physical resolution dict `{X, Y, Z, T}` carried on [[im-info|`ImInfo`/`FileInfo`]]; X/Y/Z in micrometers, T in seconds.
- **DROPPABLE_KEYS** — `frozenset[str]` in `nellie.im_info` defining the universe of pipeline outputs that `ImInfo.remove_marked_intermediates` is allowed to delete. The 12 image-like intermediates + `adjacency_maps` + the special `im_path` (canonical OME-TIFF). The 5 `features_*` CSVs are intentionally excluded — they are the user-facing analytical product. Drives the per-output retention UI in [[settings|napari Settings]] and the `cleanup_drop_keys` kwarg on `nellie.run.run()`. See [[decisions/0014-intermediates-policy-frozenset]].
- **flow_vector_array** — Sparse marker-match table written by [[hu-tracking|Hu-moment tracking]]. Positional schema: 6 cols 2D, 8 cols 3D. The shared artifact across [[tracking/index|tracking]].
- **Frangi vesselness** — Multi-scale Hessian-eigenvalue filter for tubular structures, the basis of [[filtering]].
- **good_axes / good_dims** — Boolean gates set by the [[im-info|`FileInfo` verifier]]. Both must be true before `save_ome_tiff` will write.
- **Hu moments** — Translation/scale/rotation invariant image moments used in [[hu-tracking]] for shape-similarity matching. Moment 7 (mirror) is intentionally omitted.
- **ImInfo** — Wrapper around a verified `FileInfo`. Owns the on-disk OME-TIFF, axes/`dim_res`, and `pipeline_paths`. The handle every [[pipeline|pipeline]] stage takes. Constructor is thin (no I/O); use `ImInfo.from_file_info(file_info)` for the canonical construct-and-load entry point.
- **load_image(path)** — Canonical one-call programmatic entry point in `nellie.im_info`. Equivalent to `FileInfo(path)` → `find_metadata` → `load_metadata` → `ImInfo.from_file_info(file_info)`. Used by `run.py`, tests, demos. The napari fileselect widget intentionally splits these steps to support user override of axes/dim_res via UI.
- **MetadataExtractor** — Protocol defined in `nellie/im_info/extractors/protocol.py`; 5 implementations (OME, ImageJ, ImageJ-tif-tags, ND2, raw-TIFF) with a `parse_dim_res() -> DimRes` method. The seam that makes adding a new format (CZI, LIF, bioio) a single-file addition. See [[im-info]].
- **mocap markers** — Sparse anchor points produced by [[mocap-marking]] (LoG-of-distance peaks). Inputs to [[hu-tracking]].
- **node** — A skeleton junction-radius patch. Default-skipped [[feature-extraction|hierarchy level]].
- **OME-TIFF** — The canonical on-disk representation rewritten by `ImInfo` so all stages can assume `T[Z]YX` axes and embedded `dim_res`.
- **pipeline_paths** — String-keyed dict of canonical output file paths on `ImInfo`. The on-disk contract that lets [[pipeline|stages]] communicate without passing Python objects.
- **radius-adaptive pattern matching** — [[hu-tracking|Hu-moment tracking]]'s defining trick: ROI radius scales with the local distance-transform value. Wide windows for blobs, thin for tubules.
- **reassigned label** — Label propagated through time by [[voxel-reassignment]]. The `reassigned_label` column on the [[feature-extraction|components level]] carries this identity.
- **skel_relabelled** — `im_skel_relabelled` from [[networking]]: every voxel of an object gets a branch ID via per-object EDT.
- **stage Config** — Frozen `@dataclass` colocated above each algorithmic stage class (`FrangiConfig`, `LabelConfig`, `NetworkConfig`, `MarkersConfig`, `HuMomentTrackingConfig`, `VoxelReassignerConfig`, `HierarchyConfig`). Holds tuning knobs; `Stage(im_info, config, viewer, num_t)` is the canonical construction shape (`Hierarchy` is asymmetric — no `num_t` constructor param per PRD #112 resolved decision #3). `Stage.config` preserves original intent across cascade-mutated `Stage.device` / `Stage.low_memory` runtime state. Each Config validates inputs at construction time via `__post_init__` (PR #136): bad `device` strings (reuses `adaptive_run.normalize_device` for the canonical `{"auto","cpu","gpu","cuda"}` allowed-set), bad allowed-set strings (`MarkersConfig.use_im`, `HuMomentTrackingConfig.mode`), non-positive numerics, inverted radius ranges. The [[settings|napari settings widget]] returns Configs directly from `get_*_params()` since PR #138 — no dict→`Config(**kwargs)` wrap at the processor seam. Modify the relevant test_*.py file when adding new fields.
- **voxel** — The lowest [[feature-extraction|hierarchy level]]. One row per foreground voxel per frame.
