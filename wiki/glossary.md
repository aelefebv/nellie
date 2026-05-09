---
created: 2026-05-06
modified: 2026-05-09
---

# Glossary

Project-specific vocabulary, abbreviations, and domain terms. Each entry: term, one-line definition, optional `[[article]]` pointer for deeper context.

## Terms

- **adaptive_run** — Runtime device/memory mode arbitrator in `nellie/utils/adaptive_run.py`. Cascades through `(device, low_memory)` candidates on OOM / GPU-unavailable. See [[gpu-runtime]].
- **branch (skeleton)** — A connected segment of the [[networking|skeleton]] between junctions/tips. One of the five [[feature-extraction|hierarchy levels]].
- **component / organelle** — A connected instance label (one mitochondrion, one ER cluster). A [[feature-extraction|hierarchy level]].
- **dim_res** — Per-axis physical resolution dict `{X, Y, Z, T}` carried on [[im-info|`ImInfo`/`FileInfo`]]; X/Y/Z in micrometers, T in seconds.
- **flow_vector_array** — Sparse marker-match table written by [[hu-tracking|Hu-moment tracking]]. Positional schema: 6 cols 2D, 8 cols 3D. The shared artifact across [[tracking/index|tracking]].
- **Frangi vesselness** — Multi-scale Hessian-eigenvalue filter for tubular structures, the basis of [[filtering]].
- **good_axes / good_dims** — Boolean gates set by the [[im-info|`FileInfo` verifier]]. Both must be true before `save_ome_tiff` will write.
- **Hu moments** — Translation/scale/rotation invariant image moments used in [[hu-tracking]] for shape-similarity matching. Moment 7 (mirror) is intentionally omitted.
- **ImInfo** — Wrapper around a verified `FileInfo`. Owns the on-disk OME-TIFF, axes/`dim_res`, and `pipeline_paths`. The handle every [[pipeline|pipeline]] stage takes.
- **mocap markers** — Sparse anchor points produced by [[mocap-marking]] (LoG-of-distance peaks). Inputs to [[hu-tracking]].
- **node** — A skeleton junction-radius patch. Default-skipped [[feature-extraction|hierarchy level]].
- **OME-TIFF** — The canonical on-disk representation rewritten by `ImInfo` so all stages can assume `T[Z]YX` axes and embedded `dim_res`.
- **pipeline_paths** — String-keyed dict of canonical output file paths on `ImInfo`. The on-disk contract that lets [[pipeline|stages]] communicate without passing Python objects.
- **radius-adaptive pattern matching** — [[hu-tracking|Hu-moment tracking]]'s defining trick: ROI radius scales with the local distance-transform value. Wide windows for blobs, thin for tubules.
- **reassigned label** — Label propagated through time by [[voxel-reassignment]]. The `reassigned_label` column on the [[feature-extraction|components level]] carries this identity.
- **skel_relabelled** — `im_skel_relabelled` from [[networking]]: every voxel of an object gets a branch ID via per-object EDT.
- **stage Config** — Frozen `@dataclass` colocated above each algorithmic stage class (`FrangiConfig`, `LabelConfig`, `NetworkConfig`, `MarkersConfig`, `HuMomentTrackingConfig`, `VoxelReassignerConfig`, `HierarchyConfig`). Holds tuning knobs; `Stage(im_info, config, viewer, num_t)` is the canonical construction shape. `Stage.config` preserves original intent across cascade-mutated `Stage.device` / `Stage.low_memory` runtime state.
- **voxel** — The lowest [[feature-extraction|hierarchy level]]. One row per foreground voxel per frame.
