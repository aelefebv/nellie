---
created: 2026-05-06
modified: 2026-05-07
---

# Segmentation

Converts raw 2D/3D/4D microscopy intensity volumes into instance-labelled organelle masks, motion-anchor markers, and a topological skeleton of the network. Built for tubular/networked structures (mitochondria, ER) but general enough for blob organelles.

## Stage order

The execution order in `nellie/run.py` (see [[pipeline]]) is:

1. [[filtering|`filtering.Filter`]] — raw → Frangi vesselness response (`im_preprocessed`)
2. [[labelling|`labelling.Label`]] — Frangi → instance labels (`im_instance_label`)
3. [[networking|`networking.Network`]] — labels + Frangi → skeleton + pixel-class + branch-relabelled volume
4. [[mocap-marking|`mocap_marking.Markers`]] — labels + intensity → marker points + distance map + border mask

Note that `Markers` runs **after** `Network` despite being conceptually "between" labelling and tracking. All inter-stage outputs are on-disk memmaps via `ImInfo.pipeline_paths`.

## Interactions

- Inputs from [[im-info|ImInfo]] (shape, axes, `dim_res`, output paths).
- Backend dispatch and OOM cascade via [[gpu-runtime|adaptive_run]].
- Outputs consumed by [[tracking/index|tracking]] (markers, distance, instance labels, branch labels) and [[feature-extraction]] (instance labels, branch labels, skeleton, pixel-class, distance, border).
- `__init__.py` re-exports the four stage classes; nothing else is in the package surface.

## Cross-stage gotchas

- **Outputs are intermediate disk artifacts, not return values.** Each stage allocates a memmap and `flush()`es per frame; nothing useful comes back from `.run()`.
- **The adaptive cascade is identical across stages.** All four share the `device_order × low_memory × OOM` retry plan; on GPU OOM they fall back to chunked-CPU mid-frame. Tests drive `device="cpu"` explicitly to be deterministic.
- **`min_radius_um` is floored to `dim_res["X"]`** in every stage — you can't ask for sub-pixel structures.
- **2D vs 3D branches everywhere** (keyed off `im_info.no_z`). Anisotropic Z is handled by scaling sigma with `z_ratio = z_res / x_res`.
- **T-axis edge case.** When `no_t` or `num_t == 1`, output memmap is 3D not 4D — code special-cases both shapes when writing.

## Invariants

- Each stage assumes its predecessor's outputs exist on disk at the documented `pipeline_paths` keys.
- **Label IDs are per-frame, not stable across frames.** Cross-frame stability is the job of [[voxel-reassignment]].
- Stages must not mutate input memmaps — pinned for [[filtering]] by `test_input_memmap_unchanged` (and `test_2d_output_invariants`); the other stages share the invariant but are not yet covered by tests.
