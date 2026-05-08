---
created: 2026-05-08
modified: 2026-05-08
---

# Dechaos scan — `nellie/segmentation/mocap_marking.py` (`Markers`)

One-shot review of the next pipeline stage to receive the test+hoist treatment. Findings are intended to feed an upcoming three-slice PRD that mirrors PR #77 (Network). Durable items should be folded into [[mocap-marking]] gotchas after the PRD is open.

**Reference templates**: Network (`nellie/segmentation/networking.py`, post-#83) is the closest mirror — same backend pattern, same chunked low-memory variants, same outer `adaptive_run.mode_candidates` cascade. Use it as the after-picture.

---

## Resolved decisions (2026-05-08)

1. **Pre-slice `prefer_gpu` vs `device`** → **Option A**: drop `prefer_gpu` from the `Markers` constructor and from `nellie_napari/nellie_settings.py` (`mocap_prefer_gpu` widget at line 275). Users who relied on `prefer_gpu=False` should pass `device="cpu"` instead. Matches Filter/Label/Network constructor shape.
2. **Slice 3 inner `_run_frame` OOM cascade** → **Option A**: delete the inner cascade entirely. Outer `adaptive_run.mode_candidates` in `run()` handles OOM by retrying the whole stage with the next `(dev, low)` candidate. Loses partial-progress recovery on long time series; eliminates the cross-frame `self.low_memory` / `self.max_chunk_voxels` state-leak that the wiki currently flags as a footgun.
3. **Wiki fold timing** → **Defer to during/after the slices**, not now. Findings (a) cross-frame state mutation and (c) dual `prefer_gpu`/`device` API will be obsolete after Slice 2/3 lands; only (b) silent `or 1.0` X-resolution fallback is worth folding, and the cleanest moment is when Slice 2 decides whether to drop the silent fallback.

---

## Pass 1 — System Map

- **Stage 4 of 7** in `nellie/run.py` (line 89). Two in-tree call sites: `run.py:89` (`Markers(im_info, device=device)` — no `low_memory`) and `nellie_napari/nellie_processor.py:412` (`**step_kwargs` from the settings widget).
- **Class**: `Markers`, single class in `nellie/segmentation/mocap_marking.py`, 836 lines.
- **Inputs (memmaps via `ImInfo.pipeline_paths`)**: `im_instance_label` (from Label), `im` (raw, from `im_info.im_path`), optionally `im_preprocessed` (from Filter, when `use_im='frangi'`).
- **Outputs (memmaps)**: `im_marker` (uint8 binary), `im_distance` (float32), `im_border` (uint8 binary). All three are always written, even when `use_im='frangi'` — downstream Hierarchy + HuMomentTracking depend on `im_distance` and `im_border`.
- **External deps**: `nellie.utils.adaptive_run` (canonical backend helpers), `scipy.ndimage`, optionally `cupyx.scipy.ndimage` via `cupy`, `numpy`, `nellie.im_info.verifier.ImInfo`.
- **Tests**: none. `tests/test_mocap_marking.py` does not exist.
- **Architectural shape**: per-frame outer loop (`_run_mocap_marking`) → per-frame work (`_run_frame` wrapping `_run_frame_impl`) → frame writes + flushes inline. Outer `run()` uses `adaptive_run.mode_candidates` for dev/low retries. **Inner `_run_frame` runs its own OOM cascade on top of that** (see Pass 3 / Pass 4).

## Pass 2 — Boundary Scan

| # | Mixed concern | Where | Suggested seam |
|---|---|---|---|
| 1 | Local backend resolution duplicates `adaptive_run` | `_resolve_backend` (180–194), `_try_import_cupy` (196–217), `_is_oom_error` (240–250), `_free_gpu_memory` (252–258), `_switch_to_cpu` (260–265) | Delete; call `adaptive_run.resolve_backend` / `try_import_cupy` / `is_oom_error` / `free_gpu_memory(self.xp)` / `self._set_backend("cpu")`. Same change Network's Slice 3 made (see [[networking]]). |
| 2 | Local chunking helpers shadow `nellie.utils.chunking` | `_compute_chunk_shape` (267–274), `_iter_chunks` (276–289) | `compute_chunk_shape` is byte-identical — delete and import. **`_iter_chunks` here yields a 5-tuple** (adds `starts`, `ext_starts`); the shared helper yields a 3-tuple. Either extend the shared helper with `include_starts: bool`, or keep the 5-tuple local with a comment pointing at the divergence. |
| 3 | Dev driver in module body | `__main__` block (831–836) hard-codes a Windows path | Delete (Network did the same in #82). |
| 4 | Halo math is stage-specific | `_log_halo` (291–300), `_nms_halo` (302–304) | Keep local — depends on `self.sigmas`, `self.peak_min_distance`, `self.z_ratio`, `self.im_info.no_z`. Not reusable. |
| 5 | I/O mixed with dispatch in the per-frame loop | `_run_mocap_marking` (752–781): writes + flushes inline | Mirrors Network/Label pattern. Don't split — keep boundary at the stage level. |

## Pass 3 — Responsibility Scan

The class wears four hats — backend management, chunking, marker math (LoG / NMS / distance / border), pipeline orchestration. Same as Filter/Label/Network. Acceptable inheritance pattern. The cleanups below are **dead code, not split candidates**.

| Item | Location | Status | Action |
|---|---|---|---|
| `_get_t` | 364–374 | Dead. `__init__` (116–120) already resolves `self.num_t`. | Delete. (Network deleted its analogue.) |
| `self.sigmas = []` | 137 | Initialized then overwritten in `_set_default_sigmas` (351). | Drop the init line. |
| `self.shape = ()` | 139 | Overwritten in `_allocate_memory` (388). | Drop the init line. |
| `self.debug = None` | 148 | Never read, never set elsewhere. | Drop. (Network deleted same.) |
| `self.use_gpu` | 157, 235, 264 | Derivable from `self.device_type == "cuda"`; kept in sync at three sites. | Drop attribute; replace 1 callsite (none in code today besides the constructor noise). |
| `_run_frame` inner OOM cascade | 705–750 | Architectural overlap with outer `mode_candidates` cascade. **Mutates `self.low_memory`, `self.max_chunk_voxels`, and (via `_switch_to_cpu`) the backend in place — these mutations persist across all later frames** in the same `run()` call. | See Pass 8 / Slice 3 — decide whether to delete or document. |

The wiki gotchas already note the cross-frame mutation (`_run_frame has its own OOM cascade that mutates self and persists across frames`). The dechaos finding is the same; the decision is what to do about it.

## Pass 4 — Dependency Scan

| # | Type | Issue | Impact |
|---|---|---|---|
| 1 | **Hidden config / dual-API** | Constructor takes both `prefer_gpu: bool` (default `True`) AND `device: str` (default `"auto"`). They interact at lines 153–155: only when `device=="auto"` AND `prefer_gpu=False` does `device` flip to `"cpu"`. Any other combination silently ignores `prefer_gpu`. The napari widget exposes BOTH (`mocap_prefer_gpu` checkbox at `nellie_napari/nellie_settings.py:275` AND `mocap_device` combo at line 278). | Already on the queue ("`Markers` has overlapping `prefer_gpu: bool` + `device: str` args; resolve which device-flag wins before the batched Config PRD"). **This needs to be resolved before Slice 2** — otherwise Slice 2 cleans a constructor that gets re-cleaned in the Config slice. See Pass 8 pre-slice. |
| 2 | Silent fallback on missing X resolution | `x_res = self.im_info.dim_res.get('X') or 1.0` (122). Network just indexes `self.im_info.dim_res['X']` (would `KeyError`). Same on Z (123). | Markers will silently produce markers at unrealistic scales if `dim_res['X']` is missing/None. Pick one defensive posture across stages. |
| 3 | Silent input flooring | `min_radius_um = max(min_radius_um, float(x_res))` (129) coerces user input to ≥ one pixel. Documented in docstring; still surprising. | Keep — but pin in test (assert effective `min_radius_um` for a sub-pixel input). |
| 4 | Inconsistent backend attribute names across stages | `self._xp` / `self._ndi` private + `xp` / `ndi_backend` properties here; Network exposes `self.xp` / `self.ndi` directly. | Cosmetic but affects future cross-stage refactors and Config rollout. Rename to match Network. |
| 5 | `device` accepts undocumented `"cuda"` alias | Constructor docstring says `{"auto","cpu","gpu"}` (107); `_resolve_backend` (185) silently accepts `"cuda"` too. | `adaptive_run.normalize_device` is the canonical source of truth — `_set_backend` already routes through it (231). Backend hoist makes the constructor go through it too, fixing the gap. |

## Pass 5 — Contract Scan

Pinnable invariants for Slice 1 tests (no test pins any of these today):

**Output dtypes / ranges**
- `im_marker`: uint8, values ∈ {0, 1}.
- `im_distance`: float32, values ≥ 0, **≤ 2 × `max_radius_px`** (clamp at line 448 — wiki documents this).
- `im_border`: uint8, values ∈ {0, 1}.

**Spatial relationships**
- `border ∩ mask == ∅` (border is `dilation(mask) XOR mask`; line 440).
- `marker ⊆ mask` — peaks must be inside segmented objects (`valid_mask = mask & (distance_im > 0)`, line 482; same in chunked path 528). **Implicit; not asserted.**
- Empty mask → all-zero `marker`, `distance`, `border` (no error). Wiki notes this as a footgun.

**Shape branch**
- `_run_mocap_marking` (764–775) writes `[:]` if `im_marker_memmap.shape != self.shape and im_info.no_t`, else `[t]`. Same shape-mismatch branch as Network. Pin both code paths in tests.

**Backend / chunking equivalence**
- `low_memory=True` chunked path must produce identical output to unchunked path (the wiki notes `test_mocap_marking_low_memory_matches_full_2d` — but that test does not exist yet; it's an aspirational reference). Pin in Slice 1.

**Per-frame returns**
- `_run_frame_impl` returns `(marker, distance_im, border_mask)` as **CPU numpy arrays** with explicit dtypes (`uint8`, `float32`, `uint8`). Empty-mask path uses `np.zeros_like(self.im_memmap[t], dtype=...)` — depends on `im_memmap[t]` having compatible shape. Edge case if a frame has unexpected ndim.

**Loose contract bits**
- `_local_max_peak_chunked` empty return: `xp_mod.zeros((0, ndim), dtype=int)` — `dtype=int` is platform-dependent. Use `intp` or `int64` to be explicit.

## Pass 6 — Composability Scan

Mostly **already factored** — `nellie.utils.adaptive_run` and `nellie.utils.chunking` are the canonical primitives. Findings:

- `_distance_im` (419–450) is a clean reusable mask → (distance, border) primitive. No other stage needs it today; leave in place.
- `_local_max_peak` (multi-scale LoG with running best response) is interesting but stage-specific; no extraction warranted.
- The outer `run()` cascade pattern (resolve device order → `mode_candidates` → try / `is_gpu_unavailable_error` / `is_oom_error` / log / continue) is now repeated nearly verbatim across Filter/Label/Network/Markers — about 25 lines of structural sameness per stage. **Composable extraction candidate**, but keep deferred until all 4 stages reach the same shape; collapsing the pattern is a cross-stage Pass-8-style refactor better done after the 4 untested stages all use it.
- The inner `_run_frame` OOM cascade is **redundant with the outer cascade** — see Pass 3 / Pass 8.

## Pass 7 — Testability Scan

**Current state**: zero tests. Slice 1 fills this gap before Slices 2–3 touch structure.

**Fixture cascade** (already in `tests/conftest.py`): Frangi (session) → Label (session) → per-test factory. **Add**: `make_markers_imageinfo_2d/3d` (+ `_module` variants) that copies the existing Frangi + Label memmaps into a fresh ImInfo. Markers needs raw `im_memmap` (already populated by `_build_iminfo`'s source-image copy) + `im_instance_label` (Label cascade) + optionally `im_preprocessed` (Frangi cascade — only when `use_im='frangi'`). Lines: ~40 added to conftest, mirroring `_make_network_imageinfo_factory` (267–300).

**Test plan for Slice 1** (target ~17 tests, mirroring `test_networking.py`'s shape):

```text
Smoke / shape:
- 2D + 3D: run end-to-end on the yeast fixtures with device='cpu', assert outputs exist with expected shapes.
- num_t default vs explicit override.

Output contract:
- im_marker dtype == uint8, values ⊆ {0,1}.
- im_distance dtype == float32, values ≥ 0, max ≤ 2 * max_radius_px.
- im_border dtype == uint8, values ⊆ {0,1}.
- border ∩ mask == ∅ (test_border_is_outside_mask).
- marker ⊆ mask (test_markers_inside_objects).

Behavior:
- Empty mask → all-zero outputs, no error.
- use_im='distance' produces markers; use_im='frangi' produces markers (different but both >0 in non-empty objects).
- use_im='frangi' raises if Frangi memmap is absent (Markers should not silently no-op).
- Low-memory chunked == unchunked equivalence (2D, distance-mode) — pins _log_halo + _nms_halo correctness.
- viewer.status writes: assert no-op when viewer=None; assert called per-frame when viewer is a stub.

Architecture characterization (these protect the *current* behavior before Slice 3 changes it):
- Inner _run_frame OOM cascade leaves self.low_memory / self.max_chunk_voxels mutated after recovery.
- Single shape-branch in _run_mocap_marking (`shape != self.shape and no_t`) — pin both write paths.
```

**Hard-to-test**: the inner OOM cascade requires fault injection (e.g., monkeypatch `_distance_im` to raise `MemoryError` on first call). Keep the characterization minimal: assert that *after* an injected OOM, `self.low_memory == True` (or `self.device_type == "cpu"`), so Slice 3 can intentionally change the contract and the test fails loudly.

## Pass 8 — Refactor Sequencing

Mirror PR #77 (Network) — three slices, one per PR. **One pre-slice decision** to make first.

### Pre-slice (decision, not code) — resolve `prefer_gpu` vs `device`

The constructor takes both. The napari widget exposes both. Pick one of:

**Option A — drop `prefer_gpu`**: Lift to `device="cpu"` semantics for users who today set `prefer_gpu=False`. Clean signature; one-line napari widget removal; matches Network/Label.

**Option B — drop `device="auto"` semantics**: Make `device` strictly `{"cpu","gpu"}` and let `prefer_gpu` decide auto. Diverges from Filter/Label/Network — bad.

**Option C — keep both, document interaction**: Lowest-risk, but punts the dual-API problem to the cross-stage Config dataclass slice and forces it to handle the overlap.

**Recommendation**: Option A. It's the only option that matches the Network/Label/Filter constructor shape, and it lets Slice 2 deliver a constructor the cross-stage Config slice can adopt as-is.

### Slice 1 — Characterization tests (mirror PR #81)

**Clarify** + **Protect**.

- Add `tests/test_mocap_marking.py` (~17 tests, ~400 lines).
- Extend `tests/conftest.py` with `make_markers_imageinfo_2d/3d` factories cascading off existing Frangi + Label session paths.
- Pin all contracts from Pass 5 + the inner-OOM characterization from Pass 7.
- No production-code changes.

**Risk**: low. Pure additive.

### Slice 2 — Structural cleanups (mirror PR #82)

**Clarify** + **Stabilize**.

- Apply pre-slice decision (drop `prefer_gpu` per Option A; one matching change in `nellie_napari/nellie_settings.py`).
- Delete `_get_t` (dead).
- Delete `self.sigmas = []`, `self.shape = ()`, `self.debug = None` initial state.
- Delete `__main__` block.
- Drop `self.use_gpu`.
- Rename `ndi_backend` → `ndi`, `_xp` → `xp` (internal naming match with Network).
- Decide on `or 1.0` X-resolution fallback (line 122) — recommend dropping the silent fallback (let `KeyError` surface) to match Network.
- Wiki touch-ups in [[mocap-marking]]: fold pre-slice decision into gotchas, drop reference to `prefer_gpu`.

**Risk**: medium. Constructor signature change is user-visible (napari widget + any external caller). Slice 1 tests guard the behavior contract; the widget change is mechanical.

### Slice 3 — Backend hoist (mirror PR #83)

**Stabilize** + **Separate**.

- Delete `_resolve_backend`, `_try_import_cupy`, `_is_oom_error`, `_free_gpu_memory`, `_switch_to_cpu`.
- Constructor + `_set_backend` call `adaptive_run.resolve_backend` directly.
- Inner `_run_frame` cascade (705–750) — **architectural decision**:

  **Option A (recommended): delete the inner cascade entirely.** Let the outer `mode_candidates` cascade in `run()` handle OOM by retrying the whole stage with the next `(dev, low)` candidate. Same as Network does today.

  **Tradeoff**: outer retry restarts from frame 0 (re-writing the early frames). Network accepted this. The inner cascade today is faster on partial-progress recovery but introduces the cross-frame `self.low_memory` / `self.max_chunk_voxels` mutation that the wiki already flags as a footgun. Deleting it removes that footgun.

  **Option B: keep the inner cascade but route through canonical helpers.** Replace `self._is_oom_error(exc)` → `adaptive_run.is_oom_error(exc)`, `self._free_gpu_memory()` → `adaptive_run.free_gpu_memory(self.xp)`, `self._switch_to_cpu()` → `self._set_backend("cpu")`. Document the cross-frame mutation explicitly in the wiki.

- Either option: delete the local helpers; use canonical `adaptive_run` helpers throughout.
- Slice 1's inner-OOM characterization test gets updated to match the chosen contract (deleted entirely under Option A; loosened under Option B).

**Risk**: medium-high under Option A (real behavior change on OOM), medium under Option B. Slice 1 tests guard the non-OOM contract; the OOM contract is intentionally being changed.

### Deferred (per queue policy)

- `MarkersConfig` dataclass extraction. Wait until all 4 untested stages (Markers, HuMomentTracking, VoxelReassigner, Hierarchy) have completed test+hoist so the cross-stage Config slice can land as one consistent pass. This is the queue's stated policy.

---

## Open questions for the user

1. **Pre-slice decision** — Option A (drop `prefer_gpu`)? Or do you want to keep the dual-API and punt to the Config slice?
2. **Slice 3 inner cascade** — Option A (delete) or Option B (keep, route through canonical)? Network chose Option A by accident (it didn't have the cross-frame state-mutation that Markers has) — Markers' inner cascade is more intentional, so the call here is genuinely different.
3. **Wiki fold** — once the PRD is shaped, want me to invoke `/repo-wiki` to fold the durable findings (cross-frame mutation, `or 1.0` silent fallback, dual `prefer_gpu`/`device` API) into [[mocap-marking]] gotchas?
