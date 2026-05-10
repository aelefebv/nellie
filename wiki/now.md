---
created: 2026-05-06
modified: 2026-05-09
---

# Now — current state

Snapshot of what's active in the codebase right now. Refreshed by the NOW pass (or as part of CLOSE_OUT).

## Active work

- (none right now — MPS feature just landed; see Recently shipped)

## Recently shipped

- **🎉 Apple Silicon / MPS GPU acceleration end-to-end** (2026-05-09) — PRD #140 closed via slices #141–#145 (PRs #146–#150). PyTorch+MPS is now a third backend alongside numpy and cupy; install via `pip install 'nellie[mps]'`. Filtering, labelling, networking, hu_tracking are onboarded; `device="gpu"` is platform-aware (MPS on Darwin, CUDA elsewhere); `device="mps"` is also explicit. New shim modules `torch_xp` + `torch_ndi`, new `adaptive_run.device_cascade` + `to_numpy` helpers, new `mps` pytest marker (opt-in, deselected by default). 12 latent shim bugs fixed during slice 1 (gradient edge_order, scipy reflect semantics, gaussian_laplace algorithm). Three ADRs in [[decisions/index|decisions/]]. See [[gpu-runtime]] for the rewritten architecture article and the new gotcha list. **Watch list:** chunked low-memory MPS path is not directly exercised by the smoke trio; small-fixture MPS is slower than CPU on labelling/hu_tracking (dispatch overhead vs small per-frame work — production volumes expected to flip this).

- **Closed-form 3×3 eigenvalue helpers** in [[gpu-runtime|chunking.py]] (`eigvalsh_3x3_symmetric`, `eigvalsh_3x3_components`). Drop-in replacements for `safe_eigvalsh` on the symmetric-3×3 case; ~10× faster than LAPACK on the isolated math. `safe_eigvalsh` retained as the gold-standard reference (used by tests and any future N×N caller).
- **Benchmark scaffold.** New `benchmark` pytest marker registered in `pyproject.toml` (deselected by default via `addopts`). `tests/test_filtering_perf.py` carries 7 microbenchmarks covering end-to-end Filter wall-clock, dense vs sparse vesselness paths, dispatcher overhead, and closed-form vs LAPACK eigvalsh. First adopter of an opt-in perf-test pattern in this repo.

## Watch list

- **Tracking modules are unpinned.** Until the test rebuild reaches `nellie/tracking/`, claims like dense/sparse equivalence, `_log_hu` finiteness, and the hardcoded `1.0` cost cutoff have no automated guard. Treat changes there as un-regression-tested.
- **GPU paths in the perf pass are untested.** All filtering tests (correctness + benchmarks) run `device="cpu"`. The new in-place ops, dense-path flat slicing, and closed-form eigvalsh all use CuPy-supported idioms in principle, but no CI run has exercised them on CUDA. The dispatcher's `bool(h_mask.all())` would force a D→H sync on GPU — pin this with a real GPU run before assuming the savings carry over.
