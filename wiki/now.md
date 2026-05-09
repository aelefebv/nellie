---
created: 2026-05-06
modified: 2026-05-09
---

# Now — current state

Snapshot of what's active in the codebase right now. Refreshed by the NOW pass (or as part of CLOSE_OUT).

## Active work

- **Filtering perf pass on `dechao`.** Closed-form 3×3 symmetric eigenvalues replaced LAPACK `eigvalsh` in the 3D path; dense `h_mask` fast path added; in-place reductions; cached cupy backend probe; `_get_frob_mask` no longer copies the volume to mask infs. End-to-end 3D `Filter.run()` ~2.4× faster on the yeast fixture (1223 ms → 507 ms). Opt-in benchmark suite (`tests/test_filtering_perf.py`, `pytest -m benchmark`) pins the wins. See [[segmentation/filtering#performance|filtering#Performance]].

## Recently shipped

- **Closed-form 3×3 eigenvalue helpers** in [[gpu-runtime|chunking.py]] (`eigvalsh_3x3_symmetric`, `eigvalsh_3x3_components`). Drop-in replacements for `safe_eigvalsh` on the symmetric-3×3 case; ~10× faster than LAPACK on the isolated math. `safe_eigvalsh` retained as the gold-standard reference (used by tests and any future N×N caller).
- **Benchmark scaffold.** New `benchmark` pytest marker registered in `pyproject.toml` (deselected by default via `addopts`). `tests/test_filtering_perf.py` carries 7 microbenchmarks covering end-to-end Filter wall-clock, dense vs sparse vesselness paths, dispatcher overhead, and closed-form vs LAPACK eigvalsh. First adopter of an opt-in perf-test pattern in this repo.

## Watch list

- **Tracking modules are unpinned.** Until the test rebuild reaches `nellie/tracking/`, claims like dense/sparse equivalence, `_log_hu` finiteness, and the hardcoded `1.0` cost cutoff have no automated guard. Treat changes there as un-regression-tested.
- **GPU paths in the perf pass are untested.** All filtering tests (correctness + benchmarks) run `device="cpu"`. The new in-place ops, dense-path flat slicing, and closed-form eigvalsh all use CuPy-supported idioms in principle, but no CI run has exercised them on CUDA. The dispatcher's `bool(h_mask.all())` would force a D→H sync on GPU — pin this with a real GPU run before assuming the savings carry over.
