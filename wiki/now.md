---
created: 2026-05-06
modified: 2026-05-07
---

# Now — current state

Snapshot of what's active in the codebase right now. Refreshed by the NOW pass (or as part of CLOSE_OUT).

## Active work

- **Test scaffold rebuild on `slice-3-filter-cleanup`.** The legacy suite was removed in `a797323`; pytest scaffold + filtering coverage rebuilt in `d4c806a` / `4c53b0a`. Other modules — including [[tracking/hu-tracking|HuMomentTracking]] — currently have no tests pinning their behavior.

## Recently shipped

_Empty._

## Known issues / footguns

_Empty._

## Watch list

- **Tracking modules are unpinned.** Until the test rebuild reaches `nellie/tracking/`, claims like dense/sparse equivalence, `_log_hu` finiteness, and the hardcoded `1.0` cost cutoff have no automated guard. Treat changes there as un-regression-tested.
- **Verifier is unpinned too.** `im_info/verifier.py` lost its test file in the same rebuild. Per-format metadata parsing, axis normalization, and OME provenance round-trip are all currently un-regression-tested. See [[queue]] and [[im-info]].
