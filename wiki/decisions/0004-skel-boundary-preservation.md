---
created: 2026-05-11
modified: 2026-05-11
---

# Boundary voxels in `Network._remove_connected_label_pixels` are exempt from ambiguity cleanup

`Network._remove_connected_label_pixels` zeros skeleton voxels whose 3×3(×3) neighborhood touches more than one object label, so that branches can later be carved cleanly at junctions between objects. The function makes one exception: voxels on the volume boundary (any coordinate is 0 or `shape−1` along any axis) are unconditionally preserved — they keep their original label even when the neighborhood predicate would otherwise flag them ambiguous. Every rewrite of this function — including the sparse rewrite in PRD #168 Slice 2 (#170) — must keep this exemption byte-for-byte. Status: **Accepted** (load-bearing-by-test, root cause unknown).

## Considered Options

- **Preserve the exemption (chosen).** The new sparse implementation in PRD #168 Slice 2 (#170) achieves the same outcome by filtering boundary coordinates out of the skeleton-coord array up front, so the neighbor scan never visits boundary voxels at all. Same behavior as the old `~boundary` mask AND, simpler structure, no separate boundary array.
- **Drop the exemption.** Treat boundary voxels the same as interior voxels — flag them ambiguous when their neighborhood touches multiple labels, zero them out. Rejected for this PR. The behavior originated before recorded git history and is pinned by exactly one test (`tests/test_networking.py:348-391`, `test_remove_connected_label_pixels_preserves_boundary_voxels_2d`); both the source comment at `nellie/segmentation/networking.py:224` ("Preserve original behavior") and `wiki/segmentation/networking.md:39` assert the behavior without justifying it. Removing it without first investigating downstream consumers (`_get_pixel_class`, `_get_branch_skel_labels`, `_relabel_objects`) is a behavior change of unknown blast radius.

## Consequences

- The Slice 1 synthetic-test suite includes `test_remove_connected_label_pixels_junction_on_boundary_3d` and `test_remove_connected_label_pixels_all_boundary_3d`, both of which would fail if a future rewrite dropped the exemption. The 2D test (`test_remove_connected_label_pixels_preserves_boundary_voxels_2d`) is the original pin and remains untouched.
- Future maintainers who want to remove the exemption should first investigate the downstream consumers above to determine whether removal would break the labeling contract — particularly for objects whose skeleton clings to the volume edge, where post-cleanup zeros would propagate into branch labels and per-object EDT seeds. If the investigation shows the exemption is safely removable, that is a separate behavior-change PR with its own PRD.
- The new sparse implementation's `np.where(labels > 0)` followed by an interior-coord filter is structurally clearer about the exemption than the old AND-with-boundary-mask: the rewrite makes the exemption a precondition of the scan rather than a post-hoc subtraction.

## References

- PRD #168 — sparse rewrite of `Network._remove_connected_label_pixels`
- Slice 1 #169 — pin current behavior with snapshot + synthetic tests + this ADR (no production code changes)
- Slice 2 #170 — sparse implementation that preserves the exemption by construction
