"""
Example usage and adjacency remapping using voxel matches.

This script avoids allocating large dense voxel-by-voxel matrices and optionally
uses GPU (CuPy) for accumulation when available.
"""
import pickle

import numpy as np

from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.tracking.voxel_reassignment import VoxelReassigner


def accumulate_pair_counts(src_ids, dst_ids, n_src, n_dst, use_gpu=True):
    """
    Accumulate counts into an (n_src, n_dst) matrix given parallel vectors
    src_ids and dst_ids, optionally using cupy if available.
    """
    src_ids = np.asarray(src_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_ids, dtype=np.int64)

    if src_ids.size == 0 or dst_ids.size == 0:
        return np.zeros((n_src, n_dst), dtype=np.uint32)

    if use_gpu:
        try:
            import cupy as cp

            src_gpu = cp.asarray(src_ids)
            dst_gpu = cp.asarray(dst_ids)
            counts_gpu = cp.zeros((n_src, n_dst), dtype=cp.uint32)
            cp.add.at(counts_gpu, (src_gpu, dst_gpu), 1)
            counts = cp.asnumpy(counts_gpu)

            del src_gpu, dst_gpu, counts_gpu
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            return counts
        except Exception:
            pass

    counts = np.zeros((n_src, n_dst), dtype=np.uint32)
    np.add.at(counts, (src_ids, dst_ids), 1)
    return counts


def main():
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    file_info = FileInfo(im_path)
    file_info.find_metadata()
    file_info.load_metadata()
    im_info = ImInfo(file_info)

    run_obj = VoxelReassigner(im_info, num_t=3)
    run_obj.run()

    edges_loaded = pickle.load(open(im_info.pipeline_paths["adjacency_maps"], "rb"))

    # ------------------------------------------------------------------
    # Example: branch adjacency between t0 and t1 without dense v_t
    # ------------------------------------------------------------------
    mask_01 = run_obj.obj_label_memmap[:2] > 0
    mask_voxels_0 = np.argwhere(mask_01[0])
    mask_voxels_1 = np.argwhere(mask_01[1])

    t0_coords_in_mask_0 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_0)}
    t1_coords_in_mask_1 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_1)}

    matches_t0_t1_prev, matches_t0_t1_next = run_obj.running_matches[0]

    idx_matches_0 = []
    idx_matches_1 = []
    for coord_prev, coord_next in zip(matches_t0_t1_prev, matches_t0_t1_next):
        key_prev = tuple(int(c) for c in coord_prev)
        key_next = tuple(int(c) for c in coord_next)
        if key_prev in t0_coords_in_mask_0 and key_next in t1_coords_in_mask_1:
            idx_matches_0.append(t0_coords_in_mask_0[key_prev])
            idx_matches_1.append(t1_coords_in_mask_1[key_next])

    idx_matches_0 = np.asarray(idx_matches_0, dtype=np.int64)
    idx_matches_1 = np.asarray(idx_matches_1, dtype=np.int64)

    b_v0 = edges_loaded["b_v"][0].astype(np.uint8)
    b_v1 = edges_loaded["b_v"][1].astype(np.uint8)

    branch_labels_0 = np.argmax(b_v0, axis=0)
    branch_labels_1 = np.argmax(b_v1, axis=0)

    branch_ids_0 = branch_labels_0[idx_matches_0]
    branch_ids_1 = branch_labels_1[idx_matches_1]

    b0_b1 = accumulate_pair_counts(
        branch_ids_0,
        branch_ids_1,
        n_src=b_v0.shape[0],
        n_dst=b_v1.shape[0],
        use_gpu=True,
    )

    max_idx = np.argmax(b0_b1, axis=0) + 1

    mask_branches = np.zeros(mask_01.shape, dtype=np.uint16)
    mask_branches[0][tuple(mask_voxels_0.T)] = branch_labels_0 + 1

    new_branch_labels_1 = max_idx[branch_labels_1]
    mask_branches[1][tuple(mask_voxels_1.T)] = new_branch_labels_1

    # ------------------------------------------------------------------
    # Example: node adjacency between t0 and t1 (analogous to branches)
    # ------------------------------------------------------------------
    n_v0 = edges_loaded["n_v"][0].astype(np.uint8)
    n_v1 = edges_loaded["n_v"][1].astype(np.uint8)

    node_labels_0 = np.argmax(n_v0, axis=0)
    node_labels_1 = np.argmax(n_v1, axis=0)

    node_ids_0 = node_labels_0[idx_matches_0]
    node_ids_1 = node_labels_1[idx_matches_1]

    n0_n1 = accumulate_pair_counts(
        node_ids_0,
        node_ids_1,
        n_src=n_v0.shape[0],
        n_dst=n_v1.shape[0],
        use_gpu=True,
    )

    max_idx_n = np.argmax(n0_n1, axis=0) + 1

    mask_nodes = np.zeros(mask_01.shape, dtype=np.uint16)
    mask_nodes[0][tuple(mask_voxels_0.T)] = node_labels_0 + 1
    new_node_labels_1 = max_idx_n[node_labels_1]
    mask_nodes[1][tuple(mask_voxels_1.T)] = new_node_labels_1


if __name__ == "__main__":
    main()
