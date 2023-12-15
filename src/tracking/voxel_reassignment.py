import heapq
import numpy as np
from scipy.spatial import cKDTree
from src import logger
from src.im_info.im_info import ImInfo
from src.tracking.flow_interpolation import FlowInterpolator
from src.utils.general import get_reshaped_image


class VoxelReassigner:
    def __init__(self, im_info: ImInfo,
                 num_t=None, skeleton_labels=True):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        self.skeleton_labels = skeleton_labels

        self.running_matches = []

        self.voxel_matches_path = None
        self.label_memmap = None
        self.reassigned_memmap = None

        self.debug = None

    def _match_forward(self, flow_interpolator, vox_prev, vox_next, t):
        vectors_interpx_prev = flow_interpolator.interpolate_coord(vox_prev, t)
        if vectors_interpx_prev is None:
            return [], [], []
        # only keep voxels that are not nan
        kept_prev_vox_idxs = ~np.isnan(vectors_interpx_prev).any(axis=1)
        # only keep vectors where the voxel is not nan
        vectors_interpx_prev = vectors_interpx_prev[kept_prev_vox_idxs]
        # get centroids in t1 from voxels in t0 + interpolated flow at that voxel
        vox_prev_kept = vox_prev[kept_prev_vox_idxs]
        centroids_next_interpx = vox_prev_kept + vectors_interpx_prev
        if len(centroids_next_interpx) == 0:
            return [], [], []
        # now we have estimated centroids in t1 (centroids_next_interpx) and linked voxels in t0 (vox_prev[kept_prev_vox_idxs]).
        # we then have to match t1 voxels (vox_next) to estimated t1 centroids (centroids_next_interpx)
        match_dist, matched_idx = self._match_voxels_to_centroids(vox_next, centroids_next_interpx)
        vox_matched_to_centroids = vox_next[matched_idx.tolist()]
        # then link those t1 voxels back to the t0 voxels
        # now we have linked t0 voxels (vox_prev_kept) to t1 voxels (vox_matched_to_centroids)
        # but we have to make sure the link is within a distance constraint.
        vox_prev_matched_valid, vox_next_matched_valid, distances_valid = self._distance_threshold(
            vox_prev_kept, vox_matched_to_centroids
        )
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def _match_backward(self, flow_interpolator, vox_next, vox_prev, t):
        # interpolate flow vectors to all voxels in t1 from centroids derived from t0 centroids + t0 flow vectors
        vectors_interpx_prev = flow_interpolator.interpolate_coord(vox_next, t)
        if vectors_interpx_prev is None:
            return [], [], []
        # only keep voxels that are not nan
        kept_next_vox_idxs = ~np.isnan(vectors_interpx_prev).any(axis=1)
        # only keep vectors where the voxel is not nan
        vectors_interpx_prev = vectors_interpx_prev[kept_next_vox_idxs]
        # get centroids in t0 from voxels in t1 - interpolated flow (from t0 to t1) at that voxel
        vox_next_kept = vox_next[kept_next_vox_idxs]
        centroids_prev_interpx = vox_next_kept - vectors_interpx_prev
        if len(centroids_prev_interpx) == 0:
            return [], [], []
        # now we have estimated centroids in t0 (centroids_prev_interpx) and linked voxels in t1 (vox_next[kept_next_vox_idxs]).
        # we then have to match t0 voxels (vox_prev) to estimated t0 centroids (centroids_prev_interpx)
        match_dist, matched_idx = self._match_voxels_to_centroids(vox_prev, centroids_prev_interpx)
        vox_matched_to_centroids = vox_prev[matched_idx.tolist()]
        # then link those t1 voxels (vox_next_kept) back to the t0 voxels (vox_matched_to_centroids).
        # but we have to make sure the link is within a distance constraint.
        vox_prev_matched_valid, vox_next_matched_valid, distances_valid = self._distance_threshold(
            vox_matched_to_centroids, vox_next_kept
        )
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def _match_voxels_to_centroids(self, coords_real, coords_interpx):
        coords_interpx = np.array(coords_interpx) * self.flow_interpolator_fw.scaling
        coords_real = np.array(coords_real) * self.flow_interpolator_fw.scaling
        tree = cKDTree(coords_real)
        dist, idx = tree.query(coords_interpx, k=1, workers=-1)
        return dist, idx

    def _assign_unique_matches(self, vox_prev_matches, vox_next_matches, distances):
        # create a dict where the key is a voxel in t1, and the value is a list of distances and t0 voxels matched to it
        vox_next_dict = {}
        for match_idx, match_next in enumerate(vox_next_matches):
            match_next_tuple = tuple(match_next)
            if match_next_tuple not in vox_next_dict.keys():
                vox_next_dict[match_next_tuple] = [[], []]
            vox_next_dict[match_next_tuple] [0].append(distances[match_idx])
            vox_next_dict[match_next_tuple][1].append(vox_prev_matches[match_idx])

        # now assign matches based on the t1 voxel's closest (in distance) matched t0 voxel
        vox_prev_matches_final = []
        vox_next_matches_final = []
        for match_next_tuple, (distance_match_list, vox_prev_match_list) in vox_next_dict.items():
            if len(distance_match_list) == 1:
                vox_prev_matches_final.append(vox_prev_match_list[0])
                vox_next_matches_final.append(match_next_tuple)
                continue
            min_idx = np.argmin(distance_match_list)
            vox_prev_matches_final.append(vox_prev_match_list[min_idx])
            vox_next_matches_final.append(match_next_tuple)
        #
        # vox_prev_dict = {}
        # for match_idx, match_prev in enumerate(vox_prev_matches):
        #     match_prev_tuple = tuple(match_prev)
        #     if match_prev_tuple not in vox_prev_dict.keys():
        #         vox_prev_dict[match_prev_tuple] = [[], []]
        #     vox_prev_dict[match_prev_tuple][0].append(distances[match_idx])
        #     vox_prev_dict[match_prev_tuple][1].append(vox_next_matches[match_idx])
        #
        # vox_prev_matches_final_2 = []
        # vox_next_matches_final_2 = []
        # for match_prev_tuple, (distance_match_list, vox_next_match_list) in vox_prev_dict.items():
        #     if len(distance_match_list) == 1:
        #         vox_prev_matches_final_2.append(match_prev_tuple)
        #         vox_next_matches_final_2.append(vox_next_match_list[0])
        #         continue
        #     min_idx = np.argmin(distance_match_list)
        #     vox_prev_matches_final_2.append(match_prev_tuple)
        #     vox_next_matches_final_2.append(vox_next_match_list[min_idx])

        # Create a priority queue with (distance, prev_voxel, next_voxel) tuples
        priority_queue = [(distances[i], tuple(vox_prev_matches[i]), tuple(vox_next_matches[i]))
                          for i in range(len(distances))]
        heapq.heapify(priority_queue)  # Convert list to a heap in-place

        assigned_prev = set()
        assigned_next = set()
        vox_prev_matches_final = []
        vox_next_matches_final = []

        while priority_queue:
            # Pop the smallest distance tuple from the heap
            distance, prev_voxel, next_voxel = heapq.heappop(priority_queue)

            if prev_voxel not in assigned_prev or next_voxel not in assigned_next:
                # If neither of the voxels has been assigned, then assign them
                vox_prev_matches_final.append(prev_voxel)
                vox_next_matches_final.append(next_voxel)
                assigned_prev.add(prev_voxel)
                assigned_next.add(next_voxel)

        return vox_prev_matches_final, vox_next_matches_final


        # return vox_prev_matches_final_1, vox_next_matches_final_1

    def _distance_threshold(self, vox_prev_matched, vox_next_matched):
        distances = np.linalg.norm((vox_prev_matched - vox_next_matched) * self.flow_interpolator_fw.scaling, axis=1)
        distance_mask = distances < self.flow_interpolator_fw.max_distance_um
        vox_prev_matched_valid = vox_prev_matched[distance_mask]
        vox_next_matched_valid = vox_next_matched[distance_mask]
        distances_valid = distances[distance_mask]
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def match_voxels(self, vox_prev, vox_next, t):
        # forward interpolation:
        # from t0 voxels and interpolated flow, get t1 centroids.
        #  match nearby t1 voxels to t1 centroids, which are linked to t0 voxels.
        logger.debug(f'Forward voxel matching for t: {t}')
        vox_prev_matches_fw, vox_next_matches_fw, distances_fw = self._match_forward(
            self.flow_interpolator_fw, vox_prev, vox_next, t
        )

        # backward interpolation:
        # from t0 centroids and real flow, get t1 centroids.
        #  interpolate flow at nearby t1 voxels. subtract flow from voxels to get t0 centroids.
        #  match nearby t0 voxels to t0 centroids, which are linked to t1 voxels.
        logger.debug(f'Backward voxel matching for t: {t}')
        vox_prev_matches_bw, vox_next_matches_bw, distances_bw = self._match_backward(
            self.flow_interpolator_bw, vox_next, vox_prev, t + 1
        )

        logger.debug(f'Assigning unique matches for t: {t}')
        vox_prev_matches = np.concatenate([vox_prev_matches_fw, vox_prev_matches_bw])
        vox_next_matches = np.concatenate([vox_next_matches_fw, vox_next_matches_bw])
        distances = np.concatenate([distances_fw, distances_bw])

        vox_prev_matches_unique, vox_next_matches_unique = self._assign_unique_matches(vox_prev_matches, vox_next_matches, distances)

        # return vox_prev_matches_unique, vox_next_matches_unique


        vox_next_matches_unique = np.array(vox_next_matches_unique)
        vox_next_matched_tuples = set([tuple(coord) for coord in vox_next_matches_unique])
        vox_next_unmatched = np.array([coord for coord in vox_next if tuple(coord) not in vox_next_matched_tuples])
        unmatched_diff = np.inf
        while unmatched_diff:
            num_unmatched = len(vox_next_unmatched)
            if num_unmatched == 0:
                break
            tree = cKDTree(vox_next_matches_unique * self.flow_interpolator_fw.scaling)
            dists, idxs = tree.query(vox_next_unmatched * self.flow_interpolator_fw.scaling, k=1, workers=-1)
            unmatched_matches = np.array([
                [vox_prev_matches_unique[idx], vox_next_unmatched[i]]
                for i, idx in enumerate(idxs) if dists[i] < self.flow_interpolator_fw.max_distance_um
            ])
            if len(unmatched_matches) == 0:
                break
            # add unmatched matches to coords_matched
            vox_prev_matches_unique = np.concatenate([vox_prev_matches_unique, unmatched_matches[:, 0]])
            vox_next_matches_unique = np.concatenate([vox_next_matches_unique, unmatched_matches[:, 1]])
            vox_next_matched_tuples = set([tuple(coord) for coord in vox_next_matches_unique])
            vox_next_unmatched = np.array([coord for coord in vox_next if tuple(coord) not in vox_next_matched_tuples])
            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff = num_unmatched - new_num_unmatched
            logger.debug(f'Reassigned {unmatched_diff}/{num_unmatched} unassigned voxels. '
                         f'{new_num_unmatched} remain.')
        return np.array(vox_prev_matches_unique), np.array(vox_next_matches_unique)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for voxel reassignment.')
        self.voxel_matches_path = self.im_info.pipeline_paths['voxel_matches']

        if self.skeleton_labels:
            label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        else:
            label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

        # reassigned_label_path = self.im_info.create_output_path('im_instance_label_reassigned')
        reassigned_label_path = self.im_info.pipeline_paths['im_instance_label_reassigned']
        self.reassigned_memmap = self.im_info.allocate_memory(reassigned_label_path, shape=self.shape,
                                                              dtype='int32',
                                                              description='instance segmentation',
                                                              return_memmap=True)

    def _run_frame(self, t, all_mask_coords):
        logger.info(f'Reassigning pixels in frame {t+1} of {self.num_t - 1}')

        vox_prev = all_mask_coords[t]
        vox_next = all_mask_coords[t + 1]
        if len(vox_prev) == 0 or len(vox_next) == 0:
            return True

        matched_prev, matched_next = self.match_voxels(vox_prev, vox_next, t)
        if len(matched_prev) == 0:
            return True
        matched_prev = matched_prev.astype('uint16')
        matched_next = matched_next.astype('uint16')

        self.running_matches.append([matched_prev, matched_next])

        # save the matches to a npy file
        # np.save(self.voxel_matches_path, np.array([matched_prev, matched_next]))

        self.reassigned_memmap[t + 1][tuple(matched_next.T)] = self.reassigned_memmap[t][tuple(matched_prev.T)]

        return False

    def _run_reassignment(self):
        vox_prev = np.argwhere(self.label_memmap[0] > 0)
        self.reassigned_memmap[0][tuple(vox_prev.T)] = self.label_memmap[0][tuple(vox_prev.T)]
        all_mask_coords = [np.argwhere(self.label_memmap[t] > 0) for t in range(self.num_t)]

        for t in range(self.num_t - 1):
            no_matches = self._run_frame(t, all_mask_coords)

            if no_matches:
                break

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_reassignment()
        # save running matches to npy
        np.save(self.voxel_matches_path, np.array(self.running_matches, dtype=object))


if __name__ == "__main__":
    tif_file = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(tif_file)
    run_obj = VoxelReassigner(im_info, num_t=3)
    run_obj.run()

    # import os
    # top_dir = r"D:\test_files\nelly_gav_tests"
    # # get all non-folder files
    # all_files = os.listdir(top_dir)
    # all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    # for file_num, tif_file in enumerate(all_files):
    #     im_info = ImInfo(tif_file)
    #     print(f'Processing file {file_num + 1} of {len(all_files)}')
    #     im_info.create_output_path('im_instance_label')
    #     run_obj = VoxelReassigner(im_info)
    #     run_obj.run()