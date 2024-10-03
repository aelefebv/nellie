import heapq

import numpy as np
from scipy.spatial import cKDTree

from nellie import logger
from nellie.im_info.verifier import ImInfo
from nellie.tracking.flow_interpolation import FlowInterpolator


class VoxelReassigner:
    """
    A class for voxel reassignment across time points using forward and backward flow interpolation.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    num_t : int
        Number of timepoints in the image.
    flow_interpolator_fw : FlowInterpolator
        Flow interpolator for forward timepoint matching.
    flow_interpolator_bw : FlowInterpolator
        Flow interpolator for backward timepoint matching.
    running_matches : list
        List of running matches for voxel reassignment between timepoints.
    voxel_matches_path : str or None
        Path to save the voxel matches array.
    branch_label_memmap : np.ndarray or None
        Memory-mapped data for relabeled branches.
    obj_label_memmap : np.ndarray or None
        Memory-mapped data for object labels.
    reassigned_branch_memmap : np.ndarray or None
        Memory-mapped data for reassigned branches.
    reassigned_obj_memmap : np.ndarray or None
        Memory-mapped data for reassigned object labels.
    viewer : Any
        Optional viewer (e.g., for visualization purposes).

    Methods
    -------
    _match_forward(flow_interpolator, vox_prev, vox_next, t)
        Matches voxels forward using flow interpolation.
    _match_backward(flow_interpolator, vox_next, vox_prev, t)
        Matches voxels backward using flow interpolation.
    _match_voxels_to_centroids(coords_real, coords_interpx)
        Matches voxels to centroids using nearest neighbor search.
    _assign_unique_matches(vox_prev_matches, vox_next_matches, distances)
        Assigns unique matches between timepoint voxels based on minimum distances.
    _distance_threshold(vox_prev_matched, vox_next_matched)
        Filters voxel matches by applying a distance threshold.
    match_voxels(vox_prev, vox_next, t)
        Matches voxels between two consecutive timepoints using forward and backward interpolation.
    _get_t()
        Gets the number of timepoints in the dataset.
    _allocate_memory()
        Allocates memory for voxel reassignment data, including memory-mapped arrays.
    _run_frame(t, all_mask_coords, reassigned_memmap)
        Runs the voxel reassignment process for a single timepoint.
    _run_reassignment(label_type)
        Runs the voxel reassignment process for all frames, for either branch or object labels.
    run()
        Main method to execute voxel reassignment for both branch and object labels.
    """
    def __init__(self, im_info: ImInfo, num_t=None,
                 viewer=None):
        """
        Initializes the VoxelReassigner class with image metadata and timepoints.

        Parameters
        ----------
        im_info : ImInfo
            Image metadata and memory-mapped data.
        num_t : int, optional
            Number of timepoints in the dataset. If None, it is inferred from the image metadata (default is None).
        viewer : Any, optional
            Optional viewer for visualization purposes (default is None).
        """
        self.im_info = im_info

        if self.im_info.no_t:
            return

        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        self.running_matches = []

        self.voxel_matches_path = None
        self.branch_label_memmap = None
        self.obj_label_memmap = None
        self.reassigned_branch_memmap = None
        self.reassigned_obj_memmap = None

        self.debug = None

        self.viewer = viewer

    def _match_forward(self, flow_interpolator, vox_prev, vox_next, t):
        """
        Matches voxels forward in time using flow interpolation.

        Parameters
        ----------
        flow_interpolator : FlowInterpolator
            Flow interpolator for forward voxel matching.
        vox_prev : np.ndarray
            Voxels from the previous timepoint.
        vox_next : np.ndarray
            Voxels from the next timepoint.
        t : int
            Current timepoint index.

        Returns
        -------
        tuple
            Arrays of matched voxels from the previous and next timepoints and valid distances between them.
        """
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
        """
        Matches voxels backward in time using flow interpolation.

        Parameters
        ----------
        flow_interpolator : FlowInterpolator
            Flow interpolator for backward voxel matching.
        vox_next : np.ndarray
            Voxels from the next timepoint.
        vox_prev : np.ndarray
            Voxels from the previous timepoint.
        t : int
            Current timepoint index.

        Returns
        -------
        tuple
            Arrays of matched voxels from the previous and next timepoints and valid distances between them.
        """
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
        """
        Matches real voxel coordinates to interpolated centroids using nearest neighbor search.

        Parameters
        ----------
        coords_real : np.ndarray
            Real voxel coordinates.
        coords_interpx : np.ndarray
            Interpolated centroid coordinates.

        Returns
        -------
        tuple
            Arrays of distances and indices of matched centroids.
        """
        coords_interpx = np.array(coords_interpx) * self.flow_interpolator_fw.scaling
        coords_real = np.array(coords_real) * self.flow_interpolator_fw.scaling
        tree = cKDTree(coords_real)
        dist, idx = tree.query(coords_interpx, k=1, workers=-1)
        return dist, idx

    def _assign_unique_matches(self, vox_prev_matches, vox_next_matches, distances):
        """
        Assigns unique voxel matches based on the minimum distance criteria.

        Parameters
        ----------
        vox_prev_matches : np.ndarray
            Array of matched voxels from the previous timepoint.
        vox_next_matches : np.ndarray
            Array of matched voxels from the next timepoint.
        distances : np.ndarray
            Array of distances between matched voxels.

        Returns
        -------
        tuple
            Arrays of uniquely matched voxels for the previous and next timepoints.
        """
        # create a dict where the key is a voxel in t1, and the value is a list of distances and t0 voxels matched to it
        vox_next_dict = {}
        for match_idx, match_next in enumerate(vox_next_matches):
            match_next_tuple = tuple(match_next)
            if match_next_tuple not in vox_next_dict.keys():
                vox_next_dict[match_next_tuple] = [[], []]
            vox_next_dict[match_next_tuple][0].append(distances[match_idx])
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

        # create a priority queue with (distance, prev_voxel, next_voxel) tuples
        priority_queue = [(distances[i], tuple(vox_prev_matches[i]), tuple(vox_next_matches[i]))
                          for i in range(len(distances))]
        heapq.heapify(priority_queue)  # Convert list to a heap in-place

        assigned_prev = set()
        assigned_next = set()
        vox_prev_matches_final = []
        vox_next_matches_final = []

        while priority_queue:
            # pop the smallest distance tuple from the heap
            distance, prev_voxel, next_voxel = heapq.heappop(priority_queue)

            if prev_voxel not in assigned_prev or next_voxel not in assigned_next:
                # if neither of the voxels has been assigned, then assign them
                vox_prev_matches_final.append(prev_voxel)
                vox_next_matches_final.append(next_voxel)
                assigned_prev.add(prev_voxel)
                assigned_next.add(next_voxel)

        return vox_prev_matches_final, vox_next_matches_final

    def _distance_threshold(self, vox_prev_matched, vox_next_matched):
        """
        Filters voxel matches by applying a distance threshold.

        Parameters
        ----------
        vox_prev_matched : np.ndarray
            Array of matched voxels from the previous timepoint.
        vox_next_matched : np.ndarray
            Array of matched voxels from the next timepoint.

        Returns
        -------
        tuple
            Arrays of valid voxel matches and corresponding distances.
        """
        distances = np.linalg.norm((vox_prev_matched - vox_next_matched) * self.flow_interpolator_fw.scaling, axis=1)
        distance_mask = distances < self.flow_interpolator_fw.max_distance_um
        vox_prev_matched_valid = vox_prev_matched[distance_mask]
        vox_next_matched_valid = vox_next_matched[distance_mask]
        distances_valid = distances[distance_mask]
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def match_voxels(self, vox_prev, vox_next, t):
        """
        Matches voxels between two consecutive timepoints using both forward and backward interpolation.

        Parameters
        ----------
        vox_prev : np.ndarray
            Voxels from the previous timepoint.
        vox_next : np.ndarray
            Voxels from the next timepoint.
        t : int
            Current timepoint index.

        Returns
        -------
        tuple
            Arrays of matched voxels from the previous and next timepoints.
        """
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

        vox_prev_matches_unique, vox_next_matches_unique = self._assign_unique_matches(vox_prev_matches,
                                                                                       vox_next_matches, distances)

        vox_next_matches_unique = np.array(vox_next_matches_unique)
        if len(vox_next_matches_unique) == 0:
            return [], []
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
        """
        Gets the number of timepoints from the image metadata or sets it if not provided.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        """
        Allocates memory for voxel reassignment, including initializing memory-mapped arrays for branch and object labels.
        """
        logger.debug('Allocating memory for voxel reassignment.')
        self.voxel_matches_path = self.im_info.pipeline_paths['voxel_matches']

        self.branch_label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.obj_label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.shape = self.branch_label_memmap.shape

        reassigned_branch_label_path = self.im_info.pipeline_paths['im_branch_label_reassigned']
        self.reassigned_branch_memmap = self.im_info.allocate_memory(reassigned_branch_label_path,
                                                                     dtype='int32',
                                                                     description='branch label reassigned',
                                                                     return_memmap=True)

        reassigned_obj_label_path = self.im_info.pipeline_paths['im_obj_label_reassigned']
        self.reassigned_obj_memmap = self.im_info.allocate_memory(reassigned_obj_label_path,
                                                                  dtype='int32',
                                                                  description='object label reassigned',
                                                                  return_memmap=True)

    def _run_frame(self, t, all_mask_coords, reassigned_memmap):
        """
        Reassigns voxels in a single timepoint based on voxel matches with the previous timepoint.

        Parameters
        ----------
        t : int
            Current timepoint index.
        all_mask_coords : list
            List of voxel coordinates for each timepoint.
        reassigned_memmap : np.ndarray
            Memory-mapped array for the reassigned labels.

        Returns
        -------
        bool
            Returns True if no matches are found, otherwise False.
        """
        logger.info(f'Reassigning pixels in frame {t + 1} of {self.num_t - 1}')

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

        reassigned_memmap[t + 1][tuple(matched_next.T)] = reassigned_memmap[t][tuple(matched_prev.T)]

        return False

    def _run_reassignment(self, label_type):
        """
        Runs voxel reassignment for all frames based on the specified label type (either 'branch' or 'obj').

        Parameters
        ----------
        label_type : str
            The label type, either 'branch' or 'obj'.
        """
        # todo, be able to specify which frame to start at.
        if label_type == 'branch':
            label_memmap = self.branch_label_memmap
            reassigned_memmap = self.reassigned_branch_memmap
        elif label_type == 'obj':
            label_memmap = self.obj_label_memmap
            reassigned_memmap = self.reassigned_obj_memmap
        else:
            raise ValueError('label_type must be "branch" or "obj".')
        vox_prev = np.argwhere(label_memmap[0] > 0)
        reassigned_memmap[0][tuple(vox_prev.T)] = label_memmap[0][tuple(vox_prev.T)]
        all_mask_coords = [np.argwhere(label_memmap[t] > 0) for t in range(self.num_t)]

        for t in range(self.num_t - 1):
            if self.viewer is not None:
                self.viewer.status = f'Reassigning voxels. Frame: {t + 1} of {self.num_t}.'
            no_matches = self._run_frame(t, all_mask_coords, reassigned_memmap)

            if no_matches:
                break

    def run(self):
        """
        Main method to execute voxel reassignment for both branch and object labels.
        """
        if self.im_info.no_t:
            return
        self._get_t()
        self._allocate_memory()
        self._run_reassignment('branch')
        self._run_reassignment('obj')
        # save running matches to npy
        np.save(self.voxel_matches_path, np.array(self.running_matches, dtype=object))


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3
    run_obj = VoxelReassigner(im_info, num_t=num_t)
    run_obj.run()

    import pickle

    # This section seems random, but it allows for finding links between any level of the hierarchy to any
    #  other level in the hierarchy at any time point via lots of dot products.
    edges_loaded = pickle.load(open(im_info.pipeline_paths['adjacency_maps'], "rb"))

    mask_01 = run_obj.obj_label_memmap[:2] > 0
    mask_voxels_0 = np.argwhere(mask_01[0])
    mask_voxels_1 = np.argwhere(mask_01[1])

    t0_coords_in_mask_0 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_0)}
    t1_coords_in_mask_1 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_1)}

    idx_matches_0 = [t0_coords_in_mask_0[tuple(coord)] for coord in run_obj.running_matches[0][0]]
    idx_matches_1 = [t1_coords_in_mask_1[tuple(coord)] for coord in run_obj.running_matches[0][1]]
    # sort based on idx_matches_0
    sorted_idx_matches_0_order = np.argsort(idx_matches_0)
    sorted_idx_matches_0 = np.array(idx_matches_0)[sorted_idx_matches_0_order]
    sorted_idx_matches_1 = np.array(idx_matches_1)[sorted_idx_matches_0_order]

    v_t = np.zeros((len(mask_voxels_0), len(mask_voxels_1)), dtype=np.uint16)
    v_t[sorted_idx_matches_0, sorted_idx_matches_1] = True

    b_v = edges_loaded['b_v'][0].astype(np.uint16)
    # dot product b_v and v_t
    import cupy as cp


    def dot_product_in_chunks(a, b, chunk_size=100):
        result = cp.zeros((a.shape[0], b.shape[1]), dtype=cp.uint8)
        for start_row in range(0, a.shape[1], chunk_size):
            print(start_row, start_row + chunk_size)
            end_row = start_row + chunk_size
            v_t_chunk = b[start_row:end_row, :]
            b_v_chunk = a[:, start_row:end_row]
            result += cp.dot(b_v_chunk, v_t_chunk)  # Adjust this line as per your logic
        return result


    # Convert your numpy arrays to cupy arrays
    v_t_cp = cp.array(v_t, dtype=cp.uint8)
    b_v_cp = cp.array(b_v, dtype=cp.uint8)

    # Perform dot product in chunks
    b0_v1_cp = dot_product_in_chunks(b_v_cp, v_t_cp)

    # Convert the result back to a numpy array if needed
    b1_v1 = edges_loaded['b_v'][1].astype(np.uint16)
    b1_v1_cp = cp.array(b1_v1, dtype=cp.uint8)
    b0_b1_cp = dot_product_in_chunks(b0_v1_cp, b1_v1_cp.T)
    b0_b1 = cp.asnumpy(b0_b1_cp)
    # b0_b1 are the new edges between branches in time 0 and time 1, values are the number of voxels in common between (aka weighting to give the edges)

    # find the indices of the maximum value in each col
    max_idx = np.argmax(b0_b1, axis=0) + 1

    mask_branches = np.zeros(mask_01.shape, dtype=np.uint16)
    branch_labels_0 = np.argmax(b_v.T, axis=1)
    branch_labels_1 = np.argmax(b1_v1.T, axis=1)

    mask_branches[0][tuple(mask_voxels_0.T)] = branch_labels_0 + 1

    # replace any non-zero values in b1_v1 with the max_idx
    new_branch_labels_1 = max_idx[branch_labels_1]
    mask_branches[1][tuple(mask_voxels_1.T)] = new_branch_labels_1 + 1
    # these branches are relabelled by t0 branch labels.

    # lets do this with nodes, too
    n_v = edges_loaded['n_v'][0].astype(np.uint16)
    n_v_cp = cp.array(n_v, dtype=cp.uint8)
    n0_v1_cp = dot_product_in_chunks(n_v_cp, v_t_cp)
    n1_v1 = edges_loaded['n_v'][1].astype(np.uint16)
    n1_v1_cp = cp.array(n1_v1, dtype=cp.uint8)
    n0_n1_cp = dot_product_in_chunks(n0_v1_cp, n1_v1_cp.T)

    n0_n1 = cp.asnumpy(n0_n1_cp)
    max_idx_n = np.argmax(n0_n1, axis=0) + 1

    mask_nodes = np.zeros(mask_01.shape, dtype=np.uint16)
    node_labels_0 = np.argmax(n_v.T, axis=1)
    node_labels_1 = np.argmax(n1_v1.T, axis=1)
    mask_nodes[0][tuple(mask_voxels_0.T)] = node_labels_0 + 1
    new_node_labels_1 = max_idx_n[node_labels_1]
    mask_nodes[1][tuple(mask_voxels_1.T)] = new_node_labels_1 + 1

    # # todo useful for getting continuous tracks for voxels
    # matches_t0_t1 = run_obj.running_matches[0][1]
    # matches_t1_t2 = run_obj.running_matches[1][0]
    #
    # t1_coords_in_t0_t1 = {tuple(coord): idx for idx, coord in enumerate(matches_t0_t1)}
    # t1_coords_in_t1_t2 = {tuple(coord): idx for idx, coord in enumerate(matches_t1_t2)}
    #
    # t0_coords_in_t0_t1 = {tuple(coord): idx for idx, coord in enumerate(run_obj.running_matches[0][0])}
    #
    # # Create the continuous track list
    # continuous_tracks = []
    # for t1_coord, t0_idx in t1_coords_in_t0_t1.items():
    #     if t1_coord in t1_coords_in_t1_t2:
    #         t0_coord = run_obj.running_matches[0][0][t0_idx]
    #         t2_idx = t1_coords_in_t1_t2[t1_coord]
    #         t2_coord = run_obj.running_matches[1][1][t2_idx]
    #         continuous_tracks.append([t0_coord, t1_coord, t2_coord])
    #
    # napari_tracks = []
    # for i, track in enumerate(continuous_tracks):
    #     napari_tracks.append([i, 0, track[0][0], track[0][1], track[0][2]])
    #     napari_tracks.append([i, 1, track[1][0], track[1][1], track[1][2]])
    #     napari_tracks.append([i, 2, track[2][0], track[2][1], track[2][2]])
