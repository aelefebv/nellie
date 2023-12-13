from src import logger
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
import numpy as np
import pandas as pd


class VoxelMovement:
    def __init__(self, im_info: ImInfo, num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if self.im_info.no_z:
            self.scaling = (im_info.dim_sizes['Y'], im_info.dim_sizes['X'])
        else:
            self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.voxel_matches_path = None
        self.label_memmap = None

        self.debug = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for voxel movement analysis.')
        self.voxel_matches_path = self.im_info.pipeline_paths['voxel_matches']

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

    def _get_match_vec(self, match):
        scaled_0 = match[0].astype('float32') * self.scaling
        scaled_1 = match[1].astype('float32') * self.scaling
        match_vec = scaled_1 - scaled_0
        # euc_dist = np.linalg.norm(scaled_0 - scaled_1, axis=1)
        return match_vec

    def _get_min_euc_dist(self, labels, match_vec):
        euc_dist = np.linalg.norm(match_vec, axis=1)
        labels = np.array(labels)

        df = pd.DataFrame({'labels': labels, 'euc_dist': euc_dist})

        idxmin = df.groupby('labels')['euc_dist'].idxmin()
        return idxmin

    def _get_reference_vector(self, match, idxmin, match_vec, labels):
        idxmin_0 = idxmin.index.values
        idxmin_1 = idxmin.values

        # get the index of idxmin_0 that matches each item in labels
        idxmin_0_idx = np.searchsorted(idxmin_0, labels)
        ref_vec_idxs = idxmin_1[idxmin_0_idx]
        ref_points = match[0][ref_vec_idxs]
        ref_vecs = match_vec[ref_vec_idxs]
        return ref_vecs, ref_points


    def _run_frame(self, t, match):
        labels = self.label_memmap[t][tuple(match[0].T)]
        match_vec = self._get_match_vec(match)
        idxmin = self._get_min_euc_dist(labels, match_vec)
        ref_vecs, ref_points = self._get_reference_vector(match, idxmin, match_vec, labels)
        ref_vec_subtracted_vecs = match_vec - ref_vecs
        ref_point_subtracted_points_0 = (match[0].astype('float32') - ref_points) * self.scaling
        ref_point_subtracted_points_1 = (match[1].astype('float32') - ref_points) * self.scaling
        magnitude = np.linalg.norm(ref_vec_subtracted_vecs, axis=1)
        angle = np.arctan2(ref_vec_subtracted_vecs[:, 1], ref_vec_subtracted_vecs[:, 0])
        # get angle in degrees between 0 and 180
        angle = np.abs(angle) * 180 / np.pi
        angle = np.where(angle > 180, 360 - angle, angle)


        # viewer.add_points(match[0][idxmin], size=1, face_color='red')
        # viewer.add_points(match[1][idxmin], size=1, face_color='blue')

    def _run_voxel_movement_analysis(self):
        voxel_matches = np.load(self.voxel_matches_path, allow_pickle=True)
        for t in range(self.num_t):
            self._run_frame(t, voxel_matches[t])
        print('hi')

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_voxel_movement_analysis()

if __name__ == "__main__":
    tif_file = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(tif_file)
    run_obj = VoxelMovement(im_info, num_t=3)
    run_obj.run()
