from src import logger
from src_2.im_info.im_info import ImInfo
from src_2.tracking.flow_interpolation import FlowInterpolator
from src_2.utils.general import get_reshaped_image
import numpy as np
import pandas as pd


class CoordMovement:
    def __init__(self, im_info: ImInfo, num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if self.im_info.no_z:
            self.scaling = (im_info.dim_sizes['Y'], im_info.dim_sizes['X'])
        else:
            self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

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
        # remove nan values
        df = df[~np.isnan(df['euc_dist'])]

        idxmin = df.groupby('labels')['euc_dist'].idxmin()
        return idxmin

    def _get_reference_vector(self, match, idxmin, match_vec, labels):
        idxmin_0 = idxmin.index.values
        idxmin_1 = idxmin.values

        # get the index of idxmin_0 that matches each item in labels
        idxmin_0_idx = np.searchsorted(idxmin_0, labels)
        nan_idxs = np.argwhere(np.isnan(idxmin_0_idx))
        ref_vec_idxs = idxmin_1[idxmin_0_idx].astype('int32')
        # test =
        ref_points = match[:, 0][ref_vec_idxs]
        ref_vecs = match_vec[ref_vec_idxs]
        return ref_vecs, ref_points


    def _run_frame(self, t):
        label_vals = self.label_memmap[t][self.label_memmap[t]>0]
        df = pd.DataFrame(label_vals, columns=['label_vals'])
        coords_1 = np.argwhere(self.label_memmap[t]>0).astype('float32')
        df[['t1_z', 't1_y', 't1_x']] = pd.DataFrame(coords_1)

        vec12 = self.flow_interpolator_fw.interpolate_coord(coords_1, t)
        vec01 = self.flow_interpolator_bw.interpolate_coord(coords_1, t)

        coords_0 = coords_1 - vec01
        coords_2 = coords_1 + vec12
        df[['t0_z', 't0_y', 't0_x']] = pd.DataFrame(coords_0)
        df[['t2_z', 't2_y', 't2_x']] = pd.DataFrame(coords_2)

        vec12_scaled = vec12 * self.scaling
        vec01_scaled = vec01 * self.scaling
        df[['vec12_z', 'vec12_y', 'vec12_x']] = pd.DataFrame(vec12_scaled)
        df[['vec01_z', 'vec01_y', 'vec01_x']] = pd.DataFrame(vec01_scaled)

        idxmin_01 = self._get_min_euc_dist(label_vals, vec01_scaled)
        idxmin_12 = self._get_min_euc_dist(label_vals, vec12_scaled)

        match_01 = np.stack([coords_0, coords_1], axis=1)
        match_12 = np.stack([coords_1, coords_2], axis=1)

        ref_vecs_01, ref_points_01 = self._get_reference_vector(match_01, idxmin_01, vec01_scaled, label_vals)
        ref_vecs_12, ref_points_12 = self._get_reference_vector(match_12, idxmin_12, vec12_scaled, label_vals)

        ref_vec_subtracted_vecs_01 = vec01_scaled - ref_vecs_01
        ref_vec_subtracted_vecs_12 = vec12_scaled - ref_vecs_12

        # ref_point_subtracted_points_01_0 = (match_01[:, 0] - ref_points_01) * self.scaling
        # ref_point_subtracted_points_01_1 = (match_01[:, 1] - ref_points_01) * self.scaling
        #
        # ref_point_subtracted_points_12_0 = (match_12[:, 0] - ref_points_12) * self.scaling
        # ref_point_subtracted_points_12_1 = (match_12[:, 1] - ref_points_12) * self.scaling

        magnitude_01 = np.linalg.norm(ref_vec_subtracted_vecs_01, axis=1)
        angle_01 = np.arctan2(ref_vec_subtracted_vecs_01[:, 1], ref_vec_subtracted_vecs_01[:, 0])
        # get angle in degrees between 0 and 180
        angle_01 = np.abs(angle_01) * 180 / np.pi
        angle_01 = np.where(angle_01 > 180, 360 - angle_01, angle_01)

        magnitude_12 = np.linalg.norm(ref_vec_subtracted_vecs_12, axis=1)
        angle_12 = np.arctan2(ref_vec_subtracted_vecs_12[:, 1], ref_vec_subtracted_vecs_12[:, 0])
        # get angle in degrees between 0 and 180
        angle_12 = np.abs(angle_12) * 180 / np.pi
        angle_12 = np.where(angle_12 > 180, 360 - angle_12, angle_12)

        # todo now can extract other features

        # # # tracks are backward coords, label coords, and forward coords
        # match = np.stack([coords_0, coords_1, coords_2], axis=1)
        # properties = {'magnitude': [], 'angle': []}
        # tracks = []
        # skip_num = 10
        # for track_num, track in enumerate(match[::skip_num]):
        #     if np.any(np.isnan(track)):
        #         continue
        #     tracks.append([track_num, 0, *track[0]])
        #     tracks.append([track_num, 1, *track[1]])
        #     tracks.append([track_num, 2, *track[2]])
        #     properties['magnitude'].append(0)
        #     properties['magnitude'].append(magnitude_01[track_num*skip_num])
        #     properties['magnitude'].append(magnitude_12[track_num*skip_num])
        #     properties['angle'].append(0)
        #     properties['angle'].append(angle_01[track_num*skip_num])
        #     properties['angle'].append(angle_12[track_num*skip_num])
        # viewer.add_tracks(tracks, properties=properties)
        # viewer.add_points(match_01[:, 0][idxmin_01], size=1, face_color='red')
        # viewer.add_points(match_01[:, 1][idxmin_01], size=1, face_color='blue')

    def _run_coord_movement_analysis(self):
        for t in range(1, self.num_t-1):
            self._run_frame(t)
        print('hi')

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_coord_movement_analysis()

if __name__ == "__main__":
    tif_file = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(tif_file)
    run_obj = CoordMovement(im_info, num_t=3)
    run_obj.run()
