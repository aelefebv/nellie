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
        # #get indices of all nan vals
        # nan_idx = np.argwhere(np.isnan(df['euc_dist']))
        # remove nan values
        df = df[~np.isnan(df['euc_dist'])]

        # todo all the labels that don't have a corresponding df index are being wonky

        idxmin = df.groupby('labels')['euc_dist'].idxmin()
        return idxmin

    def _get_reference_vector(self, match, idxmin, match_vec, labels):
        idxmin_0 = idxmin.index.values
        idxmin_1 = idxmin.values

        # get the index of idxmin_0 that matches each item in labels
        idxmin_0_idx = np.searchsorted(idxmin_0, labels)
        ref_vec_idxs = idxmin_1[idxmin_0_idx].astype('int32')
        # ref_vec_idxs = np.where(np.any(np.isnan(match[:, 0]), axis=1), np.nan, ref_vec_idxs)

        ref_points = match[:, 0][ref_vec_idxs]
        ref_vecs = match_vec[ref_vec_idxs]
        # anywhere where match[:, 0] is nan, ref_vecs and ref_points is nan
        ref_vecs[np.isnan(match[:, 0])] = np.nan
        ref_points[np.isnan(match[:, 0])] = np.nan
        ref_vecs[np.isnan(match[:, 1])] = np.nan
        ref_points[np.isnan(match[:, 1])] = np.nan

        return ref_vecs, ref_points

    def _get_ref_coords(self, match, idxmin, labels):
        idxmin_0 = idxmin.index.values
        idxmin_1 = idxmin.values

        # get the index of idxmin_0 that matches each item in labels
        idxmin_0_idx = np.searchsorted(idxmin_0, labels)
        ref_vec_idxs = idxmin_1[idxmin_0_idx].astype('int32')

        coords_0_all = match[:, 0][ref_vec_idxs]
        coords_1_all = match[:, 1][ref_vec_idxs]

        coords_0_all[np.isnan(match[:, 0])] = np.nan
        coords_1_all[np.isnan(match[:, 0])] = np.nan
        coords_0_all[np.isnan(match[:, 1])] = np.nan
        coords_1_all[np.isnan(match[:, 1])] = np.nan
        return coords_0_all, coords_1_all


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
        ref_coords_01 = list(zip(idxmin_01.index.values, match_01[idxmin_01]))
        ref_coords_12 = list(zip(idxmin_12.index.values, match_12[idxmin_12]))

        ref_coords_all_01 = self._get_ref_coords(match_01, idxmin_01, label_vals)
        ref_coords_all_12 = self._get_ref_coords(match_12, idxmin_12, label_vals)

        # here, speed is the magnitude of the reference point vector
        dif_vec = ref_coords_all_12[0] - ref_coords_all_01[1]
        c = np.linalg.norm((ref_coords_all_01[1] - ref_coords_all_01[0]) * self.scaling, axis=1)
        new_12 = (ref_coords_all_12[0] - dif_vec, ref_coords_all_12[1] - dif_vec)
        a = np.linalg.norm((new_12[1] - new_12[0]) * self.scaling, axis=1)
        b = np.linalg.norm((new_12[1] - ref_coords_all_01[0]) * self.scaling, axis=1)

        # law of cosines to find angle for B
        # cos(B) = (a^2 + c^2 - b^2) / 2ac
        # B = arccos((a^2 + c^2 - b^2) / 2ac)
        angular_momentum_ref = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
        # replace nan with 0
        angular_momentum_ref[np.isnan(angular_momentum_ref)] = 0

        # in degrees between 0 and 180
        # angular_momentum_ref = np.abs(angular_momentum_ref) * 180 / np.pi
        # angular_momentum_ref = np.where(angular_momentum_ref > 180, 360 - angular_momentum_ref, angular_momentum_ref)

        reference_point_speed_01 = np.linalg.norm(ref_coords_all_01[0] * self.scaling -
                                                      ref_coords_all_01[1] * self.scaling, axis=1)
        reference_point_speed_12 = np.linalg.norm(ref_coords_all_12[0] * self.scaling -
                                                      ref_coords_all_12[1] * self.scaling, axis=1)

        reference_point_angle_01 = np.arctan2(ref_coords_all_01[0][:, 1] - ref_coords_all_01[1][:, 1],
                                                  ref_coords_all_01[0][:, 0] - ref_coords_all_01[1][:, 0])
        reference_point_angle_12 = np.arctan2(ref_coords_all_12[0][:, 1] - ref_coords_all_12[1][:, 1],
                                                  ref_coords_all_12[0][:, 0] - ref_coords_all_12[1][:, 0])

        ref_vecs_01, ref_points_01 = self._get_reference_vector(match_01, idxmin_01, vec01_scaled, label_vals)
        ref_vecs_12, ref_points_12 = self._get_reference_vector(match_12, idxmin_12, vec12_scaled, label_vals)

        ref_vec_subtracted_vecs_01 = vec01_scaled - ref_vecs_01
        ref_vec_subtracted_vecs_12 = vec12_scaled - ref_vecs_12
        # ref_vec_subtracted_vecs_12 = vec12_scaled - ref_vecs_01

        # ref_point_subtracted_points_01_0 = (match_01[:, 0] - ref_points_01) * self.scaling
        # ref_point_subtracted_points_01_1 = (match_01[:, 1] - ref_points_01) * self.scaling
        #
        # ref_point_subtracted_points_12_0 = (match_12[:, 0] - ref_points_12) * self.scaling
        # ref_point_subtracted_points_12_1 = (match_12[:, 1] - ref_points_12) * self.scaling

        # speed here is the magnitude of the vector minus the reference vector, to subtract out local movement
        speed_01 = np.linalg.norm(ref_vec_subtracted_vecs_01, axis=1)
        angle_01 = np.arctan2(ref_vec_subtracted_vecs_01[:, 1], ref_vec_subtracted_vecs_01[:, 0])
        # get angle in degrees between 0 and 180
        angle_01 = np.abs(angle_01) * 180 / np.pi
        angle_01 = np.where(angle_01 > 180, 360 - angle_01, angle_01)

        speed_12 = np.linalg.norm(ref_vec_subtracted_vecs_12, axis=1)
        angle_12 = np.arctan2(ref_vec_subtracted_vecs_12[:, 1], ref_vec_subtracted_vecs_12[:, 0])
        # get angle in degrees between 0 and 180
        angle_12 = np.abs(angle_12) * 180 / np.pi
        angle_12 = np.where(angle_12 > 180, 360 - angle_12, angle_12)

        reference_acceleration = reference_point_speed_12 - reference_point_speed_01
        reference_angular_momentum = np.abs(reference_point_angle_12 - reference_point_angle_01)
        # degrees between 0 and 180
        reference_angular_momentum = np.abs(reference_angular_momentum) * 180 / np.pi
        reference_angular_momentum = np.where(reference_angular_momentum > 180, 360 - reference_angular_momentum,
                                                reference_angular_momentum)


        acceleration = speed_12 - speed_01
        angular_momentum = np.abs(angle_12 - angle_01)

        ref_point_matches = np.stack([ref_points_01, ref_points_12], axis=1)
        # only keep unique matches
        ref_point_matches = np.unique(ref_point_matches, axis=0)
        ref_point_magnitude = np.linalg.norm(ref_point_matches[:, 1] - ref_point_matches[:, 0], axis=1)
        ref_point_angle = np.arctan2(ref_point_matches[:, 1, 1] - ref_point_matches[:, 0, 1],
                                        ref_point_matches[:, 1, 0] - ref_point_matches[:, 0, 0])
        # get angle in degrees between 0 and 180
        ref_point_angle = np.abs(ref_point_angle) * 180 / np.pi
        ref_point_angle = np.where(ref_point_angle > 180, 360 - ref_point_angle, ref_point_angle)

        # todo now can extract other features

        # # # tracks are backward coords, label coords, and forward coords
        import napari
        viewer = napari.Viewer()
        viewer.add_image(self.label_memmap)

        tracks_ref = []
        running_track_num = 0
        skip_num = 1
        for track_num, (label, track) in enumerate(ref_coords_01[::skip_num]):
            if np.any(np.isnan(track)):
                continue
            tracks_ref.append([running_track_num, 0, *track[0]])
            tracks_ref.append([running_track_num, 1, *track[1]])
            running_track_num += 1
        # viewer.add_tracks(tracks_ref)
        for track_num, (label, track) in enumerate(ref_coords_12[::skip_num]):
            if np.any(np.isnan(track)):
                continue
            tracks_ref.append([running_track_num, 1, *track[0]])
            tracks_ref.append([running_track_num, 2, *track[1]])
            running_track_num += 1
        viewer.add_tracks(tracks_ref)
        print('hi')

        # idx where idxmin_01 index is 2056
        # idx_test = np.argwhere(idxmin_01.index.values == 2056)

        match = np.stack([coords_0, coords_1, coords_2], axis=1)
        properties = {'speed': [], 'angle': [], 'acceleration': [], 'angular_momentum': [], 'speed_ref': [],
                        'angle_ref': [], 'acceleration_ref': [], 'angular_momentum_ref': []}
        tracks = []
        skip_num = 1
        for track_num, track in enumerate(match[:10000]):#[::skip_num]):
            if np.any(np.isnan(track)):
                continue
            tracks.append([track_num, 0, *track[0]])
            tracks.append([track_num, 1, *track[1]])
            tracks.append([track_num, 2, *track[2]])
            properties['speed'].append(0)
            properties['speed'].append(speed_01[track_num*skip_num])
            properties['speed'].append(speed_12[track_num*skip_num])
            properties['angle'].append(0)
            properties['angle'].append(angle_01[track_num*skip_num])
            properties['angle'].append(angle_12[track_num*skip_num])
            properties['acceleration'].append(acceleration[track_num*skip_num])
            properties['acceleration'].append(acceleration[track_num*skip_num])
            properties['acceleration'].append(acceleration[track_num*skip_num])
            properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
            properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
            properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
            properties['speed_ref'].append(0)
            properties['speed_ref'].append(reference_point_speed_01[track_num*skip_num])
            properties['speed_ref'].append(reference_point_speed_12[track_num*skip_num])
            properties['angle_ref'].append(0)
            properties['angle_ref'].append(reference_point_angle_01[track_num*skip_num])
            properties['angle_ref'].append(reference_point_angle_12[track_num*skip_num])
            properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
            properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
            properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
            properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
            properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
            properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
        viewer.add_tracks(tracks, properties=properties)
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
