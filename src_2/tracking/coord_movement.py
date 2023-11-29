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

        ref_points = match[ref_vec_idxs]
        ref_vecs = match_vec[ref_vec_idxs]
        # anywhere where match[:, 0] is nan, ref_vecs and ref_points is nan
        ref_vecs[np.any(np.isnan(match[:, 0]), axis=1)] = np.nan
        ref_points[np.any(np.isnan(match[:, 0]), axis=1), ...] = np.nan
        ref_vecs[np.any(np.isnan(match[:, 1]), axis=1)] = np.nan
        ref_points[np.any(np.isnan(match[:, 1]), axis=1), ...] = np.nan

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

    def _get_degrees_law_of_cos(self, a, b, c, deg=True):
        # to get the angle of b
        # cos(B) = (a^2 + c^2 - b^2) / 2ac
        B = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
        # replace nan with 0
        B[np.isnan(B)] = 0
        if not deg:
            return B
        # convert to degrees between 0 and 180
        B = np.abs(B) * 180 / np.pi
        B = np.where(B > 180, 360 - B, B)
        return B

    def _get_ref_based_features(self, label_vals, coords_0, coords_1, coords_2, vec01_scaled, vec12_scaled):
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
        reference_point_speed_01 = np.linalg.norm((ref_coords_all_01[1] - ref_coords_all_01[0]) * self.scaling, axis=1)
        new_12 = (ref_coords_all_12[0] - dif_vec, ref_coords_all_12[1] - dif_vec)
        reference_point_speed_12 = np.linalg.norm((new_12[1] - new_12[0]) * self.scaling, axis=1)
        b = np.linalg.norm((new_12[1] - ref_coords_all_01[0]) * self.scaling, axis=1)

        ref_travel_angle_change = self._get_degrees_law_of_cos(reference_point_speed_12, b, reference_point_speed_01)

        ref_vecs_01, ref_points_01 = self._get_reference_vector(match_01, idxmin_01, vec01_scaled, label_vals)
        ref_vecs_12, ref_points_12 = self._get_reference_vector(match_12, idxmin_12, vec12_scaled, label_vals)

        dereffed_vec01 = vec01_scaled - ref_vecs_01
        dereffed_vec12 = vec12_scaled - ref_vecs_12

        coord_speed_01 = np.linalg.norm(dereffed_vec01, axis=1)
        coord_speed_12 = np.linalg.norm(dereffed_vec12, axis=1)
        coord_acceleration = coord_speed_12 - coord_speed_01

        points_2_no_ref = coords_2 * self.scaling - ref_vecs_12 - ref_vecs_01
        points_1_no_ref = coords_1 * self.scaling - ref_vecs_01
        points_0_no_ref = coords_0 * self.scaling

        coord_travel_angle_change = self._get_degrees_law_of_cos(np.linalg.norm(points_1_no_ref - points_0_no_ref, axis=1),
                                                              np.linalg.norm(points_2_no_ref - points_0_no_ref, axis=1),
                                                              np.linalg.norm(points_2_no_ref - points_1_no_ref, axis=1))

        ref_acceleration = reference_point_speed_12 - reference_point_speed_01

        # a = (coords_0 - ref_points_01[:, 0]) * self.scaling
        # c = (coords_1 - coords_0) * self.scaling - ref_vecs_01
        # b = (coords_1 - ref_points_01[:, 1]) * self.scaling - ref_vecs_01
        # momentum_angle = self._get_degrees_law_of_cos(
        #     np.linalg.norm(a, axis=1),
        #     np.linalg.norm(b, axis=1),
        #     np.linalg.norm(c, axis=1),
        #     deg=False
        # )
        # p = np.linalg.norm(ref_vecs_01, axis=1) * np.cos(momentum_angle)
        # r_vec = (coords_0 - ref_points_01[:, 0]) * self.scaling
        # angular_momentum = np.cross(r_vec, p)

    def _get_nonref_features(self, label_vals, coords_0, coords_1, coords_2, vec01_scaled, vec12_scaled):
        coord_speed_01 = np.linalg.norm(vec01_scaled, axis=1)
        coord_speed_12 = np.linalg.norm(vec12_scaled, axis=1)
        coord_acceleration = coord_speed_12 - coord_speed_01

        a = np.linalg.norm(vec01_scaled, axis=1)
        c = np.linalg.norm(vec12_scaled, axis=1)
        b = np.linalg.norm((coords_2 - coords_0) * self.scaling, axis=1)

        coord_travel_angle_change = self._get_degrees_law_of_cos(a, b, c)

    def _get_angular_velocity(self, r0, r1):
        ang_disp_um = np.divide(np.cross(r0, r1, axis=1).T, (np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1))).T

        ang_vel_um_s = ang_disp_um / self.im_info.dim_sizes['T']

        ang_vel_magnitude = np.linalg.norm(ang_vel_um_s, axis=1)

        ang_vel_orientation = (ang_vel_um_s.T / ang_vel_magnitude).T
        ang_vel_orientation = np.where(np.isnan(ang_vel_orientation), ang_vel_um_s, ang_vel_orientation)
        ang_vel_orientation = np.where(np.isinf(ang_vel_orientation), ang_vel_um_s, ang_vel_orientation)

        return ang_vel_um_s, ang_vel_magnitude, ang_vel_orientation

    def _get_linear_velocity(self, r0, r1):
        lin_disp_um = r1 - r0

        lin_vel_um_s = lin_disp_um / self.im_info.dim_sizes['T']

        lin_vel_magnitude = np.linalg.norm(lin_vel_um_s, axis=1)

        lin_vel_orientation = (lin_vel_um_s.T / lin_vel_magnitude).T
        lin_vel_orientation = np.where(np.isnan(lin_vel_orientation), lin_vel_um_s, lin_vel_orientation)
        lin_vel_orientation = np.where(np.isinf(lin_vel_orientation), lin_vel_um_s, lin_vel_orientation)

        return lin_vel_um_s, lin_vel_magnitude, lin_vel_orientation

    def _get_features(self, label_vals, coords_0, coords_1, coords_2):
        # todo, should also include a way to specify a reference point
        vec01 = coords_1 - coords_0
        vec12 = coords_2 - coords_1
        vec01_scaled = vec01 * self.scaling
        vec12_scaled = vec12 * self.scaling

        idxmin_01 = self._get_min_euc_dist(label_vals, vec01_scaled)
        idxmin_12 = self._get_min_euc_dist(label_vals, vec12_scaled)

        ref_coords_all_01 = self._get_ref_coords(np.stack([coords_0, coords_1], axis=1), idxmin_01, label_vals)
        ref_coords_all_12 = self._get_ref_coords(np.stack([coords_1, coords_2], axis=1), idxmin_12, label_vals)

        ref_coords_um_01 = (ref_coords_all_01[0] * self.scaling, ref_coords_all_01[1] * self.scaling)
        ref_coords_um_12 = (ref_coords_all_12[0] * self.scaling, ref_coords_all_12[1] * self.scaling)

        pos_coords_um_0 = coords_0 * self.scaling
        pos_coords_um_1 = coords_1 * self.scaling
        pos_coords_um_2 = coords_2 * self.scaling

        com_coords_um_0 = np.nanmean(pos_coords_um_0, axis=0)
        com_coords_um_1 = np.nanmean(pos_coords_um_1, axis=0)
        com_coords_um_2 = np.nanmean(pos_coords_um_2, axis=0)

        r0_com_rel = pos_coords_um_0 - com_coords_um_0
        r1_com_rel = pos_coords_um_1 - com_coords_um_1
        r2_com_rel = pos_coords_um_2 - com_coords_um_2

        com_ang_vel_vec_01, com_ang_vel_mag_01, com_ang_vel_ori_01 = self._get_angular_velocity(r0_com_rel, r1_com_rel)
        com_ang_vel_vec_12, com_ang_vel_mag_12, com_ang_vel_ori_12 = self._get_angular_velocity(r1_com_rel, r2_com_rel)

        com_ang_acc_vec = (com_ang_vel_vec_12 - com_ang_vel_vec_01) / self.im_info.dim_sizes['T']
        com_ang_acc_mag = np.linalg.norm(com_ang_acc_vec, axis=1)
        com_ang_ori_change = (com_ang_vel_ori_12 - com_ang_vel_ori_01) / self.im_info.dim_sizes['T']
        com_ang_acc_ori = com_ang_acc_vec / com_ang_acc_mag[:, None]

        com_lin_vel_vec_01, com_lin_vel_mag_01, com_lin_vel_ori_01 = self._get_linear_velocity(r0_com_rel, r1_com_rel)
        com_lin_vel_vec_12, com_lin_vel_mag_12, com_lin_vel_ori_12 = self._get_linear_velocity(r1_com_rel, r2_com_rel)

        com_lin_acc_vec = (com_lin_vel_vec_12 - com_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        com_lin_acc_mag = np.linalg.norm(com_lin_acc_vec, axis=1)
        com_lin_ori_change = (com_lin_vel_ori_12 - com_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        com_lin_acc_ori = com_lin_acc_vec / com_lin_acc_mag[:, None]

        r0_com_rel_mag = np.linalg.norm(r0_com_rel, axis=1)
        r1_com_rel_mag = np.linalg.norm(r1_com_rel, axis=1)
        r2_com_rel_mag = np.linalg.norm(r2_com_rel, axis=1)

        com_directionality_01 = (r1_com_rel_mag - r0_com_rel_mag) / self.im_info.dim_sizes['T']
        com_directionality_12 = (r2_com_rel_mag - r1_com_rel_mag) / self.im_info.dim_sizes['T']

        com_directionality_acceleration = (com_directionality_12 - com_directionality_01) / self.im_info.dim_sizes['T']

        # com_unit_vec_01 = r0_com_rel / np.linalg.norm(r0_com_rel, axis=1)[:, None]
        # com_unit_vec_01[np.isnan(com_unit_vec_01)] = 0
        #
        # com_unit_vec_12 = r1_com_rel / np.linalg.norm(r1_com_rel, axis=1)[:, None]
        # com_unit_vec_12[np.isnan(com_unit_vec_12)] = 0
        #
        # com_directionality_01 = np.dot(com_lin_vel_vec_01, com_unit_vec_01, axis=1)  # pos is antero, neg is retro
        # com_directionality_12 = np.dot(com_lin_vel_vec_12, com_unit_vec_12, axis=1)
        #
        # com_directionality_acceleration = com_directionality_12 - com_directionality_01

        com_lin_acc_vec = (com_lin_vel_vec_12 - com_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        com_lin_acc_mag = np.linalg.norm(com_lin_acc_vec, axis=1)
        com_lin_acc_ori = com_lin_acc_vec / com_lin_acc_mag[:, None]

        r0_rel_01 = pos_coords_um_0 - ref_coords_um_01[0]
        r1_rel_01 = pos_coords_um_1 - ref_coords_um_01[1]

        r1_rel_12 = pos_coords_um_1 - ref_coords_um_12[0]
        r2_rel_12 = pos_coords_um_2 - ref_coords_um_12[1]

        rel_ang_vel_vec_01, rel_ang_vel_mag_01, rel_ang_vel_ori_01 = self._get_angular_velocity(r0_rel_01, r1_rel_01)
        rel_ang_vel_vec_12, rel_ang_vel_mag_12, rel_ang_vel_ori_12 = self._get_angular_velocity(r1_rel_12, r2_rel_12)

        rel_ang_acc_vec = (rel_ang_vel_vec_12 - rel_ang_vel_vec_01) / self.im_info.dim_sizes['T']
        rel_ang_acc_mag = np.linalg.norm(rel_ang_acc_vec, axis=1)
        rel_ang_ori_change = (rel_ang_vel_ori_12 - rel_ang_vel_ori_01) / self.im_info.dim_sizes['T']
        rel_ang_acc_ori = rel_ang_acc_vec / rel_ang_acc_mag[:, None]

        rel_lin_vel_vec_01, rel_lin_vel_mag_01, rel_lin_vel_ori_01 = self._get_linear_velocity(r0_rel_01, r1_rel_01)
        rel_lin_vel_vec_12, rel_lin_vel_mag_12, rel_lin_vel_ori_12 = self._get_linear_velocity(r1_rel_12, r2_rel_12)

        rel_lin_acc_vec = (rel_lin_vel_vec_12 - rel_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        rel_lin_acc_mag = np.linalg.norm(rel_lin_acc_vec, axis=1)
        rel_lin_ori_change = (rel_lin_vel_ori_12 - rel_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        rel_lin_acc_ori = rel_lin_acc_vec / rel_lin_acc_mag[:, None]

        lin_vel_vec_01, lin_vel_mag_01, lin_vel_ori_01 = self._get_linear_velocity(pos_coords_um_0, pos_coords_um_1)
        lin_vel_vec_12, lin_vel_mag_12, lin_vel_ori_12 = self._get_linear_velocity(pos_coords_um_1, pos_coords_um_2)

        lin_acc_vec = (lin_vel_vec_12 - lin_vel_vec_01) / self.im_info.dim_sizes['T']
        lin_acc_mag = np.linalg.norm(lin_acc_vec, axis=1)
        lin_ori_change = (lin_vel_ori_12 - lin_vel_ori_01) / self.im_info.dim_sizes['T']
        lin_acc_ori = lin_acc_vec / lin_acc_mag[:, None]

        ref_lin_vel_vec_01, ref_lin_vel_mag_01, ref_lin_vel_ori_01 = self._get_linear_velocity(ref_coords_um_01[0], ref_coords_um_01[1])
        ref_lin_vel_vec_12, ref_lin_vel_mag_12, ref_lin_vel_ori_12 = self._get_linear_velocity(ref_coords_um_12[0], ref_coords_um_12[1])

        ref_lin_acc_vec = (ref_lin_vel_vec_12 - ref_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        ref_lin_acc_mag = np.linalg.norm(ref_lin_acc_vec, axis=1)
        ref_lin_ori_change = (ref_lin_vel_ori_12 - ref_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        ref_lin_acc_ori = ref_lin_acc_vec / ref_lin_acc_mag[:, None]
        print('hi')
        # todo get the average orientation of all the angular velocity vectors of a label
        #  then get the angle (using dot product) between each point's orientation and the average orientation
        #  will give the rotational alignment of the label
        # todo do the same for linear velocity

    def _run_frame(self, t):
        label_vals = self.label_memmap[t][self.label_memmap[t]>0]
        coords_1 = np.argwhere(self.label_memmap[t]>0).astype('float32')

        vec12 = self.flow_interpolator_fw.interpolate_coord(coords_1, t)
        vec01 = self.flow_interpolator_bw.interpolate_coord(coords_1, t)

        coords_0 = coords_1 - vec01
        coords_2 = coords_1 + vec12
        self._get_features(label_vals, coords_0, coords_1, coords_2)

        # vec12_scaled = vec12 * self.scaling
        # vec01_scaled = vec01 * self.scaling
        #
        # self._get_ref_based_features(label_vals, coords_0, coords_1, coords_2, vec01_scaled, vec12_scaled)

        # todo now can extract other features

        # # tracks are backward coords, label coords, and forward coords
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(self.label_memmap)
        #
        # tracks_ref = []
        # running_track_num = 0
        # skip_num = 1
        # for track_num, (label, track) in enumerate(ref_coords_01[::skip_num]):
        #     if np.any(np.isnan(track)):
        #         continue
        #     tracks_ref.append([running_track_num, 0, *track[0]])
        #     tracks_ref.append([running_track_num, 1, *track[1]])
        #     running_track_num += 1
        # # viewer.add_tracks(tracks_ref)
        # for track_num, (label, track) in enumerate(ref_coords_12[::skip_num]):
        #     if np.any(np.isnan(track)):
        #         continue
        #     tracks_ref.append([running_track_num, 1, *track[0]])
        #     tracks_ref.append([running_track_num, 2, *track[1]])
        #     running_track_num += 1
        # viewer.add_tracks(tracks_ref)
        # print('hi')
        #
        # # idx where idxmin_01 index is 2056
        # # idx_test = np.argwhere(idxmin_01.index.values == 2056)
        #
        # match = np.stack([coords_0, coords_1, coords_2], axis=1)
        # properties = {'speed': [], 'angle': [], 'acceleration': [], 'angular_momentum': [], 'travel_angle_diff': [],
        #               'speed_ref': [], 'acceleration_ref': [], 'angular_momentum_ref': []}
        # tracks = []
        # skip_num = 20
        # for track_num, track in enumerate(match[::skip_num]):
        #     if np.any(np.isnan(track)):
        #         continue
        #     tracks.append([track_num, 0, *track[0]])
        #     tracks.append([track_num, 1, *track[1]])
        #     tracks.append([track_num, 2, *track[2]])
        #     properties['speed'].append(0)
        #     properties['speed'].append(speed_01[track_num*skip_num])
        #     properties['speed'].append(speed_12[track_num*skip_num])
        #     properties['angle'].append(0)
        #     properties['angle'].append(travel_angles_01[track_num*skip_num])
        #     properties['angle'].append(travel_angles_12[track_num*skip_num])
        #     properties['acceleration'].append(acceleration[track_num*skip_num])
        #     properties['acceleration'].append(acceleration[track_num*skip_num])
        #     properties['acceleration'].append(acceleration[track_num*skip_num])
        #     properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
        #     properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
        #     properties['angular_momentum'].append(angular_momentum[track_num*skip_num])
        #     properties['travel_angle_diff'].append(travel_angle_diff[track_num*skip_num])
        #     properties['travel_angle_diff'].append(travel_angle_diff[track_num*skip_num])
        #     properties['travel_angle_diff'].append(travel_angle_diff[track_num*skip_num])
        #     properties['speed_ref'].append(0)
        #     properties['speed_ref'].append(reference_point_speed_01[track_num*skip_num])
        #     properties['speed_ref'].append(reference_point_speed_12[track_num*skip_num])
        #     properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
        #     properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
        #     properties['acceleration_ref'].append(reference_acceleration[track_num*skip_num])
        #     properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
        #     properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
        #     properties['angular_momentum_ref'].append(angular_momentum_ref[track_num*skip_num])
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
