from src import logger
from src.im_info.im_info import ImInfo
from src.tracking.flow_interpolation import FlowInterpolator
from src.utils.general import get_reshaped_image
import numpy as np
import pandas as pd


class CoordMovement:
    def __init__(self, im_info: ImInfo, num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        # todo, could just extract velocity stats with 2..
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        if self.num_t < 3:
            raise ValueError('num_t must be at least 3')

        if self.im_info.no_z:
            self.scaling = (im_info.dim_sizes['Y'], im_info.dim_sizes['X'])
        else:
            self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        self.voxel_matches_path = None
        self.skel_label_memmap = None
        self.label_memmap = None

        self.organelle_feature_df = None
        self.branch_feature_df = None

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

        skel_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.skel_label_memmap = get_reshaped_image(skel_label_memmap, self.num_t, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        raw_im = self.im_info.get_im_memmap(self.im_info.im_path)
        self.raw_im = get_reshaped_image(raw_im, self.num_t, self.im_info)


        self.organelle_features_path = self.im_info.pipeline_paths['organelle_motility_features']
        self.branch_features_path = self.im_info.pipeline_paths['branch_motility_features']

        self.shape = self.skel_label_memmap.shape

        rel_ang_vel_mag_12_im = self.im_info.pipeline_paths['rel_ang_vel_mag_12']
        self.rel_ang_vel_mag_12_im = self.im_info.allocate_memory(rel_ang_vel_mag_12_im, shape=self.shape, dtype='double',
                                                          description='rel_ang_vel_mag_12_im im',
                                                          return_memmap=True)

        rel_lin_vel_mag_12_im = self.im_info.pipeline_paths['rel_lin_vel_mag_12']
        self.rel_lin_vel_mag_12_im = self.im_info.allocate_memory(rel_lin_vel_mag_12_im, shape=self.shape, dtype='double',
                                                          description='rel_lin_vel_mag_12_im im',
                                                          return_memmap=True)

        rel_ang_acc_mag_im = self.im_info.pipeline_paths['rel_ang_acc_mag']
        self.rel_ang_acc_mag_im = self.im_info.allocate_memory(rel_ang_acc_mag_im, shape=self.shape, dtype='double',
                                                          description='rel_ang_acc_mag_im im',
                                                          return_memmap=True)

        rel_lin_acc_mag_im = self.im_info.pipeline_paths['rel_lin_acc_mag']
        self.rel_lin_acc_mag_im = self.im_info.allocate_memory(rel_lin_acc_mag_im, shape=self.shape, dtype='double',
                                                          description='rel_lin_acc_mag_im im',
                                                          return_memmap=True)

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

    def _get_ref_coords(self, match, idxmin, labels):
        idxmin_0 = idxmin.index.values
        idxmin_1 = idxmin.values

        # get the index of idxmin_0 that matches each item in labels
        idxmin_0_idx = np.searchsorted(idxmin_0, labels)
        bad_idxs = np.argwhere(idxmin_0_idx == len(idxmin_0))
        idxmin_0_idx[bad_idxs] = 0
        ref_vec_idxs = idxmin_1[idxmin_0_idx].astype('int32')

        coords_0_all = match[:, 0][ref_vec_idxs]
        coords_1_all = match[:, 1][ref_vec_idxs]

        coords_0_all[np.isnan(match[:, 0])] = np.nan
        coords_1_all[np.isnan(match[:, 0])] = np.nan
        coords_0_all[np.isnan(match[:, 1])] = np.nan
        coords_1_all[np.isnan(match[:, 1])] = np.nan
        coords_0_all[bad_idxs] = np.nan
        coords_1_all[bad_idxs] = np.nan
        return coords_0_all, coords_1_all

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

    def _get_voxel_features(self, label_vals, coords_0, coords_1, coords_2):
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

        self.feature_df['com_ang_vel_mag_01'] = com_ang_vel_mag_01
        self.feature_df['com_ang_vel_mag_12'] = com_ang_vel_mag_12
        self.feature_df['com_ang_acc_mag'] = com_ang_acc_mag

        com_lin_vel_vec_01, com_lin_vel_mag_01, com_lin_vel_ori_01 = self._get_linear_velocity(r0_com_rel, r1_com_rel)
        com_lin_vel_vec_12, com_lin_vel_mag_12, com_lin_vel_ori_12 = self._get_linear_velocity(r1_com_rel, r2_com_rel)

        com_lin_acc_vec = (com_lin_vel_vec_12 - com_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        com_lin_acc_mag = np.linalg.norm(com_lin_acc_vec, axis=1)
        com_lin_ori_change = (com_lin_vel_ori_12 - com_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        com_lin_acc_ori = com_lin_acc_vec / com_lin_acc_mag[:, None]

        self.feature_df['com_lin_vel_mag_01'] = com_lin_vel_mag_01
        self.feature_df['com_lin_vel_mag_12'] = com_lin_vel_mag_12
        self.feature_df['com_lin_acc_mag'] = com_lin_acc_mag

        r0_com_rel_mag = np.linalg.norm(r0_com_rel, axis=1)
        r1_com_rel_mag = np.linalg.norm(r1_com_rel, axis=1)
        r2_com_rel_mag = np.linalg.norm(r2_com_rel, axis=1)

        com_directionality_01 = (r1_com_rel_mag - r0_com_rel_mag) / self.im_info.dim_sizes['T']
        com_directionality_12 = (r2_com_rel_mag - r1_com_rel_mag) / self.im_info.dim_sizes['T']

        com_directionality_acceleration = (com_directionality_12 - com_directionality_01) / self.im_info.dim_sizes['T']

        self.feature_df['com_directionality_01'] = com_directionality_01
        self.feature_df['com_directionality_12'] = com_directionality_12
        self.feature_df['com_directionality_acceleration'] = com_directionality_acceleration

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

        self.feature_df['rel_ang_vel_mag_01'] = rel_ang_vel_mag_01
        self.feature_df['rel_ang_vel_mag_12'] = rel_ang_vel_mag_12
        self.feature_df['rel_ang_acc_mag'] = rel_ang_acc_mag

        rel_lin_vel_vec_01, rel_lin_vel_mag_01, rel_lin_vel_ori_01 = self._get_linear_velocity(r0_rel_01, r1_rel_01)
        rel_lin_vel_vec_12, rel_lin_vel_mag_12, rel_lin_vel_ori_12 = self._get_linear_velocity(r1_rel_12, r2_rel_12)

        rel_lin_acc_vec = (rel_lin_vel_vec_12 - rel_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        rel_lin_acc_mag = np.linalg.norm(rel_lin_acc_vec, axis=1)
        rel_lin_ori_change = (rel_lin_vel_ori_12 - rel_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        rel_lin_acc_ori = rel_lin_acc_vec / rel_lin_acc_mag[:, None]

        self.feature_df['rel_lin_vel_mag_01'] = rel_lin_vel_mag_01
        self.feature_df['rel_lin_vel_mag_12'] = rel_lin_vel_mag_12
        self.feature_df['rel_lin_acc_mag'] = rel_lin_acc_mag

        lin_vel_vec_01, lin_vel_mag_01, lin_vel_ori_01 = self._get_linear_velocity(pos_coords_um_0, pos_coords_um_1)
        lin_vel_vec_12, lin_vel_mag_12, lin_vel_ori_12 = self._get_linear_velocity(pos_coords_um_1, pos_coords_um_2)

        lin_acc_vec = (lin_vel_vec_12 - lin_vel_vec_01) / self.im_info.dim_sizes['T']
        lin_acc_mag = np.linalg.norm(lin_acc_vec, axis=1)
        lin_ori_change = (lin_vel_ori_12 - lin_vel_ori_01) / self.im_info.dim_sizes['T']
        lin_acc_ori = lin_acc_vec / lin_acc_mag[:, None]

        self.feature_df['lin_vel_mag_01'] = lin_vel_mag_01
        self.feature_df['lin_vel_mag_12'] = lin_vel_mag_12
        self.feature_df['lin_acc_mag'] = lin_acc_mag

        ref_lin_vel_vec_01, ref_lin_vel_mag_01, ref_lin_vel_ori_01 = self._get_linear_velocity(ref_coords_um_01[0], ref_coords_um_01[1])
        ref_lin_vel_vec_12, ref_lin_vel_mag_12, ref_lin_vel_ori_12 = self._get_linear_velocity(ref_coords_um_12[0], ref_coords_um_12[1])

        ref_lin_acc_vec = (ref_lin_vel_vec_12 - ref_lin_vel_vec_01) / self.im_info.dim_sizes['T']
        ref_lin_acc_mag = np.linalg.norm(ref_lin_acc_vec, axis=1)
        ref_lin_ori_change = (ref_lin_vel_ori_12 - ref_lin_vel_ori_01) / self.im_info.dim_sizes['T']
        ref_lin_acc_ori = ref_lin_acc_vec / ref_lin_acc_mag[:, None]

        self.feature_df['ref_lin_vel_mag_01'] = ref_lin_vel_mag_01
        self.feature_df['ref_lin_vel_mag_12'] = ref_lin_vel_mag_12
        self.feature_df['ref_lin_acc_mag'] = ref_lin_acc_mag

        # todo get the average orientation of all the angular velocity vectors of a label
        #  then get the angle (using dot product) between each point's orientation and the average orientation
        #  will give the rotational alignment of the label
        # todo do the same for linear velocity

        # todo for every voxel, make a mask of all flow vectors that are within X pixels of it
        #  then, we can look at, for example, % of vectors going towards that voxel, rel to its movement (convergence)
        #  or % of vectors going away from that voxel, relative to its movement (divergence)
        #  or alignment of vectors in its vicinity, to look at turbidity
        #  other useful metrics? vorticity?

    def _get_df_stats(self, group_df):
        mean_df = group_df.mean()
        mean_df.columns = [f'{col}_mean' for col in mean_df.columns]
        std_df = group_df.std()
        std_df.columns = [f'{col}_std' for col in std_df.columns]
        max_df = group_df.max()
        max_df.columns = [f'{col}_max' for col in max_df.columns]
        min_df = group_df.min()
        min_df.columns = [f'{col}_min' for col in min_df.columns]
        median_df = group_df.median()
        median_df.columns = [f'{col}_median' for col in median_df.columns]
        q_25_df = group_df.quantile(0.25)
        q_25_df.columns = [f'{col}_q_25' for col in q_25_df.columns]
        q_75_df = group_df.quantile(0.75)
        q_75_df.columns = [f'{col}_q_75' for col in q_75_df.columns]

        main_label_df = pd.concat([mean_df, std_df, max_df, min_df, median_df, q_25_df, q_75_df], axis=1)
        main_label_df = main_label_df.reset_index()

        return main_label_df

    def _get_label_features(self, t):
        # drop na
        copy_df = self.feature_df.copy()

        # remove the label and t columns
        # main_copy_df = copy_df.dropna()
        main_copy_df = copy_df.drop(columns=['label', 't'])
        # group self.df_features by 'main_label'
        group_df_main = main_copy_df.groupby('main_label')
        # drop 'label' column from group_df_main

        organelle_feature_df = self._get_df_stats(group_df_main)
        organelle_feature_df['t'] = t

        skel_copy_df = copy_df.copy()
        skel_copy_df = skel_copy_df.drop(columns=['main_label', 't'])
        group_df_skel = skel_copy_df.groupby('label')

        branch_feature_df = self._get_df_stats(group_df_skel)
        branch_feature_df['t'] = t

        return organelle_feature_df, branch_feature_df

    def _save_features(self):
        logger.debug('Saving spatial features.')
        self.organelle_feature_df.to_csv(self.organelle_features_path, index=False)
        self.branch_feature_df.to_csv(self.branch_features_path, index=False)

    def _run_frame(self, t):
        valid_pxs = self.skel_label_memmap[t] > 0
        skel_label_vals = self.skel_label_memmap[t][valid_pxs]
        obj_label_vals = self.label_memmap[t][valid_pxs]
        self.feature_df = pd.DataFrame({'t': t}, index=range(len(skel_label_vals)))
        self.feature_df['label'] = skel_label_vals
        self.feature_df['main_label'] = obj_label_vals
        coords_1 = np.argwhere(self.skel_label_memmap[t] > 0).astype('float32')

        vec12 = self.flow_interpolator_fw.interpolate_coord(coords_1, t)
        vec01 = self.flow_interpolator_bw.interpolate_coord(coords_1, t)

        coords_0 = coords_1 - vec01
        coords_2 = coords_1 + vec12
        self._get_voxel_features(skel_label_vals, coords_0, coords_1, coords_2)
        # rel_ang_vel_im = np.zeros(self.skel_label_memmap[t].shape, dtype='float32')
        self.rel_ang_vel_mag_12_im[t][valid_pxs] = self.feature_df['rel_ang_vel_mag_01'].values
        self.rel_lin_vel_mag_12_im[t][valid_pxs] = self.feature_df['rel_lin_vel_mag_12'].values
        self.rel_ang_acc_mag_im[t][valid_pxs] = self.feature_df['rel_ang_acc_mag'].values
        self.rel_lin_acc_mag_im[t][valid_pxs] = self.feature_df['rel_lin_acc_mag'].values
        organelle_feature_df, branch_feature_df = self._get_label_features(t)
        if self.organelle_feature_df is None:
            self.organelle_feature_df = organelle_feature_df
        else:
            self.organelle_feature_df = pd.concat([self.organelle_feature_df, organelle_feature_df], axis=0)
        if self.branch_feature_df is None:
            self.branch_feature_df = branch_feature_df
        else:
            self.branch_feature_df = pd.concat([self.branch_feature_df, branch_feature_df], axis=0)


        # vec12_scaled = vec12 * self.scaling
        # vec01_scaled = vec01 * self.scaling
        #
        # self._get_ref_based_features(label_vals, coords_0, coords_1, coords_2, vec01_scaled, vec12_scaled)

        # todo now can extract other features

        # # tracks are backward coords, label coords, and forward coords
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(self.skel_label_memmap)
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
        # self.feature_df.lin_vel_mag_01[np.isnan(self.feature_df.lin_vel_mag_01)] = 0
        # self.feature_df.lin_vel_mag_12[np.isnan(self.feature_df.lin_vel_mag_12)] = 0
        # self.feature_df.lin_acc_mag[np.isnan(self.feature_df.lin_acc_mag)] = 0
        # self.feature_df.rel_lin_vel_mag_01[np.isnan(self.feature_df.rel_lin_vel_mag_01)] = 0
        # self.feature_df.rel_lin_vel_mag_12[np.isnan(self.feature_df.rel_lin_vel_mag_12)] = 0
        # self.feature_df.rel_ang_vel_mag_01[np.isnan(self.feature_df.rel_ang_vel_mag_01)] = 0
        # self.feature_df.rel_ang_vel_mag_12[np.isnan(self.feature_df.rel_ang_vel_mag_12)] = 0
        # self.feature_df.com_lin_vel_mag_01[np.isnan(self.feature_df.com_lin_vel_mag_01)] = 0
        # self.feature_df.com_lin_vel_mag_12[np.isnan(self.feature_df.com_lin_vel_mag_12)] = 0
        # self.feature_df.com_ang_vel_mag_01[np.isnan(self.feature_df.com_ang_vel_mag_01)] = 0
        # self.feature_df.com_ang_vel_mag_12[np.isnan(self.feature_df.com_ang_vel_mag_12)] = 0
        # self.feature_df.com_directionality_01[np.isnan(self.feature_df.com_directionality_01)] = 0
        # self.feature_df.com_directionality_12[np.isnan(self.feature_df.com_directionality_12)] = 0

        # import napari
        # viewer = napari.Viewer()
        #
        #
        # viewer.add_image(self.raw_im, scale=[1.71, 1, 1])
        # match = np.stack([coords_0, coords_1, coords_2], axis=1)
        # properties = {'lin_vel': [], 'lin_acc': [],
        #               'rel_lin_vel': [], 'rel_ang_vel': [],
        #               'com_lin_vel': [], 'com_ang_vel': [], 'com_directionality': [],
        #               'ref_lin_vel': []}
        # tracks = []
        # skip_num = 1
        # for track_num, track in enumerate(match[::skip_num]):
        #     if np.any(np.isnan(track)):
        #         continue
        #     tracks.append([track_num, 0, *track[0]])
        #     tracks.append([track_num, 1, *track[1]])
        #     tracks.append([track_num, 2, *track[2]])
        #     properties['lin_vel'].append(0)
        #     properties['lin_vel'].append(self.feature_df.lin_vel_mag_01[track_num*skip_num])
        #     properties['lin_vel'].append(self.feature_df.lin_vel_mag_12[track_num*skip_num])
        #     properties['lin_acc'].append(0)
        #     properties['lin_acc'].append(self.feature_df.lin_acc_mag[track_num*skip_num])
        #     properties['lin_acc'].append(self.feature_df.lin_acc_mag[track_num*skip_num])
        #     properties['rel_lin_vel'].append(0)
        #     properties['rel_lin_vel'].append(self.feature_df.rel_lin_vel_mag_01[track_num*skip_num])
        #     properties['rel_lin_vel'].append(self.feature_df.rel_lin_vel_mag_12[track_num*skip_num])
        #     properties['rel_ang_vel'].append(0)
        #     properties['rel_ang_vel'].append(self.feature_df.rel_ang_vel_mag_01[track_num*skip_num])
        #     properties['rel_ang_vel'].append(self.feature_df.rel_ang_vel_mag_12[track_num*skip_num])
        #     properties['com_lin_vel'].append(0)
        #     properties['com_lin_vel'].append(self.feature_df.com_lin_vel_mag_01[track_num*skip_num])
        #     properties['com_lin_vel'].append(self.feature_df.com_lin_vel_mag_12[track_num*skip_num])
        #     properties['com_ang_vel'].append(0)
        #     properties['com_ang_vel'].append(self.feature_df.com_ang_vel_mag_01[track_num*skip_num])
        #     properties['com_ang_vel'].append(self.feature_df.com_ang_vel_mag_12[track_num*skip_num])
        #     properties['com_directionality'].append(0)
        #     properties['com_directionality'].append(self.feature_df.com_directionality_01[track_num*skip_num])
        #     properties['com_directionality'].append(self.feature_df.com_directionality_12[track_num*skip_num])
        #     properties['ref_lin_vel'].append(0)
        #     properties['ref_lin_vel'].append(self.feature_df.ref_lin_vel_mag_01[track_num*skip_num])
        #     properties['ref_lin_vel'].append(self.feature_df.ref_lin_vel_mag_12[track_num*skip_num])
        # viewer.add_tracks(tracks, properties=properties, scale=[1.71, 1, 1])
        # viewer.add_tracks(tracks, scale=[1.71, 1, 1])
        # viewer.add_points(match_01[:, 0][idxmin_01], size=1, face_color='red')
        # viewer.add_points(match_01[:, 1][idxmin_01], size=1, face_color='blue')

    def _run_coord_movement_analysis(self):
        for t in range(1, self.num_t-1):
            print(f'Processing {t} of {self.num_t-1}')
            self._run_frame(t)
        self._save_features()


    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_coord_movement_analysis()

if __name__ == "__main__":
    tif_file = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(tif_file)
    run_obj = CoordMovement(im_info, num_t=3)
    run_obj.run()
