import pickle

import numpy as np
from scipy import spatial
from skimage.measure import regionprops

from nellie import logger
from nellie.im_info.im_info import ImInfo
from nellie.tracking.flow_interpolation import FlowInterpolator
from nellie.utils.general import get_reshaped_image
import pandas as pd
import time


class Hierarchy:
    def __init__(self, im_info: ImInfo, num_t: int):
        self.im_info = im_info
        self.num_t = num_t
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_raw = None
        self.im_struct = None
        self.im_distance = None
        self.im_skel = None
        self.im_pixel_class = None
        self.label_components = None
        self.label_branches = None
        self.im_border_mask = None

        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        self.shape = None

        self.voxels = None
        self.nodes = None
        self.branches = None
        self.components = None
        self.image = None

    def _get_t(self):
        if self.num_t is None and not self.im_info.no_t:
            # if self.im_info.no_t:
            #     raise ValueError("No time dimension in image.")
            self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        # if self.num_t < 3:
        #     raise ValueError("num_t must be at least 3")
        return self.num_t

    def _allocate_memory(self):
        im_raw = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_raw = get_reshaped_image(im_raw, self.num_t, self.im_info)
        im_struct = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_struct = get_reshaped_image(im_struct, self.num_t, self.im_info)
        label_components = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_components = get_reshaped_image(label_components, self.num_t, self.im_info)
        label_branches = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.label_branches = get_reshaped_image(label_branches, self.num_t, self.im_info)
        im_skel = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.im_skel = get_reshaped_image(im_skel, self.num_t, self.im_info)
        im_pixel_class = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.im_pixel_class = get_reshaped_image(im_pixel_class, self.num_t, self.im_info)
        im_distance = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance = get_reshaped_image(im_distance, self.num_t, self.im_info)
        im_border_mask = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_border'])
        self.im_border_mask = get_reshaped_image(im_border_mask, self.num_t, self.im_info)
        if not self.im_info.no_t:
            im_obj_reassigned = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_obj_label_reassigned'])
            self.im_obj_reassigned = get_reshaped_image(im_obj_reassigned, self.num_t, self.im_info)
            im_branch_reassigned = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_branch_label_reassigned'])
            self.im_branch_reassigned = get_reshaped_image(im_branch_reassigned, self.num_t, self.im_info)

        self.shape = self.im_raw.shape
        # self.im_info.shape = self.shape

    def _get_hierarchies(self):
        self.voxels = Voxels(self)
        logger.info("Running voxel analysis")
        start = time.time()
        self.voxels.run()
        end = time.time()
        v_time = end - start

        self.nodes = Nodes(self)
        logger.info("Running node analysis")
        start = time.time()
        self.nodes.run()
        end = time.time()
        n_time = end - start

        self.branches = Branches(self)
        logger.info("Running branch analysis")
        start = time.time()
        self.branches.run()
        end = time.time()
        b_time = end - start

        self.components = Components(self)
        logger.info("Running component analysis")
        start = time.time()
        self.components.run()
        end = time.time()
        c_time = end - start

        self.image = Image(self)
        logger.info("Running image analysis")
        start = time.time()
        self.image.run()
        end = time.time()
        i_time = end - start

        logger.debug(f"Voxel analysis took {v_time} seconds")
        logger.debug(f"Node analysis took {n_time} seconds")
        logger.debug(f"Branch analysis took {b_time} seconds")
        logger.debug(f"Component analysis took {c_time} seconds")
        logger.debug(f"Image analysis took {i_time} seconds")

    def _save_dfs(self):
        voxel_features, voxel_headers = create_feature_array(self.voxels)
        voxel_df = pd.DataFrame(voxel_features, columns=voxel_headers)
        voxel_df.to_csv(self.im_info.pipeline_paths['features_voxels'], index=True)

        node_features, node_headers = create_feature_array(self.nodes)
        node_df = pd.DataFrame(node_features, columns=node_headers)
        node_df.to_csv(self.im_info.pipeline_paths['features_nodes'], index=True)

        branch_features, branch_headers = create_feature_array(self.branches, self.branches.branch_label)
        branch_df = pd.DataFrame(branch_features, columns=branch_headers)
        branch_df.to_csv(self.im_info.pipeline_paths['features_branches'], index=True)

        component_features, component_headers = create_feature_array(self.components, self.components.component_label)
        component_df = pd.DataFrame(component_features, columns=component_headers)
        component_df.to_csv(self.im_info.pipeline_paths['features_components'], index=True)

        image_features, image_headers = create_feature_array(self.image)
        image_df = pd.DataFrame(image_features, columns=image_headers)
        image_df.to_csv(self.im_info.pipeline_paths['features_image'], index=True)

    def _save_adjacency_maps(self):
        # edge list:
        v_b = [self.voxels.branch_labels[t][:, None] == self.branches.branch_label[t] for t in
               range(len(self.voxels.time))]
        v_n = []
        for t in range(len(self.voxels.time)):
            num_voxels = len(self.voxels.coords[t])
            num_nodes = len(self.nodes.nodes[t])
            v_n_temp = np.zeros((num_voxels, num_nodes), dtype=bool)
            for voxel, nodes in enumerate(self.voxels.node_labels[t]):
                v_n_temp[voxel, nodes] = True
            v_n.append(v_n_temp)
        v_o = [self.voxels.component_labels[t][:, None] == self.components.component_label[t] for t in
               range(len(self.voxels.time))]
        v_i = [np.ones((len(self.voxels.coords[t]), 1), dtype=bool) for t in range(len(self.voxels.time))]

        n_v = [v_n[t].T for t in range(len(self.voxels.time))]
        n_b = [self.nodes.branch_label[t][:, None] == self.branches.branch_label[t] for t in
               range(len(self.nodes.time))]
        n_o = [self.nodes.component_label[t][:, None] == self.components.component_label[t] for t in
               range(len(self.nodes.time))]
        n_i = [np.ones((len(self.nodes.nodes[t]), 1), dtype=bool) for t in range(len(self.nodes.time))]

        b_v = [v_b[t].T for t in range(len(self.voxels.time))]
        b_n = [n_b[t].T for t in range(len(self.nodes.time))]
        b_o = [self.branches.component_label[t][:, None] == self.components.component_label[t] for t in
               range(len(self.branches.time))]
        b_i = [np.ones((len(self.branches.branch_label[t]), 1), dtype=bool) for t in
               range(len(self.branches.time))]

        o_v = [v_o[t].T for t in range(len(self.voxels.time))]
        o_n = [n_o[t].T for t in range(len(self.nodes.time))]
        o_b = [b_o[t].T for t in range(len(self.branches.time))]
        o_i = [np.ones((len(self.components.component_label[t]), 1), dtype=bool) for t in
               range(len(self.components.time))]

        i_v = [v_i[t].T for t in range(len(self.voxels.time))]
        i_n = [n_i[t].T for t in range(len(self.nodes.time))]
        i_b = [b_i[t].T for t in range(len(self.branches.time))]
        i_o = [o_i[t].T for t in range(len(self.components.time))]

        # create a dict with all the edges
        # could also link voxels between t frames
        edges = {
            "v_b": v_b, "v_n": v_n, "v_o": v_o, "v_i": v_i,
            "n_v": n_v, "n_b": n_b, "n_o": n_o, "n_i": n_i,
            "b_v": b_v, "b_n": b_n, "b_o": b_o, "b_i": b_i,
            "o_v": o_v, "o_n": o_n, "o_b": o_b, "o_i": o_i,
            "i_v": i_v, "i_n": i_n, "i_b": i_b, "i_o": i_o,
        }
        # pickle and save
        with open(self.im_info.pipeline_paths['adjacency_maps'], "wb") as f:
            pickle.dump(edges, f)

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._get_hierarchies()
        print("Saving dataframes and adjacency maps")
        self._save_dfs()
        self._save_adjacency_maps()


def append_to_array(to_append):
    new_array = []
    new_headers = []
    for feature, stats in to_append.items():
        if type(stats) is not dict:
            stats = {'raw': stats}
        for stat, vals in stats.items():
            vals = np.array(vals)
            if len(vals.shape) > 1:
                for i in range(len(vals[0])):
                    new_array.append(vals[:, i])
                    new_headers.append(f'{feature}_{stat}_{i}')
            else:
                new_array.append(vals)
                new_headers.append(f'{feature}_{stat}')
    return new_array, new_headers


def create_feature_array(level, labels=None):
    full_array = None
    headers = None
    all_attr = []
    attr_dict = []
    if node_attr := getattr(level, 'aggregate_node_metrics', None):
        all_attr.append(node_attr)
    if voxel_attr := getattr(level, 'aggregate_voxel_metrics', None):
        all_attr.append(voxel_attr)
    if branch_attr := getattr(level, 'aggregate_branch_metrics', None):
        all_attr.append(branch_attr)
    if component_attr := getattr(level, 'aggregate_component_metrics', None):
        all_attr.append(component_attr)
    inherent_features = level.stats_to_aggregate
    for feature in inherent_features:
        if feature_vals := getattr(level, feature, None):
            all_attr.append([{feature: feature_vals[t]} for t in range(len(feature_vals))])

    for t in range(len(all_attr[0])):
        time_dict = {}
        for attr in all_attr:
            time_dict.update(attr[t])
        attr_dict.append(time_dict)

    for t in range(len(attr_dict)):
        to_append = attr_dict[t]
        time_array, new_headers = append_to_array(to_append)
        if labels is None:
            labels_t = range(len(time_array[0]))
        else:
            labels_t = labels[t]
        time_array.insert(0, labels_t)
        # append a list of t values to the start of time_array
        time_array.insert(0, np.array([t] * len(time_array[0])))
        if headers is None:
            headers = new_headers
        if full_array is None:
            full_array = np.array(time_array).T
        else:
            time_array = np.array(time_array).T
            full_array = np.vstack([full_array, time_array])

    headers.insert(0, 'label')
    headers.insert(0, 't')
    return full_array, headers


class Voxels:
    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.coords = []

        # add voxel metrics
        self.intensity = []
        self.structure = []

        self.vec01 = []
        self.vec12 = []

        self.ang_acc = []
        self.ang_acc_mag = []
        self.ang_vel_mag = []
        self.ang_vel_orient = []
        self.ang_vel = []
        self.lin_acc = []
        self.lin_acc_mag = []
        self.lin_vel_mag = []
        self.lin_vel_orient = []
        self.lin_vel = []

        self.ang_acc_rel = []
        self.ang_acc_rel_mag = []
        self.ang_vel_mag_rel = []
        self.ang_vel_orient_rel = []
        self.ang_vel_rel = []
        self.lin_acc_rel = []
        self.lin_acc_rel_mag = []
        self.lin_vel_mag_rel = []
        self.lin_vel_orient_rel = []
        self.lin_vel_rel = []
        self.directionality_rel = []
        self.directionality_acc_rel = []

        self.ang_acc_com = []
        self.ang_acc_com_mag = []
        self.ang_vel_mag_com = []
        self.ang_vel_orient_com = []
        self.ang_vel_com = []
        self.lin_acc_com = []
        self.lin_acc_com_mag = []
        self.lin_vel_mag_com = []
        self.lin_vel_orient_com = []
        self.lin_vel_com = []
        self.directionality_com = []
        self.directionality_acc_com = []

        self.node_labels = []
        self.branch_labels = []
        self.component_labels = []
        self.image_name = []

        self.node_dim0_lims = []
        self.node_dim1_lims = []
        self.node_dim2_lims = []
        self.node_voxel_idxs = []

        self.stats_to_aggregate = [
            "ang_acc", "ang_acc_com", "ang_acc_com_mag", "ang_acc_mag", "ang_acc_rel", "ang_acc_rel_mag",
            "ang_vel", "ang_vel_com", "ang_vel_mag", "ang_vel_mag_com", "ang_vel_mag_rel",
            "ang_vel_orient", "ang_vel_orient_com", "ang_vel_orient_rel", "ang_vel_rel",
            "directionality_acc_com", "directionality_acc_rel", "directionality_com", "directionality_rel",
            "lin_acc", "lin_acc_com", "lin_acc_com_mag", "lin_acc_mag", "lin_acc_rel", "lin_acc_rel_mag",
            "lin_vel", "lin_vel_com", "lin_vel_mag", "lin_vel_mag_rel", "lin_vel_orient", "lin_vel_orient_com",
            "lin_vel_orient_rel", "lin_vel_mag_com",
            "lin_vel_rel", "intensity", "structure"
        ]

        self.features_to_save = self.stats_to_aggregate

    def _get_node_info(self, t, frame_coords):
        # get all network pixels
        skeleton_pixels = np.argwhere(self.hierarchy.im_pixel_class[t] > 0)
        skeleton_radius = self.hierarchy.im_distance[t][tuple(skeleton_pixels.T)]

        # create bounding boxes of size largest_thickness around each skeleton pixel
        lims_dim0 = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 0, np.newaxis]).astype(int)
        lims_dim0[:, 1] += 1
        lims_dim1 = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 1, np.newaxis]).astype(int)
        lims_dim1[:, 1] += 1

        lims_dim0[lims_dim0 < 0] = 0
        lims_dim1[lims_dim1 < 0] = 0

        if not self.hierarchy.im_info.no_z:
            lims_dim2 = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 2, np.newaxis]).astype(
                int)
            lims_dim2[:, 1] += 1
            lims_dim2[lims_dim2 < 0] = 0
            max_dim0 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Z')]
            max_dim1 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Y')]
            max_dim2 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('X')]
            lims_dim2[lims_dim2 > max_dim2] = max_dim2
        else:
            max_dim0 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Y')]
            max_dim1 = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('X')]

        lims_dim0[lims_dim0 > max_dim0] = max_dim0
        lims_dim1[lims_dim1 > max_dim1] = max_dim1

        self.node_dim0_lims.append(lims_dim0)
        self.node_dim1_lims.append(lims_dim1)

        self.node_dim2_lims.append(lims_dim2) if not self.hierarchy.im_info.no_z else None

        frame_coord_nodes_idxs = []
        # if a frame coord is within the bounding box of a skeleton pixel, add the skeleton pixel's
        #  idx to the list for that frame coord
        for i, frame_coord in enumerate(frame_coords):
            if not self.hierarchy.im_info.no_z:
                dim0_coord, dim1_coord, dim2_coord = frame_coord
                dim2_mask = (lims_dim2[:, 0] <= dim2_coord) & (lims_dim2[:, 1] >= dim2_coord)
            else:
                dim0_coord, dim1_coord = frame_coord
            dim0_mask = (lims_dim0[:, 0] <= dim0_coord) & (lims_dim0[:, 1] >= dim0_coord)
            dim1_mask = (lims_dim1[:, 0] <= dim1_coord) & (lims_dim1[:, 1] >= dim1_coord)
            mask = dim0_mask & dim1_mask & dim2_mask if not self.hierarchy.im_info.no_z else dim0_mask & dim1_mask
            frame_coord_nodes_idxs.append(np.argwhere(mask).flatten())

        self.node_labels.append(frame_coord_nodes_idxs)

        # initialize an empty dictionary for frame_coord_nodes_idxs
        node_voxel_idxs = []

        for i, skeleton_px in enumerate(skeleton_pixels):
            if not self.hierarchy.im_info.no_z:
                bbox_voxels = np.argwhere(
                    (lims_dim0[i, 0] <= frame_coords[:, 0]) & (lims_dim0[i, 1] >= frame_coords[:, 0]) &
                    (lims_dim1[i, 0] <= frame_coords[:, 1]) & (lims_dim1[i, 1] >= frame_coords[:, 1]) &
                    (lims_dim2[i, 0] <= frame_coords[:, 2]) & (lims_dim2[i, 1] >= frame_coords[:, 2]))
            else:
                bbox_voxels = np.argwhere(
                    (lims_dim0[i, 0] <= frame_coords[:, 0]) & (lims_dim0[i, 1] >= frame_coords[:, 0]) &
                    (lims_dim1[i, 0] <= frame_coords[:, 1]) & (lims_dim1[i, 1] >= frame_coords[:, 1]))
            node_voxel_idxs.append(bbox_voxels.flatten())

        self.node_voxel_idxs.append(node_voxel_idxs)

    def _get_min_euc_dist(self, t, vec):
        euc_dist = np.linalg.norm(vec, axis=1)
        branch_labels = self.branch_labels[t]

        df = pd.DataFrame({'euc_dist': euc_dist, 'branch_label': branch_labels})
        df = df[~np.isnan(df['euc_dist'])]
        idxmin = df.groupby('branch_label')['euc_dist'].idxmin()
        # if there are no non-nan values in a branch, give idxmin at that branch index a value of nan
        missing_branches = np.setdiff1d(np.unique(branch_labels), df['branch_label'])
        for missing_branch in missing_branches:
            idxmin[missing_branch] = np.nan

        return idxmin

    def _get_ref_coords(self, coords_a, coords_b, idxmin, t):
        vals_a = idxmin[self.branch_labels[t]].values
        vals_a_no_nan = vals_a.copy()
        vals_a_no_nan[np.isnan(vals_a_no_nan)] = 0
        vals_a_no_nan = vals_a_no_nan.astype(int)

        vals_b = idxmin[self.branch_labels[t]].values
        vals_b_no_nan = vals_b.copy()
        vals_b_no_nan[np.isnan(vals_b_no_nan)] = 0
        vals_b_no_nan = vals_b_no_nan.astype(int)
        ref_a = coords_a[vals_a_no_nan]
        ref_b = coords_b[vals_b_no_nan]

        ref_a[np.isnan(vals_a)] = np.nan
        ref_b[np.isnan(vals_b)] = np.nan

        return ref_a, ref_b

    def _get_motility_stats(self, t, coords_1_px):
        coords_1_px = coords_1_px.astype('float32')
        if self.hierarchy.im_info.no_z:
            dims = 2
        else:
            dims = 3

        vec01 = []
        vec12 = []
        if t > 0:
            vec01_px = self.hierarchy.flow_interpolator_bw.interpolate_coord(coords_1_px, t)
            vec01 = vec01_px * self.hierarchy.spacing
            self.vec01.append(vec01)
        else:
            self.vec01.append(np.full((len(coords_1_px), dims), np.nan))

        if t < self.hierarchy.num_t - 1:
            vec12_px = self.hierarchy.flow_interpolator_fw.interpolate_coord(coords_1_px, t)
            vec12 = vec12_px * self.hierarchy.spacing
            self.vec12.append(vec12)
        else:
            self.vec12.append(np.full((len(coords_1_px), dims), np.nan))

        coords_1 = coords_1_px * self.hierarchy.spacing
        coords_com_1 = np.nanmean(coords_1, axis=0)
        r1_rel_com = coords_1 - coords_com_1
        r1_com_mag = np.linalg.norm(r1_rel_com, axis=1)

        if len(vec01) and len(vec12):
            coords_0_px = coords_1_px - vec01_px
            coords_0 = coords_0_px * self.hierarchy.spacing

            lin_vel_01, lin_vel_mag_01, lin_vel_orient_01 = self._get_linear_velocity(coords_0, coords_1)
            ang_vel_01, ang_vel_mag_01, ang_vel_orient_01 = self._get_angular_velocity(coords_0, coords_1)

            idxmin01 = self._get_min_euc_dist(t, vec01)
            ref_coords01 = self._get_ref_coords(coords_0, coords_1, idxmin01, t)
            ref_coords01[0][np.isnan(vec01)] = np.nan
            ref_coords01[1][np.isnan(vec01)] = np.nan
            r0_rel_01 = coords_0 - ref_coords01[0]
            r1_rel_01 = coords_1 - ref_coords01[1]

            lin_vel_rel_01, lin_vel_mag_rel_01, lin_vel_orient_rel_01 = self._get_linear_velocity(r0_rel_01, r1_rel_01)
            ang_vel_rel_01, ang_vel_mag_rel_01, ang_vel_orient_rel_01 = self._get_angular_velocity(r0_rel_01, r1_rel_01)

            coords_com_0 = np.nanmean(coords_0, axis=0)
            r0_rel_com = coords_0 - coords_com_0
            lin_vel_com_01, lin_vel_mag_com_01, lin_vel_orient_com_01 = self._get_linear_velocity(r0_rel_com,
                                                                                                  r1_rel_com)
            ang_vel_com_01, ang_vel_mag_com_01, ang_vel_orient_com_01 = self._get_angular_velocity(r0_rel_com,
                                                                                                   r1_rel_com)

            r0_com_mag = np.linalg.norm(r0_rel_com, axis=1)
            directionality_com_01 = np.abs(r0_com_mag - r1_com_mag) / (r0_com_mag + r1_com_mag)

            r0_rel_mag_01 = np.linalg.norm(r0_rel_01, axis=1)
            r1_rel_mag_01 = np.linalg.norm(r1_rel_01, axis=1)
            directionality_rel_01 = np.abs(r0_rel_mag_01 - r1_rel_mag_01) / (r0_rel_mag_01 + r1_rel_mag_01)

        if len(vec12):
            coords_2_px = coords_1_px + vec12_px
            coords_2 = coords_2_px * self.hierarchy.spacing

            lin_vel, lin_vel_mag, lin_vel_orient = self._get_linear_velocity(coords_1, coords_2)
            ang_vel, ang_vel_mag, ang_vel_orient = self._get_angular_velocity(coords_1, coords_2)

            idxmin12 = self._get_min_euc_dist(t, vec12)
            ref_coords12 = self._get_ref_coords(coords_1, coords_2, idxmin12, t)
            ref_coords12[0][np.isnan(vec12)] = np.nan
            ref_coords12[1][np.isnan(vec12)] = np.nan
            r1_rel_12 = coords_1 - ref_coords12[0]
            r2_rel_12 = coords_2 - ref_coords12[1]

            lin_vel_rel, lin_vel_mag_rel, lin_vel_orient_rel = self._get_linear_velocity(r1_rel_12, r2_rel_12)
            ang_vel_rel, ang_vel_mag_rel, ang_vel_orient_rel = self._get_angular_velocity(r1_rel_12, r2_rel_12)

            coords_com_2 = np.nanmean(coords_2, axis=0)
            r2_rel_com = coords_2 - coords_com_2
            lin_vel_com, lin_vel_mag_com, lin_vel_orient_com = self._get_linear_velocity(r1_rel_com, r2_rel_com)
            ang_vel_com, ang_vel_mag_com, ang_vel_orient_com = self._get_angular_velocity(r1_rel_com, r2_rel_com)

            r2_com_mag = np.linalg.norm(r2_rel_com, axis=1)
            directionality_com = np.abs(r2_com_mag - r1_com_mag) / (r2_com_mag + r1_com_mag)

            r2_rel_mag_12 = np.linalg.norm(r2_rel_12, axis=1)
            r1_rel_mag_12 = np.linalg.norm(r1_rel_12, axis=1)
            directionality_rel = np.abs(r2_rel_mag_12 - r1_rel_mag_12) / (r2_rel_mag_12 + r1_rel_mag_12)
        else:
            # vectors of nans
            lin_vel = np.full((len(coords_1), dims), np.nan)
            lin_vel_mag = np.full(len(coords_1), np.nan)
            lin_vel_orient = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag = np.full(len(coords_1), np.nan)
            lin_vel_rel = np.full((len(coords_1), dims), np.nan)
            lin_vel_mag_rel = np.full(len(coords_1), np.nan)
            lin_vel_orient_rel = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag_rel = np.full(len(coords_1), np.nan)
            lin_vel_com = np.full((len(coords_1), dims), np.nan)
            lin_vel_mag_com = np.full(len(coords_1), np.nan)
            lin_vel_orient_com = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag_com = np.full(len(coords_1), np.nan)
            directionality_com = np.full(len(coords_1), np.nan)
            directionality_rel = np.full(len(coords_1), np.nan)
            if dims == 3:
                ang_vel = np.full((len(coords_1), dims), np.nan)
                ang_vel_orient = np.full((len(coords_1), dims), np.nan)
                ang_vel_rel = np.full((len(coords_1), dims), np.nan)
                ang_vel_orient_rel = np.full((len(coords_1), dims), np.nan)
                ang_vel_com = np.full((len(coords_1), dims), np.nan)
                ang_vel_orient_com = np.full((len(coords_1), dims), np.nan)
            else:
                ang_vel = np.full(len(coords_1), np.nan)
                ang_vel_orient = np.full(len(coords_1), np.nan)
                ang_vel_rel = np.full(len(coords_1), np.nan)
                ang_vel_orient_rel = np.full(len(coords_1), np.nan)
                ang_vel_com = np.full(len(coords_1), np.nan)
                ang_vel_orient_com = np.full(len(coords_1), np.nan)
        self.lin_vel.append(lin_vel)
        self.lin_vel_mag.append(lin_vel_mag)
        self.lin_vel_orient.append(lin_vel_orient)
        self.ang_vel.append(ang_vel)
        self.ang_vel_mag.append(ang_vel_mag)
        self.ang_vel_orient.append(ang_vel_orient)
        self.lin_vel_rel.append(lin_vel_rel)
        self.lin_vel_mag_rel.append(lin_vel_mag_rel)
        self.lin_vel_orient_rel.append(lin_vel_orient_rel)
        self.ang_vel_rel.append(ang_vel_rel)
        self.ang_vel_mag_rel.append(ang_vel_mag_rel)
        self.ang_vel_orient_rel.append(ang_vel_orient_rel)
        self.lin_vel_com.append(lin_vel_com)
        self.lin_vel_mag_com.append(lin_vel_mag_com)
        self.lin_vel_orient_com.append(lin_vel_orient_com)
        self.ang_vel_com.append(ang_vel_com)
        self.ang_vel_mag_com.append(ang_vel_mag_com)
        self.ang_vel_orient_com.append(ang_vel_orient_com)
        self.directionality_com.append(directionality_com)
        self.directionality_rel.append(directionality_rel)

        if len(vec01) and len(vec12):
            lin_acc = (lin_vel - lin_vel_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_mag = np.linalg.norm(lin_acc, axis=1)
            ang_acc = (ang_vel - ang_vel_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_rel = (lin_vel_rel - lin_vel_rel_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_rel_mag = np.linalg.norm(lin_acc_rel, axis=1)
            ang_acc_rel = (ang_vel_rel - ang_vel_rel_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_com = (lin_vel_com - lin_vel_com_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_com_mag = np.linalg.norm(lin_acc_com, axis=1)
            ang_acc_com = (ang_vel_com - ang_vel_com_01) / self.hierarchy.im_info.dim_sizes['T']
            if self.hierarchy.im_info.no_z:
                ang_acc_mag = np.abs(ang_acc)
                ang_acc_rel_mag = np.abs(ang_acc_rel)
                ang_acc_com_mag = np.abs(ang_acc_com)
            else:
                ang_acc_mag = np.linalg.norm(ang_acc, axis=1)
                ang_acc_rel_mag = np.linalg.norm(ang_acc_rel, axis=1)
                ang_acc_com_mag = np.linalg.norm(ang_acc_com, axis=1)
            # directionality acceleration is the change of directionality based on
            #  directionality_com_01 and directionality_com
            directionality_acc_com = np.abs(directionality_com - directionality_com_01)
            directionality_acc_rel = np.abs(directionality_rel - directionality_rel_01)
        else:
            # vectors of nans
            lin_acc = np.full((len(coords_1), dims), np.nan)
            lin_acc_mag = np.full(len(coords_1), np.nan)
            ang_acc_mag = np.full(len(coords_1), np.nan)
            lin_acc_rel = np.full((len(coords_1), dims), np.nan)
            lin_acc_rel_mag = np.full(len(coords_1), np.nan)
            ang_acc_rel_mag = np.full(len(coords_1), np.nan)
            lin_acc_com = np.full((len(coords_1), dims), np.nan)
            lin_acc_com_mag = np.full(len(coords_1), np.nan)
            ang_acc_com_mag = np.full(len(coords_1), np.nan)
            directionality_acc_com = np.full(len(coords_1), np.nan)
            directionality_acc_rel = np.full(len(coords_1), np.nan)
            if dims == 3:
                ang_acc = np.full((len(coords_1), dims), np.nan)
                ang_acc_rel = np.full((len(coords_1), dims), np.nan)
                ang_acc_com = np.full((len(coords_1), dims), np.nan)
            else:
                ang_acc = np.full(len(coords_1), np.nan)
                ang_acc_rel = np.full(len(coords_1), np.nan)
                ang_acc_com = np.full(len(coords_1), np.nan)
        self.lin_acc.append(lin_acc)
        self.lin_acc_mag.append(lin_acc_mag)
        self.ang_acc.append(ang_acc)
        self.ang_acc_mag.append(ang_acc_mag)
        self.lin_acc_rel.append(lin_acc_rel)
        self.lin_acc_rel_mag.append(lin_acc_rel_mag)
        self.ang_acc_rel.append(ang_acc_rel)
        self.ang_acc_rel_mag.append(ang_acc_rel_mag)
        self.lin_acc_com.append(lin_acc_com)
        self.lin_acc_com_mag.append(lin_acc_com_mag)
        self.ang_acc_com.append(ang_acc_com)
        self.ang_acc_com_mag.append(ang_acc_com_mag)
        self.directionality_acc_com.append(directionality_acc_com)
        self.directionality_acc_rel.append(directionality_acc_rel)

    def _get_linear_velocity(self, ra, rb):
        lin_disp = rb - ra
        lin_vel = lin_disp / self.hierarchy.im_info.dim_sizes['T']
        lin_vel_mag = np.linalg.norm(lin_vel, axis=1)
        lin_vel_orient = (lin_vel.T / lin_vel_mag).T
        if self.hierarchy.im_info.no_z:
            lin_vel_orient = np.where(np.isinf(lin_vel_orient), [np.nan, np.nan], lin_vel_orient)
        else:
            lin_vel_orient = np.where(np.isinf(lin_vel_orient), [np.nan, np.nan, np.nan], lin_vel_orient)

        return lin_vel, lin_vel_mag, lin_vel_orient

    def _get_angular_velocity_2d(self, ra, rb):
        # calculate angles of ra and rb relative to x-axis
        theta_a = np.arctan2(ra[:, 1], ra[:, 0])
        theta_b = np.arctan2(rb[:, 1], rb[:, 0])

        # calculate the change in angle
        delta_theta = theta_b - theta_a

        # normalize the change in angle to be between -pi and pi
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

        # get the angular velocity
        ang_vel = delta_theta / self.hierarchy.im_info.dim_sizes['T']
        ang_vel_mag = np.abs(ang_vel)
        ang_vel_orient = np.sign(ang_vel)

        return ang_vel, ang_vel_mag, ang_vel_orient

    def _get_angular_velocity_3d(self, ra, rb):
        cross_product = np.cross(ra, rb, axis=1)
        norm = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
        ang_disp = np.divide(cross_product.T, norm.T).T
        ang_disp[norm == 0] = [np.nan, np.nan, np.nan]

        ang_vel = ang_disp / self.hierarchy.im_info.dim_sizes['T']
        ang_vel_mag = np.linalg.norm(ang_vel, axis=1)
        ang_vel_orient = (ang_vel.T / ang_vel_mag).T
        ang_vel_orient = np.where(np.isinf(ang_vel_orient), [np.nan, np.nan, np.nan], ang_vel_orient)

        return ang_vel, ang_vel_mag, ang_vel_orient

    def _get_angular_velocity(self, ra, rb):
        if self.hierarchy.im_info.no_z:
            return self._get_angular_velocity_2d(ra, rb)

        return self._get_angular_velocity_3d(ra, rb)

    def _run_frame(self, t=None):
        frame_coords = np.argwhere(self.hierarchy.label_components[t] > 0)
        self.coords.append(frame_coords)

        frame_component_labels = self.hierarchy.label_components[t][tuple(frame_coords.T)]
        self.component_labels.append(frame_component_labels)

        frame_branch_labels = self.hierarchy.label_branches[t][tuple(frame_coords.T)]
        self.branch_labels.append(frame_branch_labels)

        frame_intensity_vals = self.hierarchy.im_raw[t][tuple(frame_coords.T)]
        self.intensity.append(frame_intensity_vals)

        frame_structure_vals = self.hierarchy.im_struct[t][tuple(frame_coords.T)]
        self.structure.append(frame_structure_vals)

        frame_t = np.ones(frame_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        im_name = np.ones(frame_coords.shape[0], dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_node_info(t, frame_coords)
        self._get_motility_stats(t, frame_coords)

    def run(self):
        if self.hierarchy.num_t is None:
            self.hierarchy.num_t = 1
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


def aggregate_stats_for_class(child_class, t, list_of_idxs):
    # initialize a dictionary to hold lists of aggregated stats for each stat name
    aggregate_stats = {
        stat_name: {"mean": [], "std_dev": [], "25%": [], "50%": [], "75%": [], "min": [], "max": [], "range": [],
                    "sum": []} for
        stat_name in child_class.stats_to_aggregate if stat_name != 'reassigned_label'}

    for stat_name in child_class.stats_to_aggregate:
        if stat_name == 'reassigned_label':
            continue
        # access the relevant attribute for the current time frame
        stat_array = getattr(child_class, stat_name)[t]

        for idxs in list_of_idxs:
            stat_values = np.array(stat_array)[idxs]
            if stat_values.size == 0:
                # if there are no values for the current node, append a list of nans to each list in the aggregate_stats dictionary
                for key in aggregate_stats[stat_name].keys():
                    aggregate_stats[stat_name][key].append(np.nan)
                continue

            # calculate various statistics for the subset
            mean = np.nanmean(stat_values, axis=0)
            std_dev = np.nanstd(stat_values, axis=0)
            quartiles = np.nanquantile(stat_values, [0.25, 0.5, 0.75], axis=0)
            min_val = np.nanmin(stat_values, axis=0)
            max_val = np.nanmax(stat_values, axis=0)
            range_val = max_val - min_val
            sum_val = np.nansum(stat_values, axis=0)

            # append the calculated statistics to their respective lists in the aggregate_stats dictionary
            aggregate_stats[stat_name]["mean"].append(mean)
            aggregate_stats[stat_name]["std_dev"].append(std_dev)
            aggregate_stats[stat_name]["25%"].append(quartiles[0])
            aggregate_stats[stat_name]["50%"].append(quartiles[1])  # median
            aggregate_stats[stat_name]["75%"].append(quartiles[2])
            aggregate_stats[stat_name]["min"].append(min_val)
            aggregate_stats[stat_name]["max"].append(max_val)
            aggregate_stats[stat_name]["range"].append(range_val)
            aggregate_stats[stat_name]["sum"].append(sum_val)

    return aggregate_stats


class Nodes:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.nodes = []

        # add voxel aggregate metrics
        self.aggregate_voxel_metrics = []
        # add node metrics
        self.thickness = []
        self.divergence = []
        self.convergence = []
        self.vergere = []
        self.lin_magnitude_variability = []
        self.ang_magnitude_variability = []
        self.lin_direction_uniformity = []
        self.ang_direction_uniformity = []

        self.stats_to_aggregate = [
            "divergence", "convergence", "vergere", "lin_magnitude_variability", "ang_magnitude_variability",
            "lin_direction_uniformity", "ang_direction_uniformity", "thickness",
        ]

        self.features_to_save = self.stats_to_aggregate

        self.voxel_idxs = self.hierarchy.voxels.node_voxel_idxs
        self.branch_label = []
        self.component_label = []
        self.image_name = []

        self.node_z_lims = self.hierarchy.voxels.node_dim0_lims
        self.node_y_lims = self.hierarchy.voxels.node_dim1_lims
        self.node_x_lims = self.hierarchy.voxels.node_dim2_lims

    def _get_aggregate_voxel_stats(self, t):
        frame_agg = aggregate_stats_for_class(self.hierarchy.voxels, t, self.hierarchy.voxels.node_voxel_idxs[t])
        self.aggregate_voxel_metrics.append(frame_agg)

    def _get_node_stats(self, t):
        radius = distance_check(self.hierarchy.im_border_mask[t], self.nodes[t], self.hierarchy.spacing)
        self.thickness.append(radius * 2)

        divergence = []
        convergence = []
        vergere = []
        lin_mag_variability = []
        ang_mag_variability = []
        lin_dir_uniformity = []
        ang_dir_uniformity = []
        for i, node in enumerate(self.nodes[t]):
            vox_idxs = self.voxel_idxs[t][i]
            dist_vox_node = self.hierarchy.voxels.coords[t][vox_idxs] - self.nodes[t][i]
            dist_vox_node_mag = np.linalg.norm(dist_vox_node, axis=1, keepdims=True)
            dir_vox_node = dist_vox_node / dist_vox_node_mag

            dot_prod_01 = -np.nanmean(np.sum(-self.hierarchy.voxels.vec01[t][vox_idxs] * dir_vox_node, axis=1))
            convergence.append(dot_prod_01)

            dot_prod_12 = np.nanmean(np.sum(self.hierarchy.voxels.vec12[t][vox_idxs] * dir_vox_node, axis=1))
            divergence.append(dot_prod_12)

            vergere.append(dot_prod_01 + dot_prod_12)
            # high vergere is a funnel point (converges then diverges)
            # low vergere is a dispersal point (diverges then converges)

            lin_vel_mag = self.hierarchy.voxels.lin_vel_mag[t][vox_idxs]
            lin_mag_variability.append(np.nanstd(lin_vel_mag))
            ang_vel_mag = self.hierarchy.voxels.ang_vel_mag[t][vox_idxs]
            ang_mag_variability.append(np.nanstd(ang_vel_mag))

            lin_vel = self.hierarchy.voxels.lin_vel[t][vox_idxs]
            lin_unit_vec = lin_vel / lin_vel_mag[:, np.newaxis]
            lin_similarity_matrix = np.dot(lin_unit_vec, lin_unit_vec.T)
            np.fill_diagonal(lin_similarity_matrix, np.nan)
            lin_dir_uniformity.append(np.nanmean(lin_similarity_matrix))

            ang_vel = self.hierarchy.voxels.ang_vel[t][vox_idxs]
            ang_unit_vec = ang_vel / ang_vel_mag[:, np.newaxis]
            ang_similarity_matrix = np.dot(ang_unit_vec, ang_unit_vec.T)
            np.fill_diagonal(ang_similarity_matrix, np.nan)
            ang_dir_uniformity.append(np.nanmean(ang_similarity_matrix))
        self.divergence.append(divergence)
        self.convergence.append(convergence)
        self.vergere.append(vergere)
        self.lin_magnitude_variability.append(lin_mag_variability)
        self.ang_magnitude_variability.append(ang_mag_variability)
        self.lin_direction_uniformity.append(lin_dir_uniformity)
        self.ang_direction_uniformity.append(ang_dir_uniformity)

    def _run_frame(self, t):
        frame_skel_coords = np.argwhere(self.hierarchy.im_pixel_class[t] > 0)
        self.nodes.append(frame_skel_coords)

        frame_t = np.ones(frame_skel_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        frame_component_label = self.hierarchy.label_components[t][tuple(frame_skel_coords.T)]
        self.component_label.append(frame_component_label)

        frame_branch_label = self.hierarchy.label_branches[t][tuple(frame_skel_coords.T)]
        self.branch_label.append(frame_branch_label)

        im_name = np.ones(frame_skel_coords.shape[0], dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_aggregate_voxel_stats(t)
        self._get_node_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


def distance_check(border_mask, check_coords, spacing):
    border_coords = np.argwhere(border_mask) * spacing
    border_tree = spatial.cKDTree(border_coords)
    dist, _ = border_tree.query(check_coords * spacing, k=1)
    return dist


class Branches:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.branch_label = []

        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []
        # add branch metrics
        self.length = []
        self.thickness = []
        self.aspect_ratio = []
        self.tortuosity = []
        self.area = []
        self.axis_length_maj = []
        self.axis_length_min = []
        self.extent = []
        self.solidity = []
        self.reassigned_label = []

        self.branch_idxs = []
        self.component_label = []
        self.image_name = []

        self.stats_to_aggregate = [
            "length", "thickness", "aspect_ratio", "tortuosity", "area", "axis_length_maj", "axis_length_min",
            "extent", "solidity", "reassigned_label"
        ]

    def _get_aggregate_stats(self, t):
        voxel_labels = self.hierarchy.voxels.branch_labels[t]
        grouped_vox_idxs = [np.argwhere(voxel_labels == label).flatten()
                            for label in np.unique(voxel_labels) if label != 0]
        vox_agg = aggregate_stats_for_class(self.hierarchy.voxels, t, grouped_vox_idxs)
        self.aggregate_voxel_metrics.append(vox_agg)

        node_labels = self.hierarchy.nodes.branch_label[t]
        grouped_node_idxs = [np.argwhere(node_labels == label).flatten()
                             for label in np.unique(node_labels) if label != 0]
        node_agg = aggregate_stats_for_class(self.hierarchy.nodes, t, grouped_node_idxs)
        self.aggregate_node_metrics.append(node_agg)

    def _get_branch_stats(self, t):
        branch_idx_array_1 = np.array(self.branch_idxs[t])
        branch_idx_array_2 = np.array(self.branch_idxs[t])[:, None, :]
        dist = np.linalg.norm(branch_idx_array_1 - branch_idx_array_2, axis=-1)
        dist[dist >= 2] = 0  # remove any distances greater than adjacent pixel
        neighbors = np.sum(dist > 0, axis=1)
        tips = np.where(neighbors == 1)[0]
        lone_tips = np.where(neighbors == 0)[0]
        dist = np.triu(dist)

        neighbor_idxs = np.argwhere(dist > 0)

        # all coords idxs should be within 1 pixel of each other
        neighbor_coords_0 = self.branch_idxs[t][neighbor_idxs[:, 0]]
        neighbor_coords_1 = self.branch_idxs[t][neighbor_idxs[:, 1]]
        assert np.all(np.abs(neighbor_coords_0 - neighbor_coords_1) <= 1)

        # labels should be the exact same
        neighbor_labels_0 = self.hierarchy.im_skel[t][tuple(neighbor_coords_0.T)]
        neighbor_labels_1 = self.hierarchy.im_skel[t][tuple(neighbor_coords_1.T)]
        assert np.all(neighbor_labels_0 == neighbor_labels_1)

        scaled_coords_0 = neighbor_coords_0 * self.hierarchy.spacing
        scaled_coords_1 = neighbor_coords_1 * self.hierarchy.spacing
        distances = np.linalg.norm(scaled_coords_0 - scaled_coords_1, axis=1)
        unique_labels = np.unique(self.hierarchy.im_skel[t][self.hierarchy.im_skel[t] > 0])

        label_lengths = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            label_lengths[i] = np.sum(distances[neighbor_labels_0 == label])

        lone_tip_coords = self.branch_idxs[t][lone_tips]
        tip_coords = self.branch_idxs[t][tips]

        lone_tip_labels = self.hierarchy.im_skel[t][tuple(lone_tip_coords.T)]
        tip_labels = self.hierarchy.im_skel[t][tuple(tip_coords.T)]

        # find the distance between the two tips with the same label in tip_labels
        tip_distances = np.zeros(len(tip_labels))
        for i, label in enumerate(tip_labels):
            matched_idxs = tip_coords[tip_labels == label] * self.hierarchy.spacing
            tip_distances[i] = np.linalg.norm(matched_idxs[0] - matched_idxs[1])

        tortuosity = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            tip_dist = tip_distances[tip_labels == label]
            if len(tip_dist):
                tortuosity[i] = label_lengths[i] / tip_dist[0]
            else:
                tortuosity[i] = 1

        self.tortuosity.append(tortuosity)

        radii = distance_check(self.hierarchy.im_border_mask[t], self.branch_idxs[t], self.hierarchy.spacing)
        lone_tip_radii = radii[lone_tips]
        tip_radii = radii[tips]

        for label, radius in zip(lone_tip_labels, lone_tip_radii):
            label_lengths[unique_labels == label] += radius * 2
        for label, radius in zip(tip_labels, tip_radii):
            label_lengths[unique_labels == label] += radius

        # mean radii for each branch:
        median_thickenss = []
        thicknesses = radii * 2
        for label, thickness in zip(unique_labels, thicknesses):
            median_thickenss.append(np.median(thickness))

        # if thickness at an index is larger than the length, set it to the length, and length to thickness
        for i, thickness in enumerate(median_thickenss):
            if thickness > label_lengths[i]:
                median_thickenss[i] = label_lengths[i]
                label_lengths[i] = thickness

        aspect_ratios = label_lengths / median_thickenss

        self.aspect_ratio.append(aspect_ratios)
        self.thickness.append(median_thickenss)
        self.length.append(label_lengths)

        regions = regionprops(self.hierarchy.label_branches[t], spacing=self.hierarchy.spacing)
        areas = []
        axis_length_maj = []
        axis_length_min = []
        extent = []
        solidity = []
        reassigned_label = []

        for region in regions:
            if not self.hierarchy.im_info.no_t:
                region_reassigned_labels = self.hierarchy.im_branch_reassigned[t][tuple(region.coords.T)]
                # find which label is most common in the region via bin-counting
                reassigned_label.append(np.argmax(np.bincount(region_reassigned_labels)))
            else:
                reassigned_label.append(np.nan)
            areas.append(region.area)
            # due to bug in skimage (at the time of writing: https://github.com/scikit-image/scikit-image/issues/6630)
            try:
                maj_axis = region.major_axis_length
                min_axis = region.minor_axis_length
            except ValueError:
                maj_axis = np.nan
                min_axis = np.nan
            axis_length_maj.append(maj_axis)
            axis_length_min.append(min_axis)
            extent.append(region.extent)
            solidity.append(region.solidity)
        self.area.append(areas)
        self.axis_length_maj.append(axis_length_maj)
        self.axis_length_min.append(axis_length_min)
        self.extent.append(extent)
        self.solidity.append(solidity)
        self.reassigned_label.append(reassigned_label)

    def _run_frame(self, t):
        frame_branch_idxs = np.argwhere(self.hierarchy.im_skel[t] > 0)
        self.branch_idxs.append(frame_branch_idxs)

        frame_skel_branch_labels = self.hierarchy.im_skel[t][tuple(frame_branch_idxs.T)]

        smallest_label = int(np.min(self.hierarchy.im_skel[t][self.hierarchy.im_skel[t] > 0]))
        largest_label = int(np.max(self.hierarchy.im_skel[t]))
        frame_branch_labels = np.arange(smallest_label, largest_label + 1)
        num_branches = len(frame_branch_labels)

        frame_t = np.ones(num_branches, dtype=int) * t
        self.time.append(frame_t)

        # get the first voxel idx for each branch
        if self.hierarchy.im_info.no_z:
            frame_branch_coords = np.zeros((num_branches, 2), dtype=int)
        else:
            frame_branch_coords = np.zeros((num_branches, 3), dtype=int)
        for i in frame_branch_labels:
            branch_voxels = frame_branch_idxs[frame_skel_branch_labels == i]
            frame_branch_coords[i - 1] = branch_voxels[0]
        frame_component_label = self.hierarchy.label_components[t][tuple(frame_branch_coords.T)]
        self.component_label.append(frame_component_label)

        frame_branch_label = self.hierarchy.im_skel[t][tuple(frame_branch_coords.T)]
        self.branch_label.append(frame_branch_label)

        im_name = np.ones(num_branches, dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_aggregate_stats(t)
        self._get_branch_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


class Components:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.component_label = []
        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []
        self.aggregate_branch_metrics = []
        # add component metrics
        self.area = []
        self.axis_length_maj = []
        self.axis_length_min = []
        self.extent = []
        self.solidity = []
        self.reassigned_label = []

        self.image_name = []

        self.stats_to_aggregate = [
            "area", "axis_length_maj", "axis_length_min", "extent", "solidity", "reassigned_label",
        ]

    def _get_aggregate_stats(self, t):
        voxel_labels = self.hierarchy.voxels.component_labels[t]
        grouped_vox_idxs = [np.argwhere(voxel_labels == label).flatten() for label in np.unique(voxel_labels) if
                            label != 0]
        vox_agg = aggregate_stats_for_class(self.hierarchy.voxels, t, grouped_vox_idxs)
        self.aggregate_voxel_metrics.append(vox_agg)

        node_labels = self.hierarchy.nodes.component_label[t]
        grouped_node_idxs = [np.argwhere(node_labels == label).flatten() for label in np.unique(voxel_labels) if
                             label != 0]
        node_agg = aggregate_stats_for_class(self.hierarchy.nodes, t, grouped_node_idxs)
        self.aggregate_node_metrics.append(node_agg)

        branch_labels = self.hierarchy.branches.component_label[t]
        grouped_branch_idxs = [np.argwhere(branch_labels == label).flatten() for label in np.unique(voxel_labels) if
                               label != 0]
        branch_agg = aggregate_stats_for_class(self.hierarchy.branches, t, grouped_branch_idxs)
        self.aggregate_branch_metrics.append(branch_agg)

    def _get_component_stats(self, t):
        regions = regionprops(self.hierarchy.label_components[t], spacing=self.hierarchy.spacing)
        areas = []
        axis_length_maj = []
        axis_length_min = []
        extent = []
        solidity = []
        reassigned_label = []

        for region in regions:
            if not self.hierarchy.im_info.no_t:
                region_reassigned_labels = self.hierarchy.im_obj_reassigned[t][tuple(region.coords.T)]
                reassigned_label.append(np.argmax(np.bincount(region_reassigned_labels)))
            else:
                reassigned_label.append(np.nan)
            areas.append(region.area)
            try:
                maj_axis = region.major_axis_length
                min_axis = region.minor_axis_length
            except ValueError:
                maj_axis = np.nan
                min_axis = np.nan
            axis_length_maj.append(maj_axis)
            axis_length_min.append(min_axis)
            extent.append(region.extent)
            solidity.append(region.solidity)
        self.area.append(areas)
        self.axis_length_maj.append(axis_length_maj)
        self.axis_length_min.append(axis_length_min)
        self.extent.append(extent)
        self.solidity.append(solidity)
        self.reassigned_label.append(reassigned_label)

    def _run_frame(self, t):
        smallest_label = int(np.min(self.hierarchy.label_components[t][self.hierarchy.label_components[t] > 0]))
        largest_label = int(np.max(self.hierarchy.label_components[t]))
        frame_component_labels = np.arange(smallest_label, largest_label + 1)
        self.component_label.append(frame_component_labels)
        num_components = len(frame_component_labels)

        frame_t = np.ones(num_components, dtype=int) * t
        self.time.append(frame_t)

        im_name = np.ones(num_components, dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_aggregate_stats(t)
        self._get_component_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


class Image:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.image_name = []
        self.aggregate_voxel_metrics = []
        self.aggregate_node_metrics = []
        self.aggregate_branch_metrics = []
        self.aggregate_component_metrics = []
        self.stats_to_aggregate = []

    def _get_aggregate_stats(self, t):
        voxel_agg = aggregate_stats_for_class(self.hierarchy.voxels, t,
                                              [np.arange(len(self.hierarchy.voxels.coords[t]))])
        self.aggregate_voxel_metrics.append(voxel_agg)

        node_agg = aggregate_stats_for_class(self.hierarchy.nodes, t, [np.arange(len(self.hierarchy.nodes.nodes[t]))])
        self.aggregate_node_metrics.append(node_agg)

        branch_agg = aggregate_stats_for_class(self.hierarchy.branches, t,
                                               [self.hierarchy.branches.branch_label[t].flatten() - 1])
        self.aggregate_branch_metrics.append(branch_agg)

        component_agg = aggregate_stats_for_class(self.hierarchy.components, t,
                                                  [np.arange(len(self.hierarchy.components.component_label[t]))])
        self.aggregate_component_metrics.append(component_agg)

    def _run_frame(self, t):
        self.time.append(t)
        self.image_name.append(self.hierarchy.im_info.basename_no_ext)

        self._get_aggregate_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


if __name__ == "__main__":
    im_path = "/Users/austin/test_files/nelly/ND Stimulation Parallel 12.nd2"
    im_info = ImInfo(im_path)
    num_t = 3

    hierarchy = Hierarchy(im_info, num_t)
    hierarchy.run()
