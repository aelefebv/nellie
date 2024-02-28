import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import spatial

from src.im_info.im_info import ImInfo
from src.tracking.flow_interpolation import FlowInterpolator
from src.utils.general import get_reshaped_image


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

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                raise ValueError("No time dimension in image.")
            self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if self.num_t < 3:
            raise ValueError("num_t must be at least 3")
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
        im_distance = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance = get_reshaped_image(im_distance, self.num_t, self.im_info)
        im_skel = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.im_skel = get_reshaped_image(im_skel, self.num_t, self.im_info)
        im_pixel_class = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.im_pixel_class = get_reshaped_image(im_pixel_class, self.num_t, self.im_info)
        im_border_mask = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_border'])
        self.im_border_mask = get_reshaped_image(im_border_mask, self.num_t, self.im_info)

        self.shape = self.im_raw.shape

    def run(self):
        self._get_t()
        self._allocate_memory()


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

        self.node_z_lims = []
        self.node_y_lims = []
        self.node_x_lims = []
        self.node_voxel_idxs = []

    def _get_node_info(self, t, frame_coords):
        # get all network pixels
        skeleton_pixels = np.argwhere(self.hierarchy.im_pixel_class[t] > 0)
        skeleton_radius = self.hierarchy.im_distance[t][skeleton_pixels[:, 0], skeleton_pixels[:, 1], skeleton_pixels[:, 2]]
        largest_thickness = int(np.ceil(np.max(skeleton_radius * 2))) + 1
        # create bounding boxes of size largest_thickness around each skeleton pixel
        z_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 0, np.newaxis]).astype(int)
        z_lims[:, 1] += 1
        y_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 1, np.newaxis]).astype(int)
        y_lims[:, 1] += 1
        x_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 2, np.newaxis]).astype(int)
        x_lims[:, 1] += 1

        z_lims[z_lims < 0] = 0
        y_lims[y_lims < 0] = 0
        x_lims[x_lims < 0] = 0

        z_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Z')]
        y_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Y')]
        x_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('X')]

        z_lims[z_lims > z_max] = z_max
        y_lims[y_lims > y_max] = y_max
        x_lims[x_lims > x_max] = x_max

        self.node_z_lims.append(z_lims)
        self.node_y_lims.append(y_lims)
        self.node_x_lims.append(x_lims)

        frame_coord_nodes_idxs = []
        # if a frame coord is within the bounding box of a skeleton pixel, add the skeleton pixel's idx to the list for that frame coord
        for i, frame_coord in enumerate(frame_coords):
            z, y, x = frame_coord
            z_mask = (z_lims[:, 0] <= z) & (z_lims[:, 1] >= z)
            y_mask = (y_lims[:, 0] <= y) & (y_lims[:, 1] >= y)
            x_mask = (x_lims[:, 0] <= x) & (x_lims[:, 1] >= x)
            mask = z_mask & y_mask & x_mask
            frame_coord_nodes_idxs.append(np.argwhere(mask).flatten())

        self.node_labels.append(frame_coord_nodes_idxs)

        # Initialize an empty dictionary for frame_coord_nodes_idxs
        node_voxel_idxs = []

        for i, skeleton_px in enumerate(skeleton_pixels):
            bbox_voxels = np.argwhere((z_lims[i, 0] <= frame_coords[:, 0]) & (z_lims[i, 1] >= frame_coords[:, 0]) &
                                      (y_lims[i, 0] <= frame_coords[:, 1]) & (y_lims[i, 1] >= frame_coords[:, 1]) &
                                      (x_lims[i, 0] <= frame_coords[:, 2]) & (x_lims[i, 1] >= frame_coords[:, 2]))
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
            lin_vel_com_01, lin_vel_mag_com_01, lin_vel_orient_com_01 = self._get_linear_velocity(r0_rel_com, r1_rel_com)
            ang_vel_com_01, ang_vel_mag_com_01, ang_vel_orient_com_01 = self._get_angular_velocity(r0_rel_com, r1_rel_com)

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
            ang_vel = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag = np.full(len(coords_1), np.nan)
            ang_vel_orient = np.full((len(coords_1), dims), np.nan)
            lin_vel_rel = np.full((len(coords_1), dims), np.nan)
            lin_vel_mag_rel = np.full(len(coords_1), np.nan)
            lin_vel_orient_rel = np.full((len(coords_1), dims), np.nan)
            ang_vel_rel = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag_rel = np.full(len(coords_1), np.nan)
            ang_vel_orient_rel = np.full((len(coords_1), dims), np.nan)
            lin_vel_com = np.full((len(coords_1), dims), np.nan)
            lin_vel_mag_com = np.full(len(coords_1), np.nan)
            lin_vel_orient_com = np.full((len(coords_1), dims), np.nan)
            ang_vel_com = np.full((len(coords_1), dims), np.nan)
            ang_vel_mag_com = np.full(len(coords_1), np.nan)
            ang_vel_orient_com = np.full((len(coords_1), dims), np.nan)
            directionality_com = np.full(len(coords_1), np.nan)
            directionality_rel = np.full(len(coords_1), np.nan)
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
            ang_acc_mag = np.linalg.norm(ang_acc, axis=1)
            lin_acc_rel = (lin_vel_rel - lin_vel_rel_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_rel_mag = np.linalg.norm(lin_acc_rel, axis=1)
            ang_acc_rel = (ang_vel_rel - ang_vel_rel_01) / self.hierarchy.im_info.dim_sizes['T']
            ang_acc_rel_mag = np.linalg.norm(ang_acc_rel, axis=1)
            lin_acc_com = (lin_vel_com - lin_vel_com_01) / self.hierarchy.im_info.dim_sizes['T']
            lin_acc_com_mag = np.linalg.norm(lin_acc_com, axis=1)
            ang_acc_com = (ang_vel_com - ang_vel_com_01) / self.hierarchy.im_info.dim_sizes['T']
            ang_acc_com_mag = np.linalg.norm(ang_acc_com, axis=1)
            # directionality acceleration is the change of directionality based on directionality_com_01 and directionality_com
            directionality_acc_com = np.abs(directionality_com - directionality_com_01)
            directionality_acc_rel = np.abs(directionality_rel - directionality_rel_01)
        else:
            # vectors of nans
            lin_acc = np.full((len(coords_1), dims), np.nan)
            lin_acc_mag = np.full(len(coords_1), np.nan)
            ang_acc = np.full((len(coords_1), dims), np.nan)
            ang_acc_mag = np.full(len(coords_1), np.nan)
            lin_acc_rel = np.full((len(coords_1), dims), np.nan)
            lin_acc_rel_mag = np.full(len(coords_1), np.nan)
            ang_acc_rel = np.full((len(coords_1), dims), np.nan)
            ang_acc_rel_mag = np.full(len(coords_1), np.nan)
            lin_acc_com = np.full((len(coords_1), dims), np.nan)
            lin_acc_com_mag = np.full(len(coords_1), np.nan)
            ang_acc_com = np.full((len(coords_1), dims), np.nan)
            ang_acc_com_mag = np.full(len(coords_1), np.nan)
            directionality_acc_com = np.full(len(coords_1), np.nan)
            directionality_acc_rel = np.full(len(coords_1), np.nan)
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
        # Calculate angles of ra and rb relative to x-axis
        theta_a = np.arctan2(ra[:, 1], ra[:, 0])
        theta_b = np.arctan2(rb[:, 1], rb[:, 0])

        # Calculate the change in angle
        delta_theta = theta_b - theta_a

        # Normalize the change in angle to be between -pi and pi
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

        # Get the angular velocity
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


    def _run_frame(self, t):
        frame_coords = np.argwhere(self.hierarchy.label_components[t] > 0)
        self.coords.append(frame_coords)

        frame_component_labels = self.hierarchy.label_components[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.component_labels.append(frame_component_labels)

        frame_branch_labels = self.hierarchy.label_branches[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.branch_labels.append(frame_branch_labels)

        frame_intensity_vals = self.hierarchy.im_raw[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.intensity.append(frame_intensity_vals)

        frame_structure_vals = self.hierarchy.im_struct[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.structure.append(frame_structure_vals)

        frame_t = np.ones(frame_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        im_name = np.ones(frame_coords.shape[0], dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_node_info(t, frame_coords)
        self._get_motility_stats(t, frame_coords)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)


class Nodes:
    def __init__(self, hierarchy, voxels):
        self.hierarchy = hierarchy
        self.voxels = voxels

        self.time = []
        self.nodes = []

        # add voxel aggregate metrics
        # add node metrics
        self.thickness = []
        self.divergence = []
        self.convergence = []
        self.vergere = []
        self.lin_magnitude_variability = []
        self.ang_magnitude_variability = []
        self.lin_direction_uniformity = []
        self.ang_direction_uniformity = []

        self.voxel_idxs = voxels.node_voxel_idxs
        self.branch_label = []
        self.component_label = []
        self.image_name = []

        self.node_z_lims = voxels.node_z_lims
        self.node_y_lims = voxels.node_y_lims
        self.node_x_lims = voxels.node_x_lims

    def _get_node_stats(self, t):
        radius = distance_check(self.hierarchy.im_border_mask[t], self.nodes[t], self.hierarchy.spacing)
        self.thickness.append(radius*2)

        divergence = []
        convergence = []
        vergere = []
        lin_mag_variability = []
        ang_mag_variability = []
        lin_dir_uniformity = []
        ang_dir_uniformity = []
        for i, node in enumerate(self.nodes[t]):
            # print(i, len(self.nodes[t]))
            vox_idxs = self.voxel_idxs[t][i]
            dist_vox_node = self.voxels.coords[t][vox_idxs] - self.nodes[t][i]
            dist_vox_node_mag = np.linalg.norm(dist_vox_node, axis=1, keepdims=True)
            dir_vox_node = dist_vox_node / dist_vox_node_mag

            dot_prod_01 = -np.nanmean(np.sum(-self.voxels.vec01[t][vox_idxs] * dir_vox_node, axis=1))
            convergence.append(dot_prod_01)

            dot_prod_12 = np.nanmean(np.sum(self.voxels.vec12[t][vox_idxs] * dir_vox_node, axis=1))
            divergence.append(dot_prod_12)

            vergere.append(dot_prod_01 + dot_prod_12)
            # high vergere is a funnel point (converges then diverges)
            # low vergere is a dispersal point (diverges then converges)

            lin_vel_mag = self.voxels.lin_vel_mag[t][vox_idxs]
            lin_mag_variability.append(np.nanstd(lin_vel_mag))
            ang_vel_mag = self.voxels.ang_vel_mag[t][vox_idxs]
            ang_mag_variability.append(np.nanstd(ang_vel_mag))

            lin_vel = self.voxels.lin_vel[t][vox_idxs]
            lin_unit_vec = lin_vel / lin_vel_mag[:, np.newaxis]
            lin_similarity_matrix = np.dot(lin_unit_vec, lin_unit_vec.T)
            np.fill_diagonal(lin_similarity_matrix, np.nan)
            lin_dir_uniformity.append(np.nanmean(lin_similarity_matrix))

            ang_vel = self.voxels.ang_vel[t][vox_idxs]
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

        frame_component_label = self.hierarchy.label_components[t][frame_skel_coords[:, 0], frame_skel_coords[:, 1], frame_skel_coords[:, 2]]
        self.component_label.append(frame_component_label)

        frame_branch_label = self.hierarchy.label_branches[t][frame_skel_coords[:, 0], frame_skel_coords[:, 1], frame_skel_coords[:, 2]]
        self.branch_label.append(frame_branch_label)

        im_name = np.ones(frame_skel_coords.shape[0], dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_node_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)
        print('hi')


def distance_check(border_mask, check_coords, spacing):
    border_coords = np.argwhere(border_mask) * spacing
    border_tree = spatial.cKDTree(border_coords)
    dist, _ = border_tree.query(check_coords * spacing, k=1)
    return dist


class Branches:
    def __init__(self, hierarchy, voxels, nodes):
        self.hierarchy = hierarchy
        self.voxels = voxels
        self.nodes = nodes

        self.time = []
        self.branch_label = []

        # self.branch_voxel_label = []
        # self.branch_skel_label = []
        # add aggregate voxel metrics
        # add aggregate node metrics
        # add branch metrics
        self.length = []
        self.thickness = []
        self.aspect_ratio = []
        self.tortuosity = []

        # self.voxel_idxs = []
        self.branch_idxs = []
        self.component_label = []
        self.image_name = []

    def _get_branch_stats(self, t):
        print('hi')
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
        neighbor_labels_0 = self.hierarchy.im_skel[t][neighbor_coords_0[:, 0], neighbor_coords_0[:, 1], neighbor_coords_0[:, 2]]
        neighbor_labels_1 = self.hierarchy.im_skel[t][neighbor_coords_1[:, 0], neighbor_coords_1[:, 1], neighbor_coords_1[:, 2]]
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

        lone_tip_labels = self.hierarchy.im_skel[t][lone_tip_coords[:, 0], lone_tip_coords[:, 1], lone_tip_coords[:, 2]]
        tip_labels = self.hierarchy.im_skel[t][tip_coords[:, 0], tip_coords[:, 1], tip_coords[:, 2]]

        # todo append radius *2 for lone tips and radius for tips to length

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

    def _run_frame(self, t):
        # frame_voxel_idxs = np.argwhere(self.hierarchy.label_branches[t] > 0)
        # self.voxel_idxs.append(frame_voxel_idxs)

        frame_branch_idxs = np.argwhere(self.hierarchy.im_skel[t] > 0)
        self.branch_idxs.append(frame_branch_idxs)

        # frame_voxel_branch_labels = self.hierarchy.label_branches[t][frame_voxel_idxs[:, 0], frame_voxel_idxs[:, 1], frame_voxel_idxs[:, 2]]
        # self.branch_voxel_label.append(frame_voxel_branch_labels)

        frame_skel_branch_labels = self.hierarchy.im_skel[t][frame_branch_idxs[:, 0], frame_branch_idxs[:, 1], frame_branch_idxs[:, 2]]
        # self.branch_skel_label.append(frame_skel_branch_labels)

        smallest_label = int(np.min(self.hierarchy.im_skel[t][self.hierarchy.im_skel[t] > 0]))
        largest_label = int(np.max(self.hierarchy.im_skel[t]))
        frame_branch_labels = np.arange(smallest_label, largest_label + 1)
        num_branches = len(frame_branch_labels)

        frame_t = np.ones(num_branches, dtype=int) * t
        self.time.append(frame_t)

        # get the first voxel idx for each branch
        frame_branch_coords = np.zeros((num_branches, 3), dtype=int)
        for i in frame_branch_labels:
            branch_voxels = frame_branch_idxs[frame_skel_branch_labels == i]
            frame_branch_coords[i-1] = branch_voxels[0]
        frame_component_label = self.hierarchy.label_components[t][frame_branch_coords[:, 0], frame_branch_coords[:, 1], frame_branch_coords[:, 2]]
        self.component_label.append(frame_component_label)

        frame_branch_label = self.hierarchy.im_skel[t][frame_branch_coords[:, 0], frame_branch_coords[:, 1], frame_branch_coords[:, 2]]
        self.branch_label.append(frame_branch_label)

        im_name = np.ones(num_branches, dtype=object) * self.hierarchy.im_info.basename_no_ext
        self.image_name.append(im_name)

        self._get_branch_stats(t)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)
        print('hi')


class Components:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.component_label = []
        # add aggregate voxel metrics
        # add aggregate node metrics
        # add aggregate branch metrics
        # add component metrics


        self.image_name = []

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

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)
        print('hi')


class Image:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.image_name = []
        # add aggregate voxel metrics
        # add aggregate branch metrics
        # add aggregate component metrics
        # add image metrics

    def _run_frame(self, t):
        self.time.append(t)
        self.image_name.append(self.hierarchy.im_info.basename_no_ext)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)
        print('hi')


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3

    hierarchy = Hierarchy(im_info, num_t)
    hierarchy.run()

    voxels = Voxels(hierarchy)
    voxels.run()

    nodes = Nodes(hierarchy, voxels)
    nodes.run()

    branches = Branches(hierarchy, voxels, nodes)
    branches.run()
    #
    # components = Components(hierarchy)
    # components.run()
    #
    # image = Image(hierarchy)
    # image.run()

