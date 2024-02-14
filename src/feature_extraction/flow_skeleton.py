# first, get the skeleton image
# then find all the flow vectors coming from every mask pixel in the image.
# then, for each skeleton pixel + its flow vector (A), grab the surrounding pixels locs + their flow vectors (N).
# Find the relative flow vectors N - A, and these are the flow vectors for calculating interesting flow features.
import numpy as np

from src import logger
from src.im_info.im_info import ImInfo
from src.tracking.flow_interpolation import FlowInterpolator
from src.utils.general import get_reshaped_image


class FlowFeatures:
    def __init__(self, im_info, num_t):
        self.im_info = im_info
        self.num_t = num_t

        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        if self.num_t < 3:
            raise ValueError("num_t must be at least 3")

        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        self.skel_label_memmap = None
        self.object_label_memmap = None
        self.distance_memmap = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                raise ValueError("No time dimension in image.")
            self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if self.num_t < 3:
            raise ValueError("num_t must be at least 3")
        return self.num_t

    def _allocate_memory(self):
        logger.debug('Allocating memory for flow features')

        skel_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.skel_label_memmap = get_reshaped_image(skel_label_memmap, self.num_t, self.im_info)

        object_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.object_label_memmap = get_reshaped_image(object_label_memmap, self.num_t, self.im_info)

        distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.distance_memmap = get_reshaped_image(distance_memmap, self.num_t, self.im_info)

    def _get_lims(self, skel_coords, skel_radii):
        z_lims = (skel_radii[:, np.newaxis] * np.array([-1, 1]) + skel_coords[:, 0, np.newaxis]).astype(int)
        z_lims[:, 1] += 1
        y_lims = (skel_radii[:, np.newaxis] * np.array([-1, 1]) + skel_coords[:, 1, np.newaxis]).astype(int)
        y_lims[:, 1] += 1
        x_lims = (skel_radii[:, np.newaxis] * np.array([-1, 1]) + skel_coords[:, 2, np.newaxis]).astype(int)
        x_lims[:, 1] += 1

        z_lims[z_lims < 0] = 0
        y_lims[y_lims < 0] = 0
        x_lims[x_lims < 0] = 0

        z_max = self.im_info.shape[self.im_info.axes.index('Z')]
        y_max = self.im_info.shape[self.im_info.axes.index('Y')]
        x_max = self.im_info.shape[self.im_info.axes.index('X')]

        z_lims[z_lims > z_max] = z_max
        y_lims[y_lims > y_max] = y_max
        x_lims[x_lims > x_max] = x_max

        return z_lims, y_lims, x_lims

    def _get_skel_pixels_frame(self, t):
        skel_mask = self.skel_label_memmap[t] > 0
        skel_coords = np.argwhere(skel_mask)

        mask_im = np.array(self.object_label_memmap[t], dtype=bool) & ~np.array(skel_mask, dtype=bool)
        coords_t1 = np.argwhere(mask_im)

        skel_thicknesses = self.distance_memmap[t][skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] * 2

        mask_vecs_t0t1 = self.flow_interpolator_bw.interpolate_coord(coords_t1, t)
        mask_vecs_t1t2 = self.flow_interpolator_fw.interpolate_coord(coords_t1, t)

        coords_t0 = coords_t1 - mask_vecs_t0t1
        coords_t2 = coords_t1 + mask_vecs_t1t2

        # >0 values means moving away from skeleton point
        dot_prods_t0t1 = []
        dot_prods_t1t2 = []

        # higher values means more variability in flow speed
        flow_mag_variation_t0t1 = []
        flow_mag_variation_t1t2 = []

        # higher values means more uniform flow direction
        flow_direction_uniformity_t0t1 = []
        flow_direction_uniformity_t1t2 = []

        for i, skel_coord in enumerate(skel_coords):
            print(i, len(skel_coords))

            # get all the vecs of mask_vecs that are within the radius of the test skel coord
            dist = np.linalg.norm(coords_t1 - skel_coord, axis=1)
            # these are all the mask coords in t1 that are within the radius of the skel coord in t1
            good_idxs = dist <= skel_thicknesses[i]

            diff_t1 = coords_t1[good_idxs] - skel_coord

            norms_t0t1 = np.linalg.norm(diff_t1, axis=1, keepdims=True)
            unit_direction_vectors_t0t1 = diff_t1 / norms_t0t1
            dot_prods_t0t1.append(np.nanmean(np.sum(-mask_vecs_t0t1[good_idxs] * unit_direction_vectors_t0t1, axis=1)))

            norms_t1t2 = np.linalg.norm(diff_t1, axis=1, keepdims=True)
            unit_direction_vectors_t1t2 = diff_t1 / norms_t1t2
            dot_prods_t1t2.append(np.nanmean(np.sum(mask_vecs_t1t2[good_idxs] * unit_direction_vectors_t1t2, axis=1)))

            flow_vec_mags_t0t1 = np.linalg.norm(-mask_vecs_t0t1[good_idxs], axis=1)
            flow_vec_mags_t1t2 = np.linalg.norm(mask_vecs_t1t2[good_idxs], axis=1)

            flow_mag_variation_t0t1.append(np.nanstd(flow_vec_mags_t0t1))
            flow_mag_variation_t1t2.append(np.nanstd(flow_vec_mags_t1t2))

            # Normalize vectors to unit vectors
            norms_t0t1 = np.linalg.norm(-mask_vecs_t0t1[good_idxs], axis=1, keepdims=True)
            unit_vectors_t0t1 = -mask_vecs_t0t1[good_idxs] / norms_t0t1
            # Calculate cosine similarity matrix
            similarity_matrix_t0t1 = np.dot(unit_vectors_t0t1, unit_vectors_t0t1.T)
            # Exclude diagonal elements (self-similarity) and calculate average similarity
            np.fill_diagonal(similarity_matrix_t0t1, np.nan)
            flow_direction_uniformity_t0t1.append(np.nanmean(similarity_matrix_t0t1))

            norms_t1t2 = np.linalg.norm(mask_vecs_t1t2[good_idxs], axis=1, keepdims=True)
            unit_vectors_t1t2 = mask_vecs_t1t2[good_idxs] / norms_t1t2
            similarity_matrix_t1t2 = np.dot(unit_vectors_t1t2, unit_vectors_t1t2.T)
            np.fill_diagonal(similarity_matrix_t1t2, np.nan)
            flow_direction_uniformity_t1t2.append(np.nanmean(similarity_matrix_t1t2))

        import napari
        viewer = napari.Viewer()

        skel_mask_dot_t0t1 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_dot_t0t1[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = dot_prods_t0t1
        skel_mask_dot_t0t1[skel_mask_dot_t0t1 == 0] = np.nan
        viewer.add_image(skel_mask_dot_t0t1, name='dot_t0t1', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        skel_mask_dot_t1t2 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_dot_t1t2[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = dot_prods_t1t2
        skel_mask_dot_t1t2[skel_mask_dot_t1t2 == 0] = np.nan
        viewer.add_image(skel_mask_dot_t1t2, name='dot_t1t2', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        skel_mask_flow_mag_var_t0t1 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_flow_mag_var_t0t1[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = flow_mag_variation_t0t1
        skel_mask_flow_mag_var_t0t1[skel_mask_flow_mag_var_t0t1 == 0] = np.nan
        viewer.add_image(skel_mask_flow_mag_var_t0t1, name='flow_mag_var_t0t1', interpolation3d='nearest', colormap='turbo')

        skel_mask_flow_mag_var_t1t2 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_flow_mag_var_t1t2[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = flow_mag_variation_t1t2
        skel_mask_flow_mag_var_t1t2[skel_mask_flow_mag_var_t1t2 == 0] = np.nan
        viewer.add_image(skel_mask_flow_mag_var_t1t2, name='flow_mag_var_t1t2', interpolation3d='nearest', colormap='turbo')

        skel_mask_flow_dir_unif_t0t1 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_flow_dir_unif_t0t1[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = flow_direction_uniformity_t0t1
        skel_mask_flow_dir_unif_t0t1[skel_mask_flow_dir_unif_t0t1 == 0] = np.nan
        viewer.add_image(skel_mask_flow_dir_unif_t0t1, name='flow_dir_unif_t0t1', interpolation3d='nearest', colormap='turbo')

        skel_mask_flow_dir_unif_t1t2 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_flow_dir_unif_t1t2[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = flow_direction_uniformity_t1t2
        skel_mask_flow_dir_unif_t1t2[skel_mask_flow_dir_unif_t1t2 == 0] = np.nan
        viewer.add_image(skel_mask_flow_dir_unif_t1t2, name='flow_dir_unif_t1t2', interpolation3d='nearest', colormap='turbo')


        tracks = []
        track_props = {'frame_num': []}
        for i, skel_coord in enumerate(coords_t1):
            if np.any(np.isnan(coords_t0[i])) or np.any(np.isnan(coords_t1[i])) or np.any(np.isnan(coords_t2[i])):
                continue
            tracks.append([i, 0, *coords_t0[i]])
            tracks.append([i, 1, *coords_t1[i]])
            tracks.append([i, 2, *coords_t2[i]])
            track_props['frame_num'].extend([0, 1, 2])

        viewer.add_tracks(tracks, name='tracks', properties=track_props)

    def _get_skel_pixels_all(self):
        for t in range(1, self.num_t):
            self._get_skel_pixels_frame(t)

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._get_skel_pixels_all()


if __name__ == "__main__":
    tif_file = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(tif_file)
    run_obj = FlowFeatures(im_info, num_t=3)
    run_obj.run()

