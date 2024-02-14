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

        vergere_t0t1 = []
        vergere_t1t2 = []

        alignments_t0t1 = []
        alignments_t1t2 = []

        dot_prods_t0t1 = []
        dot_prods_t1t2 = []

        num_vec_thresh = 5
        sum_mag_thresh = 5
        for i, skel_coord in enumerate(skel_coords):
            print(i, len(skel_coords))
            # skel_vec = skel_vecs[i]

            # get all the vecs of mask_vecs that are within the radius of the test skel coord
            dist = np.linalg.norm(coords_t1 - skel_coord, axis=1)
            # these are all the mask coords in t1 that are within the radius of the skel coord in t1
            good_idxs = dist <= skel_thicknesses[i]

            diff_t0 = coords_t0[good_idxs] - skel_coord
            diff_t1 = coords_t1[good_idxs] - skel_coord
            diff_t2 = coords_t2[good_idxs] - skel_coord

            vec_t0t1 = diff_t1-diff_t0
            vec_t1t2 = diff_t2-diff_t1

            sum_vec_t0t1 = np.sum(vec_t0t1)
            sum_vec_t1t2 = np.sum(vec_t1t2)

            # remove any vectors with nan
            vec_t0t1 = vec_t0t1[~np.isnan(vec_t0t1).any(axis=1)]
            vec_t1t2 = vec_t1t2[~np.isnan(vec_t1t2).any(axis=1)]

            norm_vec_t0t1 = np.linalg.norm(vec_t0t1)
            norm_vec_t1t2 = np.linalg.norm(vec_t1t2)

            if np.abs(sum_vec_t0t1) > sum_mag_thresh or len(vec_t0t1) < num_vec_thresh:
                norm_vec_t0t1 = np.nan
            if np.abs(sum_vec_t1t2) > sum_mag_thresh or len(vec_t1t2) < num_vec_thresh:
                norm_vec_t1t2 = np.nan

            vergere_t0t1.append(np.sign(sum_vec_t0t1) * norm_vec_t0t1)
            vergere_t1t2.append(np.sign(sum_vec_t1t2) * norm_vec_t1t2)

            # reference_direction is the mean direction of the flow vectors
            flow_vectors_t0t1_normalized = vec_t0t1 / np.linalg.norm(vec_t0t1, axis=1, keepdims=True)
            reference_direction_t0t1 = np.nanmean(flow_vectors_t0t1_normalized, axis=0)
            reference_direction_normalized_t0t1 = reference_direction_t0t1 / np.linalg.norm(reference_direction_t0t1)
            alignments_t0t1.append(np.nanmean(np.dot(flow_vectors_t0t1_normalized, reference_direction_normalized_t0t1)))

            flow_vectors_t1t2_normalized = vec_t1t2 / np.linalg.norm(vec_t1t2, axis=1, keepdims=True)
            reference_direction_t1t2 = np.nanmean(flow_vectors_t1t2_normalized, axis=0)
            reference_direction_normalized_t1t2 = reference_direction_t1t2 / np.linalg.norm(reference_direction_t1t2)
            alignments_t1t2.append(np.nanmean(np.dot(flow_vectors_t1t2_normalized, reference_direction_normalized_t1t2)))

            norms_t0t1 = np.linalg.norm(diff_t0, axis=1, keepdims=True)
            unit_direction_vectors_t0t1 = diff_t0 / norms_t0t1

            dot_prods_t0t1.append(np.nanmean(np.sum(mask_vecs_t0t1[good_idxs] * unit_direction_vectors_t0t1, axis=1)))

            norms_t1t2 = np.linalg.norm(diff_t1, axis=1, keepdims=True)
            unit_direction_vectors_t1t2 = diff_t1 / norms_t1t2

            dot_prods_t1t2.append(np.nanmean(np.sum(mask_vecs_t1t2[good_idxs] * unit_direction_vectors_t1t2, axis=1)))


        import napari
        viewer = napari.Viewer()

        skel_mask_align_t0t1 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_align_t0t1[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = alignments_t0t1
        skel_mask_align_t0t1[skel_mask_align_t0t1 == 0] = np.nan
        viewer.add_image(skel_mask_align_t0t1, name='align_t0t1', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        skel_mask_align_t1t2 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_align_t1t2[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = alignments_t1t2
        skel_mask_align_t1t2[skel_mask_align_t1t2 == 0] = np.nan
        viewer.add_image(skel_mask_align_t1t2, name='align_t1t2', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        skel_mask_dot_t0t1 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_dot_t0t1[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = dot_prods_t0t1
        skel_mask_dot_t0t1[skel_mask_dot_t0t1 == 0] = np.nan
        viewer.add_image(skel_mask_dot_t0t1, name='dot_t0t1', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        skel_mask_dot_t1t2 = np.zeros_like(skel_mask, dtype=float)
        skel_mask_dot_t1t2[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = dot_prods_t1t2
        skel_mask_dot_t1t2[skel_mask_dot_t1t2 == 0] = np.nan
        viewer.add_image(skel_mask_dot_t1t2, name='dot_t1t2', interpolation3d='nearest', colormap='turbo', contrast_limits=[-1, 1])

        tracks = []
        track_props = {'frame_num': []}
        for i, skel_coord in enumerate(coords_t1):
            if np.any(np.isnan(coords_t0[i])) or np.any(np.isnan(coords_t1[i])) or np.any(np.isnan(coords_t2[i])):
                continue
            tracks.append([i, 0, *coords_t0[i]])
            tracks.append([i, 1, *coords_t1[i]])
            tracks.append([i, 2, *coords_t2[i]])
            track_props['frame_num'].extend([0, 1, 2])

        viewer.add_tracks(tracks, name='mask_tracks', properties=track_props)

        print('hi')

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

