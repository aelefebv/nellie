from src import xp, ndi, logger
from src_2.im_info.im_info import ImInfo
from src_2.utils.general import get_reshaped_image
import skimage.morphology as morph
import numpy as np
from scipy.spatial import cKDTree
import skimage.measure

from src_2.utils.gpu_functions import triangle_threshold, otsu_threshold


class Network:
    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']

        if self.im_info.no_z:
            self.scaling = (im_info.dim_sizes['Y'], im_info.dim_sizes['X'])
        else:
            self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.network_memmap = None
        self.pixel_class_memmap = None
        self.skel_memmap = None
        self.skel_relabelled_memmap = None


        self.sigmas = None

        self.debug = None

    def _remove_connected_label_pixels(self, skel_labels):
        skel_labels = skel_labels.get()
        if self.im_info.no_z:
            height, width = skel_labels.shape
        else:
            depth, height, width = skel_labels.shape

        true_coords = np.argwhere(skel_labels>0)

        pixels_to_delete = []
        for coord in true_coords:
            if self.im_info.no_z:
                y, x = coord
            else:
                z, y, x = coord

            if not self.im_info.no_z:
                if z == 0 or z == depth - 1:
                    continue
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                continue  # Skip boundary voxels

            # Extract 3x3x3 neighborhood
            if self.im_info.no_z:
                label_neighborhood = skel_labels[y - 1:y + 2, x - 1:x + 2]
            else:
                label_neighborhood = skel_labels[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]

            # Get labels of set voxels in the neighborhood
            labels_in_neighborhood = label_neighborhood[label_neighborhood > 0]

            if len(set(labels_in_neighborhood.tolist())) > 1:
                if self.im_info.no_z:
                    pixels_to_delete.append((y, x))
                else:
                    pixels_to_delete.append((z, y, x))

        if self.im_info.no_z:
            for y, x in pixels_to_delete:
                skel_labels[y, x] = 0
        else:
            for z, y, x in pixels_to_delete:
                skel_labels[z, y, x] = 0

        return xp.array(skel_labels)

    def _add_missing_skeleton_labels(self, skel_frame, label_frame, frangi_frame, thresh):
        logger.debug('Adding missing skeleton labels.')
        gpu_frame = xp.array(label_frame)
        # identify unique labels and find missing ones
        unique_labels = xp.unique(gpu_frame)
        unique_skel_labels = xp.unique(skel_frame)

        missing_labels = set(unique_labels.tolist()) - set(unique_skel_labels.tolist())

        # for each missing label, find the centroid and mark it in the skeleton
        for label in missing_labels:
            if label == 0:  # ignore bg label
                continue

            label_coords = xp.argwhere(gpu_frame == label)
            label_intensities = frangi_frame[tuple(label_coords.T)]
            # max_intensity = xp.max(label_intensities)
            # if max_intensity < thresh:
            #     continue
            # centroid is where label_intensities is maximal
            centroid = label_coords[xp.argmax(label_intensities)]

            skel_frame[tuple(centroid)] = label

        return skel_frame

    def _skeletonize(self, label_frame, frangi_frame):
        # gpu_frame = xp.array(frame)
        # test = self._remove_connected_label_pixels(cpu_frame)
        cpu_frame = np.array(label_frame)
        gpu_frame = xp.array(label_frame)

        skel = xp.array(morph.skeletonize(cpu_frame > 0).astype('bool'))
        # masked_frangi = ndi.gaussian_filter(frangi_frame, sigma=0.5) * skel
        masked_frangi = ndi.median_filter(frangi_frame, size=3) * (gpu_frame>0)# * skel
        # thresh, _ = otsu_threshold(xp.log10(masked_frangi[masked_frangi > 0]))
        thresh = triangle_threshold(xp.log10(masked_frangi[masked_frangi > 0]))
        thresh = 10**thresh
        cleaned_skel = (masked_frangi > thresh) * skel
        # skel = morph.skeletonize(test > 0).astype('bool')

        skel_labels = gpu_frame * cleaned_skel
        # unique_labels = xp.unique(skel_labels)
        label_sizes = xp.bincount(skel_labels.ravel())

        above_threshold = label_sizes > 1

        mask = xp.zeros_like(skel_labels, dtype=bool)
        mask[above_threshold[skel_labels]] = True
        mask[skel_labels == 0] = False

        skel_labels = gpu_frame * mask
        # skel_labels, _ = ndi.label(cleaned_skel)

        return skel_labels, thresh

    def _get_sigma_vec(self, sigma):
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    # def _relabel_objects(self, label_frame, skel_frame):
    #     # the non-0 pixels in label_frame will be relabelled based on the nearest skeleton pixel's label
    #     skel_coords = np.argwhere(skel_frame > 0)
    #     skel_tree = cKDTree(skel_coords)
    #     label_coords = np.argwhere(label_frame > 0)
    #     _, nearest_skel_indices = skel_tree.query(label_coords, k=1)
    #     nearest_skel_labels = skel_frame[tuple(skel_coords[nearest_skel_indices].T)]
    #     label_frame[tuple(np.transpose(label_coords))] = nearest_skel_labels

    def _relabel_objects(self, branch_skel_labels, label_frame):
        if self.im_info.no_z:
            structure = xp.ones((3, 3))
        else:
            structure = xp.ones((3, 3, 3))
        # here, skel frame should be the branch labeled frame
        relabelled_labels = branch_skel_labels.copy()
        skel_mask = xp.array(branch_skel_labels > 0).astype('uint8')
        label_mask = xp.array(label_frame > 0).astype('uint8')
        skel_border = (ndi.binary_dilation(skel_mask, iterations=1, structure=structure) ^ skel_mask) * label_mask
        vox_matched = np.argwhere((branch_skel_labels > 0).get())
        vox_next_unmatched = np.argwhere(skel_border.get())
        unmatched_diff = np.inf
        while unmatched_diff:
            num_unmatched = len(vox_next_unmatched)
            if num_unmatched == 0:
                break
            tree = cKDTree(vox_matched * self.scaling)
            dists, idxs = tree.query(vox_next_unmatched * self.scaling, k=1, workers=-1)
            unmatched_matches = np.array([[vox_matched[idx], vox_next_unmatched[i]] for i, idx in enumerate(idxs)])
            if len(unmatched_matches) == 0:
                break
            matched_labels = branch_skel_labels[tuple(np.transpose(unmatched_matches[:, 0]))]
            relabelled_labels[tuple(np.transpose(unmatched_matches[:, 1]))] = matched_labels
            relabelled_mask = xp.array(relabelled_labels > 0).astype('uint8')
            # add unmatched matches to coords_matched
            skel_border = (ndi.binary_dilation(skel_border, iterations=1, structure=structure) ^ relabelled_mask) * label_mask
            vox_next_unmatched = np.argwhere(skel_border.get())
            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff_temp = abs(num_unmatched - new_num_unmatched)
            if unmatched_diff_temp == unmatched_diff:
                break
            unmatched_diff = unmatched_diff_temp
            logger.debug(f'Reassigned {unmatched_diff}/{num_unmatched} unassigned voxels. '
                         f'{new_num_unmatched} remain.')

        return relabelled_labels

    def _local_max_peak(self, frame, mask):
        lapofg = xp.empty(((len(self.sigmas),) + frame.shape), dtype=float)
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -ndi.gaussian_laplace(frame, sigma_vec) * xp.mean(s) ** 2
            current_lapofg = current_lapofg * mask
            current_lapofg[current_lapofg < 0] = 0
            lapofg[i] = current_lapofg

        filt_footprint = xp.ones((3,) * (frame.ndim + 1))
        max_filt = ndi.maximum_filter(lapofg, footprint=filt_footprint, mode='nearest')
        peaks = xp.empty(lapofg.shape, dtype=bool)
        max_filt_mask = mask
        for filt_slice, max_filt_slice in enumerate(max_filt):
            # thresh = 10**triangle_threshold(xp.log10(max_filt_slice[max_filt_slice > 0]))
            # max_filt_mask = xp.asarray(max_filt_slice > thresh) * mask
            peaks[filt_slice] = (xp.asarray(lapofg[filt_slice]) == xp.asarray(max_filt_slice)) * max_filt_mask
        # get the coordinates of all true pixels in peaks
        coords = xp.max(peaks, axis=0)
        coords_3d = xp.argwhere(coords)
        peak_im = xp.zeros_like(frame)
        peak_im[tuple(coords_3d.T)] = 1
        return coords_3d

    def _get_pixel_class(self, skel):
        skel_mask = xp.array(skel > 0).astype('uint8')
        if self.im_info.no_z:
            weights = xp.ones((3, 3))
        else:
            weights = xp.ones((3, 3, 3))
        skel_mask_sum = ndi.convolve(skel_mask, weights=weights, mode='constant', cval=0) * skel_mask
        skel_mask_sum[skel_mask_sum > 4] = 4
        return skel_mask_sum

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for skeletonization.')
        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'], read_type='r+')
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

        # im_skel_path = self.im_info.create_output_path('im_skel')
        im_skel_path = self.im_info.pipeline_paths['im_skel']
        self.skel_memmap = self.im_info.allocate_memory(im_skel_path, shape=self.shape,
                                                                  dtype='uint16',
                                                                  description='skeleton image',
                                                                  return_memmap=True)

        # im_pixel_class = self.im_info.create_output_path('im_pixel_class')
        im_pixel_class = self.im_info.pipeline_paths['im_pixel_class']
        self.pixel_class_memmap = self.im_info.allocate_memory(im_pixel_class, shape=self.shape,
                                                                dtype='uint8',
                                                                description='pixel class image',
                                                                return_memmap=True)

        im_skel_relabelled = self.im_info.pipeline_paths['im_skel_relabelled']
        self.skel_relabelled_memmap = self.im_info.allocate_memory(im_skel_relabelled, shape=self.shape,
                                                               dtype='uint32',
                                                               description='skeleton relabelled image',
                                                               return_memmap=True)

    def _get_branch_skel_labels(self, pixel_class):
        # get the labels of the skeleton pixels that are not junctions or background
        non_junctions = pixel_class > 0
        non_junctions = non_junctions * (pixel_class != 4)
        if self.im_info.no_z:
            structure = xp.ones((3, 3))
        else:
            structure = xp.ones((3, 3, 3))
        non_junction_labels, _ = ndi.label(non_junctions, structure=structure)
        return non_junction_labels

    def _run_frame(self, t):
        logger.info(f'Running network analysis, volume {t}/{self.num_t - 1}')
        label_frame = self.label_memmap[t]
        # mask_frame = xp.array(label_frame) > 0
        frangi_frame = xp.array(self.im_frangi_memmap[t])
        skel_frame, thresh = self._skeletonize(label_frame, frangi_frame)
        skel = self._remove_connected_label_pixels(skel_frame)
        skel = self._add_missing_skeleton_labels(skel, label_frame, frangi_frame, thresh)
        # if self.im_info.no_z:
        #     structure = xp.ones((3, 3))
        # else:
        #     structure = xp.ones((3, 3, 3))
        # final_skel, _ = ndi.label(skel > 0, structure=structure)
        final_skel = (skel.get() > 0) * label_frame
        pixel_class = self._get_pixel_class(final_skel)
        branch_skel_labels = self._get_branch_skel_labels(pixel_class)
        branch_labels = self._relabel_objects(branch_skel_labels, label_frame)
        return final_skel, pixel_class.get(), branch_labels.get()

    def _clean_junctions(self, pixel_class):
        junctions = pixel_class == 4
        junction_labels = skimage.measure.label(junctions)
        junction_objects = skimage.measure.regionprops(junction_labels)
        junction_centroids = [obj.centroid for obj in junction_objects]
        for junction_num, junction in enumerate(junction_objects):
            # use ckd tree to find closest junction coord to junction centroid
            if len(junction.coords) < 2:
                continue
            junction_tree = cKDTree(junction.coords)
            _, nearest_junction_indices = junction_tree.query(junction_centroids[junction_num], k=1, workers=-1)
            # remove the nearest junction coord from the junction
            junction_coords = junction.coords.tolist()
            junction_coords.pop(nearest_junction_indices)
            pixel_class[tuple(np.array(junction_coords).T)] = 3
        return pixel_class

    def _run_networking(self):
        for t in range(self.num_t):
            skel, pixel_class, skel_relabelled_memmap = self._run_frame(t)
            # pixel_class = self._clean_junctions(pixel_class)
            if self.im_info.no_t:
                self.skel_memmap[:] = skel[:]
                self.pixel_class_memmap[:] = pixel_class[:]
                self.skel_relabelled_memmap[:] = skel_relabelled_memmap[:]
            else:
                self.skel_memmap[t] = skel
                self.pixel_class_memmap[t] = pixel_class
                self.skel_relabelled_memmap[t] = skel_relabelled_memmap
            # intensity_frame = xp.asarray(self.im_frangi_memmap[t])
            # label_frame = xp.asarray(self.label_memmap[t])
            # intensity_frame = xp.asarray(self.im_memmap[t])
            # coords3d = self._local_max_peak(intensity_frame, xp.asarray(label_frame > 0))
            # self.network_memmap[t] = frame
            # self.debug = frame
            # break

    def run(self):
        self._get_t()
        self._allocate_memory()
        # self._set_default_sigmas()
        self._run_networking()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    skel = Network(im_info, num_t=3)
    skel.run()

    # import os
    # test_folder = r"D:\test_files\beading"
    # # test_folder = r"D:\test_files\nelly_tests"
    # # test_folder = r"D:\test_files\julius_examples"
    # all_files = os.listdir(test_folder)
    # all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    # im_infos = []
    # for file in all_files:
    #     im_path = os.path.join(test_folder, file)
    #     im_info = ImInfo(im_path)
    #     # im_info = ImInfo(im_path, dim_sizes={'T': 0, 'X': 0.11, 'Y': 0.11, 'Z': 0.1})
    #     im_info.create_output_path('im_instance_label')
    #     im_info.create_output_path('im_frangi')
    #     im_infos.append(im_info)
    #
    # skeletonis = []
    # for im_info in im_infos[:1]:
    #     # skel = Network(im_info)
    #     # skel = Network(im_info, num_t=4)
    #     skel = Network(im_info)
    #     skel.run()
    #     skeletonis.append(skel)

    # # check if viewer exists as a variable
    # if 'viewer' not in locals():
    #     import napari
    #     viewer = napari.Viewer()
    # # viewer.add_points(skeletonis[0].debug.get(), name='debug', size=1, face_color='red')
    # # viewer.add_points(skeletonis[1].debug.get(), name='debug', size=1, face_color='red')
    # viewer.add_image(skeletonis[0].debug.get(), name='im')
    # viewer.add_image(skeletonis[1].debug.get(), name='im')
