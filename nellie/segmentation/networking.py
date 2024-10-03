import numpy as np
import skimage.measure
import skimage.morphology as morph
from scipy.spatial import cKDTree

from nellie import xp, ndi, logger, device_type
from nellie.im_info.verifier import ImInfo
from nellie.utils.gpu_functions import triangle_threshold, otsu_threshold


class Network:
    """
    A class for analyzing and skeletonizing network-like structures in 3D or 4D microscopy images, such as cellular branches.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    num_t : int
        Number of timepoints in the image.
    min_radius_um : float
        Minimum radius of detected objects in micrometers.
    max_radius_um : float
        Maximum radius of detected objects in micrometers.
    min_radius_px : float
        Minimum radius of detected objects in pixels.
    max_radius_px : float
        Maximum radius of detected objects in pixels.
    scaling : tuple
        Scaling factors for Z, Y, and X dimensions.
    shape : tuple
        Shape of the input image.
    im_memmap : np.ndarray or None
        Memory-mapped original image data.
    im_frangi_memmap : np.ndarray or None
        Memory-mapped Frangi-filtered image data.
    label_memmap : np.ndarray or None
        Memory-mapped label data from instance segmentation.
    network_memmap : np.ndarray or None
        Memory-mapped output for network analysis.
    pixel_class_memmap : np.ndarray or None
        Memory-mapped output for pixel classification.
    skel_memmap : np.ndarray or None
        Memory-mapped output for skeleton images.
    skel_relabelled_memmap : np.ndarray or None
        Memory-mapped output for relabeled skeletons.
    clean_skel : bool
        Whether to clean the skeletons by removing noisy parts (default is True).
    sigmas : list or None
        List of sigma values for multi-scale filtering.
    debug : dict or None
        Debugging information for tracking network analysis steps.
    viewer : object or None
        Viewer object for displaying status during processing.

    Methods
    -------
    _remove_connected_label_pixels(skel_labels)
        Removes skeleton pixels that are connected to multiple labeled regions.
    _add_missing_skeleton_labels(skel_frame, label_frame, frangi_frame, thresh)
        Adds missing labels to the skeleton where the intensity is highest within a labeled region.
    _skeletonize(label_frame, frangi_frame)
        Skeletonizes the labeled regions and cleans up the skeleton based on intensity thresholds.
    _get_sigma_vec(sigma)
        Computes the sigma vector for multi-scale filtering based on image dimensions.
    _set_default_sigmas()
        Sets the default sigma values for multi-scale filtering.
    _relabel_objects(branch_skel_labels, label_frame)
        Relabels skeleton pixels by propagating labels to nearby unlabeled pixels.
    _local_max_peak(frame, mask)
        Detects local maxima in the image using multi-scale Laplacian of Gaussian filtering.
    _get_pixel_class(skel)
        Classifies skeleton pixels into junctions, branches, and endpoints based on connectivity.
    _get_t()
        Determines the number of timepoints to process.
    _allocate_memory()
        Allocates memory for skeleton images, pixel classification, and relabeled skeletons.
    _get_branch_skel_labels(pixel_class)
        Gets the branch skeleton labels, excluding junctions and background pixels.
    _run_frame(t)
        Runs skeletonization and network analysis for a single timepoint.
    _clean_junctions(pixel_class)
        Cleans up junctions by removing closely spaced junction pixels.
    _run_networking()
        Runs the network analysis process for all timepoints in the image.
    run()
        Main method to execute the network analysis process over the image data.
    """

    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1, clean_skel=None,
                 viewer=None):
        """
        Initializes the Network object with image metadata and network analysis parameters.

        Parameters
        ----------
        im_info : ImInfo
            An instance of the ImInfo class, containing metadata and paths for the image file.
        num_t : int, optional
            Number of timepoints to process. If None, defaults to the number of timepoints in the image.
        min_radius_um : float, optional
            Minimum radius of detected objects in micrometers (default is 0.20).
        max_radius_um : float, optional
            Maximum radius of detected objects in micrometers (default is 1).
        clean_skel : bool, optional
            Whether to clean the skeleton by removing noisy parts (default is None, which means True for 3D images).
        viewer : object or None, optional
            Viewer object for displaying status during processing (default is None).
        """
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        if not self.im_info.no_z:
            if clean_skel is None:
                clean_skel = False
            self.z_ratio = self.im_info.dim_res['Z'] / self.im_info.dim_res['X']
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_res['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res['X']

        if self.im_info.no_z:
            self.scaling = (im_info.dim_res['Y'], im_info.dim_res['X'])
        else:
            self.scaling = (im_info.dim_res['Z'], im_info.dim_res['Y'], im_info.dim_res['X'])

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.network_memmap = None
        self.pixel_class_memmap = None
        self.skel_memmap = None
        self.skel_relabelled_memmap = None

        self.clean_skel = True if clean_skel is None else clean_skel

        self.sigmas = None

        self.debug = None

        self.viewer = viewer

    def _remove_connected_label_pixels(self, skel_labels):
        """
        Removes skeleton pixels that are connected to multiple labeled regions.

        This method identifies pixels that are connected to more than one labeled region in the neighborhood and removes them.

        Parameters
        ----------
        skel_labels : np.ndarray
            Skeletonized label data.

        Returns
        -------
        np.ndarray
            Cleaned skeleton data with conflicting pixels removed.
        """
        if device_type == 'cuda':
            skel_labels = skel_labels.get()

        if self.im_info.no_z:
            height, width = skel_labels.shape
        else:
            depth, height, width = skel_labels.shape

        true_coords = np.argwhere(skel_labels > 0)

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
                continue  # skip boundary voxels

            # extract 3x3x3 neighborhood
            if self.im_info.no_z:
                label_neighborhood = skel_labels[y - 1:y + 2, x - 1:x + 2]
            else:
                label_neighborhood = skel_labels[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]

            # get labels of set voxels in the neighborhood
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
        """
        Adds missing labels to the skeleton where the intensity is highest within a labeled region.

        Parameters
        ----------
        skel_frame : np.ndarray
            Skeleton data.
        label_frame : np.ndarray
            Labeled regions in the image.
        frangi_frame : np.ndarray
            Frangi-filtered image.
        thresh : float
            Threshold value used during skeleton cleaning.

        Returns
        -------
        np.ndarray
            Updated skeleton with missing labels added.
        """
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
            # centroid is where label_intensities is maximal
            centroid = label_coords[xp.argmax(label_intensities)]

            skel_frame[tuple(centroid)] = label

        return skel_frame

    def _skeletonize(self, label_frame, frangi_frame):
        """
        Skeletonizes the labeled regions and cleans up the skeleton based on intensity thresholds.

        This method applies skeletonization and optionally cleans the skeleton using intensity information.

        Parameters
        ----------
        label_frame : np.ndarray
            Labeled regions in the image.
        frangi_frame : np.ndarray
            Frangi-filtered image used for intensity-based cleaning.

        Returns
        -------
        tuple
            Skeleton data and the intensity threshold used for cleaning.
        """
        cpu_frame = np.array(label_frame)
        gpu_frame = xp.array(label_frame)

        skel = xp.array(morph.skeletonize(cpu_frame > 0).astype('bool'))

        if self.clean_skel:
            masked_frangi = ndi.median_filter(frangi_frame, size=3) * (gpu_frame > 0)  # * skel
            thresh_otsu, _ = otsu_threshold(xp.log10(masked_frangi[masked_frangi > 0]))
            thresh_otsu = 10 ** thresh_otsu
            thresh_tri = triangle_threshold(xp.log10(masked_frangi[masked_frangi > 0]))
            thresh_tri = 10 ** thresh_tri
            thresh = min(thresh_otsu, thresh_tri)
            cleaned_skel = (masked_frangi > thresh) * skel

            skel_labels = gpu_frame * cleaned_skel
            label_sizes = xp.bincount(skel_labels.ravel())

            above_threshold = label_sizes > 1

            mask = xp.zeros_like(skel_labels, dtype=bool)
            mask[above_threshold[skel_labels]] = True
            mask[skel_labels == 0] = False

            skel_labels = gpu_frame * mask
        else:
            skel_labels = gpu_frame * skel
            thresh = 0

        return skel_labels, thresh

    def _get_sigma_vec(self, sigma):
        """
        Computes the sigma vector for multi-scale filtering based on image dimensions.

        Parameters
        ----------
        sigma : float
            The sigma value to use for filtering.

        Returns
        -------
        tuple
            Sigma vector for Gaussian filtering in (Z, Y, X).
        """
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        """
        Sets the default sigma values for multi-scale filtering based on the minimum and maximum radius in pixels.
        """
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _relabel_objects(self, branch_skel_labels, label_frame):
        """
        Relabels skeleton pixels by propagating labels to nearby unlabeled pixels.

        This method uses a nearest-neighbor approach to propagate labels to unlabeled pixels in the skeleton.

        Parameters
        ----------
        branch_skel_labels : np.ndarray
            Branch skeleton labels.
        label_frame : np.ndarray
            Labeled regions in the image.

        Returns
        -------
        np.ndarray
            Relabeled skeleton.
        """
        if self.im_info.no_z:
            structure = xp.ones((3, 3))
        else:
            structure = xp.ones((3, 3, 3))
        # here, skel frame should be the branch labeled frame
        relabelled_labels = branch_skel_labels.copy()
        skel_mask = xp.array(branch_skel_labels > 0).astype('uint8')
        label_mask = xp.array(label_frame > 0).astype('uint8')
        skel_border = (ndi.binary_dilation(skel_mask, iterations=1, structure=structure) ^ skel_mask) * label_mask
        skel_label_mask = (branch_skel_labels > 0)

        if device_type == 'cuda':
            skel_label_mask = skel_label_mask.get()

        vox_matched = np.argwhere(skel_label_mask)

        if device_type == 'cuda':
            vox_next_unmatched = np.argwhere(skel_border.get())
        else:
            vox_next_unmatched = np.argwhere(skel_border)

        unmatched_diff = np.inf
        while True:
            num_unmatched = len(vox_next_unmatched)
            if num_unmatched == 0:
                break
            tree = cKDTree(vox_matched * self.scaling)
            dists, idxs = tree.query(vox_next_unmatched * self.scaling, k=1, workers=-1)
            # remove any matches that are too far away
            max_dist = 2 * np.min(self.scaling)  # sqrt 3 * max scaling
            unmatched_matches = np.array(
                [[vox_matched[idx], vox_next_unmatched[i]] for i, idx in enumerate(idxs) if dists[i] < max_dist]
            )
            if len(unmatched_matches) == 0:
                break
            matched_labels = branch_skel_labels[tuple(np.transpose(unmatched_matches[:, 0]))]
            relabelled_labels[tuple(np.transpose(unmatched_matches[:, 1]))] = matched_labels
            branch_skel_labels = relabelled_labels.copy()
            relabelled_labels_mask = relabelled_labels > 0

            if device_type == 'cuda':
                relabelled_labels_mask_cpu = relabelled_labels_mask.get()
            else:
                relabelled_labels_mask_cpu = relabelled_labels_mask

            vox_matched = np.argwhere(relabelled_labels_mask_cpu)
            relabelled_mask = relabelled_labels_mask.astype('uint8')
            # add unmatched matches to coords_matched
            skel_border = (ndi.binary_dilation(relabelled_mask, iterations=1,
                                               structure=structure) - relabelled_mask) * label_mask

            if device_type == 'cuda':
                vox_next_unmatched = np.argwhere(skel_border.get())
            else:
                vox_next_unmatched = np.argwhere(skel_border)

            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff_temp = abs(num_unmatched - new_num_unmatched)
            if unmatched_diff_temp == unmatched_diff:
                break
            unmatched_diff = unmatched_diff_temp
            logger.debug(f'Reassigned {unmatched_diff}/{num_unmatched} unassigned voxels. '
                         f'{new_num_unmatched} remain.')

        return relabelled_labels

    def _local_max_peak(self, frame, mask):
        """
        Detects local maxima in the image using multi-scale Laplacian of Gaussian filtering.

        Parameters
        ----------
        frame : np.ndarray
            The input image.
        mask : np.ndarray
            Binary mask of regions to process.

        Returns
        -------
        np.ndarray
            Coordinates of detected local maxima.
        """
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
            peaks[filt_slice] = (xp.asarray(lapofg[filt_slice]) == xp.asarray(max_filt_slice)) * max_filt_mask
        # get the coordinates of all true pixels in peaks
        coords = xp.max(peaks, axis=0)
        coords_3d = xp.argwhere(coords)
        peak_im = xp.zeros_like(frame)
        peak_im[tuple(coords_3d.T)] = 1
        return coords_3d

    def _get_pixel_class(self, skel):
        """
        Classifies skeleton pixels into junctions, branches, and endpoints based on connectivity.

        Parameters
        ----------
        skel : np.ndarray
            Skeleton data.

        Returns
        -------
        np.ndarray
            Pixel classification of skeleton points (junctions, branches, or endpoints).
        """
        skel_mask = xp.array(skel > 0).astype('uint8')
        if self.im_info.no_z:
            weights = xp.ones((3, 3))
        else:
            weights = xp.ones((3, 3, 3))
        skel_mask_sum = ndi.convolve(skel_mask, weights=weights, mode='constant', cval=0) * skel_mask
        skel_mask_sum[skel_mask_sum > 4] = 4
        return skel_mask_sum

    def _get_t(self):
        """
        Determines the number of timepoints to process.

        If `num_t` is not set and the image contains a temporal dimension, it sets `num_t` to the number of timepoints.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        """
        Allocates memory for skeleton images, pixel classification, and relabeled skeletons.

        This method creates memory-mapped arrays for the instance label data, skeleton, pixel classification, and relabeled skeletons.
        """
        logger.debug('Allocating memory for skeletonization.')
        self.label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])  # , read_type='r+')
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.im_frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        self.shape = self.label_memmap.shape

        im_skel_path = self.im_info.pipeline_paths['im_skel']
        self.skel_memmap = self.im_info.allocate_memory(im_skel_path,
                                                        dtype='uint16',
                                                        description='skeleton image',
                                                        return_memmap=True)

        im_pixel_class = self.im_info.pipeline_paths['im_pixel_class']
        self.pixel_class_memmap = self.im_info.allocate_memory(im_pixel_class,
                                                               dtype='uint8',
                                                               description='pixel class image',
                                                               return_memmap=True)

        im_skel_relabelled = self.im_info.pipeline_paths['im_skel_relabelled']
        self.skel_relabelled_memmap = self.im_info.allocate_memory(im_skel_relabelled,
                                                                   dtype='uint32',
                                                                   description='skeleton relabelled image',
                                                                   return_memmap=True)

    def _get_branch_skel_labels(self, pixel_class):
        """
        Gets the branch skeleton labels, excluding junctions and background pixels.

        Parameters
        ----------
        pixel_class : np.ndarray
            Classified skeleton pixels.

        Returns
        -------
        np.ndarray
            Branch skeleton labels.
        """
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
        """
        Runs skeletonization and network analysis for a single timepoint.

        Parameters
        ----------
        t : int
            Timepoint index.

        Returns
        -------
        tuple
            Branch skeleton labels, pixel classification, and relabeled skeletons.
        """
        logger.info(f'Running network analysis, volume {t}/{self.num_t - 1}')
        label_frame = self.label_memmap[t]
        frangi_frame = xp.array(self.im_frangi_memmap[t])
        skel_frame, thresh = self._skeletonize(label_frame, frangi_frame)
        skel = self._remove_connected_label_pixels(skel_frame)
        skel = self._add_missing_skeleton_labels(skel, label_frame, frangi_frame, thresh)
        if device_type == 'cuda':
            skel = skel.get()

        skel_pre = (skel > 0) * label_frame
        pixel_class = self._get_pixel_class(skel_pre)
        branch_skel_labels = self._get_branch_skel_labels(pixel_class)
        branch_labels = self._relabel_objects(branch_skel_labels, label_frame)
        return branch_skel_labels, pixel_class, branch_labels

    def _clean_junctions(self, pixel_class):
        """
        Cleans up junctions by removing closely spaced junction pixels.

        This method uses a KD-tree to remove redundant junction points, leaving only the centroid.

        Parameters
        ----------
        pixel_class : np.ndarray
            Pixel classification of skeleton points.

        Returns
        -------
        np.ndarray
            Cleaned pixel classification with redundant junctions removed.
        """
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
        """
        Runs the network analysis process for all timepoints in the image.

        This method processes each timepoint sequentially and applies network analysis.
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Extracting branches. Frame: {t + 1} of {self.num_t}.'
            skel, pixel_class, skel_relabelled_memmap = self._run_frame(t)
            if self.im_info.no_t or self.num_t == 1:
                if device_type == 'cuda':
                    self.skel_memmap[:] = skel[:].get()
                    self.pixel_class_memmap[:] = pixel_class[:].get()
                    self.skel_relabelled_memmap[:] = skel_relabelled_memmap[:].get()
                else:
                    self.skel_memmap[:] = skel[:]
                    self.pixel_class_memmap[:] = pixel_class[:]
                    self.skel_relabelled_memmap[:] = skel_relabelled_memmap[:]
            else:
                if device_type == 'cuda':
                    self.skel_memmap[t] = skel.get()
                    self.pixel_class_memmap[t] = pixel_class.get()
                    self.skel_relabelled_memmap[t] = skel_relabelled_memmap.get()
                else:
                    self.skel_memmap[t] = skel
                    self.pixel_class_memmap[t] = pixel_class
                    self.skel_relabelled_memmap[t] = skel_relabelled_memmap

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_networking()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    skel = Network(im_info, num_t=3)
    skel.run()
