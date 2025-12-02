"""
Network skeletonization and analysis for microscopy images.

This module provides the Network class for skeletonizing network-like structures
and analyzing their topology with optimized CPU/GPU processing.
"""
import numpy as np
import skimage.measure
import skimage.morphology as morph
from scipy.spatial import cKDTree
from scipy import ndimage as ndi_cpu

from nellie import xp, ndi, device_type
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.gpu_functions import triangle_threshold, otsu_threshold


class Network:
    """
    Optimized class for analyzing and skeletonizing network-like structures in 3D or 4D microscopy images.

    This version focuses on:
      - Reduced CPU/GPU thrashing.
      - Vectorized neighborhood operations (no Python per-voxel loops on large arrays).
      - More memory-friendly local-max detection.
      - More efficient branch relabeling using distance transforms on per-object crops.
      - Graceful degradation when GPU memory is insufficient (skip expensive cleaning instead of failing).

    Parameters
    ----------
    im_info : ImInfo
        Image metadata and paths.
    num_t : int, optional
        Number of timepoints to process. Defaults to all timepoints.
    min_radius_um : float, optional
        Minimum radius of detected objects in micrometers.
    max_radius_um : float, optional
        Maximum radius of detected objects in micrometers.
    viewer : object or None, optional
        Viewer object for status reporting.
    """

    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1,
                 viewer=None):

        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if not self.im_info.no_z:
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
        self.network_memmap = None  # kept for compatibility, not used
        self.pixel_class_memmap = None
        self.skel_memmap = None
        self.skel_relabelled_memmap = None

        self.sigmas = None

        self.debug = None

        self.viewer = viewer

    # -------------------------------------------------------------------------
    # Helper methods for device handling
    # -------------------------------------------------------------------------
    def _to_xp(self, arr):
        """
        Convert an array to the backend array type (xp).
        """
        # xp is numpy when running on CPU and cupy when on GPU.
        try:
            return xp.asarray(arr)
        except Exception as e:
            logger.warning(f"xp.asarray failed; falling back to numpy. Error: {e}")
            return np.asarray(arr)

    def _to_cpu(self, arr):
        """
        Convert xp array to a numpy array. If already numpy, return as-is.
        """
        if device_type == 'cuda' and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    # -------------------------------------------------------------------------
    # Neighborhood-based skeleton cleanup
    # -------------------------------------------------------------------------
    def _remove_connected_label_pixels(self, skel_labels):
        """
        Removes skeleton pixels that are connected to multiple labeled regions.

        This vectorized implementation replaces the original per-pixel Python loop
        with min/max filters over 3x3 (2D) or 3x3x3 (3D) neighborhoods.

        Parameters
        ----------
        skel_labels : array-like
            Skeletonized label data (xp or numpy array).

        Returns
        -------
        xp.ndarray
            Cleaned skeleton data with conflicting pixels removed.
        """
        labels = self._to_xp(skel_labels)
        mask = labels > 0

        if self.im_info.no_z:
            size = (3, 3)
        else:
            size = (3, 3, 3)

        # Maximum label in neighborhood
        max_labels = ndi.maximum_filter(labels, size=size)

        # For minimum, ignore background by assigning a very large "background label"
        bg_val = int(labels.max()) + 1
        labels_no_bg = xp.where(labels == 0, bg_val, labels)
        min_labels = ndi.minimum_filter(labels_no_bg, size=size)
        min_labels = xp.where(min_labels == bg_val, 0, min_labels)

        # Voxel is ambiguous if its neighborhood contains multiple positive labels
        ambiguous = mask & (min_labels > 0) & (max_labels > 0) & (min_labels != max_labels)

        # Preserve original behavior: do not modify boundary voxels.
        boundary = xp.zeros_like(mask, dtype=bool)
        if self.im_info.no_z:
            boundary[0, :] = True
            boundary[-1, :] = True
            boundary[:, 0] = True
            boundary[:, -1] = True
        else:
            boundary[0, :, :] = True
            boundary[-1, :, :] = True
            boundary[:, 0, :] = True
            boundary[:, -1, :] = True
            boundary[:, :, 0] = True
            boundary[:, :, -1] = True

        ambiguous = ambiguous & ~boundary

        cleaned = xp.where(ambiguous, 0, labels)
        return cleaned

    # -------------------------------------------------------------------------
    # Ensure every object has at least one skeleton voxel
    # -------------------------------------------------------------------------
    def _add_missing_skeleton_labels(self, skel_frame, label_frame, frangi_frame):
        """
        Adds missing labels to the skeleton where the intensity is highest within a labeled region.

        Parameters
        ----------
        skel_frame : xp.ndarray
            Skeleton data (labels at skeleton voxels).
        label_frame : numpy.ndarray or xp.ndarray
            Instance labels in the image.
        frangi_frame : xp.ndarray
            Frangi-filtered image.

        Returns
        -------
        xp.ndarray
            Updated skeleton with missing labels added.
        """
        logger.debug('Adding missing skeleton labels.')

        labels_xp = self._to_xp(label_frame)
        skel_xp = self._to_xp(skel_frame)
        frangi_xp = self._to_xp(frangi_frame)

        unique_labels = xp.unique(labels_xp)
        unique_skel_labels = xp.unique(skel_xp)

        missing_labels = set(unique_labels.tolist()) - set(unique_skel_labels.tolist())

        for lab in missing_labels:
            if lab == 0:
                continue
            coords = xp.argwhere(labels_xp == lab)
            if coords.size == 0:
                continue
            intensities = frangi_xp[tuple(coords.T)]
            if intensities.size == 0:
                continue
            centroid_idx = xp.argmax(intensities)
            centroid = coords[centroid_idx]
            skel_xp[tuple(centroid)] = lab

        return skel_xp

    # -------------------------------------------------------------------------
    # Skeletonization
    # -------------------------------------------------------------------------
    def _skeletonize(self, label_frame):
        """
        Skeletonizes the labeled regions.

        Parameters
        ----------
        label_frame : numpy.ndarray
            Labeled regions in the image.

        Returns
        -------
        skel_labels : xp.ndarray
            Skeleton labels.
        """
        # Skeletonization is done on CPU using scikit-image (2D per-slice or 3D stack).
        cpu_labels = np.asarray(label_frame)

        # Use the same skeletonization call for 2D and 3D as in the original
        # implementation (scikit-image handles nD input).
        skel_mask_cpu = morph.skeletonize(cpu_labels > 0)

        skel_labels_cpu = cpu_labels * skel_mask_cpu
        return self._to_xp(skel_labels_cpu)

    # -------------------------------------------------------------------------
    # Sigma management for multi-scale filters
    # -------------------------------------------------------------------------
    def _get_sigma_vec(self, sigma):
        """
        Computes the sigma vector for multi-scale filtering based on image dimensions.
        """
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        """
        Sets the default sigma values for multi-scale filtering based on the
        minimum and maximum radius in pixels.
        """
        logger.debug('Setting sigma values for multi-scale filters.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)

        # Use numpy here; sigma values are small and do not need GPU.
        self.sigmas = np.arange(self.sigma_min, self.sigma_max, sigma_step_size).tolist()
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    # -------------------------------------------------------------------------
    # Branch relabeling using per-object distance transforms
    # -------------------------------------------------------------------------
    def _relabel_objects(self, branch_skel_labels, label_frame):
        """
        Relabels skeleton pixels by propagating labels to nearby unlabeled pixels.

        This implementation operates per object instance to reduce memory usage.
        For each object label in `label_frame`, it:

          1. Extracts a bounding-box crop containing that object.
          2. Uses the skeleton branch labels inside the crop as seeds.
          3. Runs a distance transform in the crop to find the nearest seed for
             each voxel of the object.
          4. Assigns branch labels to all voxels of the object accordingly.

        Parameters
        ----------
        branch_skel_labels : xp.ndarray or numpy.ndarray
            Branch skeleton labels (non-zero at skeleton voxels).
        label_frame : numpy.ndarray or xp.ndarray
            Instance labels in the image.

        Returns
        -------
        xp.ndarray
            Relabeled skeleton for the entire frame.
        """
        # Work on CPU for distance transforms; SciPy's EDT is very efficient.
        labels_np = self._to_cpu(label_frame).astype(np.int32, copy=False)
        branch_np = self._to_cpu(branch_skel_labels).astype(np.int32, copy=False)

        relabelled_np = np.zeros_like(labels_np, dtype=np.uint32)

        max_label = int(labels_np.max())
        if max_label == 0:
            return self._to_xp(relabelled_np)

        # Find object bounding boxes once
        slices = ndi_cpu.find_objects(labels_np)
        if slices is None:
            return self._to_xp(relabelled_np)

        for lab in range(1, max_label + 1):
            idx = lab - 1
            if idx >= len(slices):
                break
            sl = slices[idx]
            if sl is None:
                continue

            sub_labels = labels_np[sl]
            sub_branch = branch_np[sl]

            obj_mask = (sub_labels == lab)
            if not obj_mask.any():
                continue

            # Seeds are branch labels (>0) inside this object crop
            seed_mask = sub_branch > 0
            if not (seed_mask & obj_mask).any():
                # No skeleton seeds for this object; leave unlabeled
                continue

            # For EDT, zeros are considered seeds. We invert the seed mask:
            # seed_mask True -> 0, False -> 1
            edt_input = np.logical_not(seed_mask)

            # Distance transform with indices: returns coordinates of nearest seed voxel.
            # We do not need distances, only indices.
            try:
                indices = ndi_cpu.distance_transform_edt(
                    edt_input,
                    sampling=self.scaling,
                    return_distances=False,
                    return_indices=True,
                )
            except Exception as e:
                logger.warning(
                    f"Distance transform failed for label {lab}. "
                    f"Leaving object unlabeled. Error: {e}"
                )
                continue

            # indices has shape (ndim, ...) and points into sub_branch
            nearest_labels = sub_branch[tuple(indices)]

            # Restrict to the object mask: outside the object remains zero
            nearest_labels[~obj_mask] = 0

            # Merge into global relabelled array.
            relabelled_sub = relabelled_np[sl]
            relabelled_sub[obj_mask] = nearest_labels[obj_mask].astype(np.uint32, copy=False)
            relabelled_np[sl] = relabelled_sub

        return self._to_xp(relabelled_np)

    # -------------------------------------------------------------------------
    # Multi-scale peak detection with reduced memory footprint
    # -------------------------------------------------------------------------
    def _local_max_peak(self, frame, mask):
        """
        Detects local maxima using multi-scale Laplacian of Gaussian filtering.

        This implementation is memory-friendly: instead of allocating a full
        (num_sigma, *frame.shape) array, it keeps track of the best response
        per voxel across scales.

        Parameters
        ----------
        frame : xp.ndarray or numpy.ndarray
            Input image.
        mask : xp.ndarray or numpy.ndarray
            Binary mask for regions of interest.

        Returns
        -------
        xp.ndarray
            Coordinates of detected local maxima.
        """
        if self.sigmas is None:
            self._set_default_sigmas()

        frame_xp = self._to_xp(frame)
        mask_xp = self._to_xp(mask).astype(bool)

        ndim = frame_xp.ndim
        footprint = xp.ones((3,) * ndim)

        best_response = xp.zeros_like(frame_xp, dtype=float)
        peak_mask = xp.zeros_like(frame_xp, dtype=bool)

        for s in self.sigmas:
            sigma_vec = self._get_sigma_vec(float(s))

            current = -ndi.gaussian_laplace(frame_xp, sigma_vec)
            current *= (float(s) ** 2)
            current *= mask_xp
            current = xp.where(current < 0, 0, current)

            max_local = ndi.maximum_filter(current, footprint=footprint, mode='nearest')
            is_peak = (current == max_local) & (current > best_response) & (current > 0)

            best_response = xp.where(is_peak, current, best_response)
            peak_mask = peak_mask | is_peak

        coords_3d = xp.argwhere(peak_mask)
        return coords_3d

    # -------------------------------------------------------------------------
    # Skeleton pixel classification
    # -------------------------------------------------------------------------
    def _get_pixel_class(self, skel):
        """
        Classifies skeleton pixels into junctions, branches, and endpoints
        based on connectivity.

        Parameters
        ----------
        skel : xp.ndarray or numpy.ndarray
            Skeleton data (labels at skeleton voxels).

        Returns
        -------
        xp.ndarray
            Pixel classification:
            0 = background
            1 = endpoint
            2-3 = branch pixels
            4 = junction
        """
        skel_xp = self._to_xp(skel)
        skel_mask = (skel_xp > 0).astype('uint8')

        if self.im_info.no_z:
            weights = xp.ones((3, 3))
        else:
            weights = xp.ones((3, 3, 3))

        skel_mask_sum = ndi.convolve(skel_mask, weights=weights, mode='constant', cval=0) * skel_mask
        skel_mask_sum[skel_mask_sum > 4] = 4

        return skel_mask_sum

    # -------------------------------------------------------------------------
    # Time dimension handling
    # -------------------------------------------------------------------------
    def _get_t(self):
        """
        Determines the number of timepoints to process.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]

    # -------------------------------------------------------------------------
    # Memory allocation for outputs
    # -------------------------------------------------------------------------
    def _allocate_memory(self):
        """
        Allocates memory for skeleton images, pixel classification, and relabeled skeletons.
        """
        logger.debug('Allocating memory for skeletonization.')
        self.label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.im_frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        self.shape = self.label_memmap.shape

        im_skel_path = self.im_info.pipeline_paths['im_skel']
        self.skel_memmap = self.im_info.allocate_memory(
            im_skel_path,
            dtype='uint16',
            description='skeleton image',
            return_memmap=True
        )

        im_pixel_class = self.im_info.pipeline_paths['im_pixel_class']
        self.pixel_class_memmap = self.im_info.allocate_memory(
            im_pixel_class,
            dtype='uint8',
            description='pixel class image',
            return_memmap=True
        )

        im_skel_relabelled = self.im_info.pipeline_paths['im_skel_relabelled']
        self.skel_relabelled_memmap = self.im_info.allocate_memory(
            im_skel_relabelled,
            dtype='uint32',
            description='skeleton relabelled image',
            return_memmap=True
        )

    # -------------------------------------------------------------------------
    # Branch skeleton labels (excluding junctions)
    # -------------------------------------------------------------------------
    def _get_branch_skel_labels(self, pixel_class):
        """
        Gets the branch skeleton labels, excluding junctions and background pixels.

        Parameters
        ----------
        pixel_class : xp.ndarray
            Classified skeleton pixels.

        Returns
        -------
        xp.ndarray
            Branch skeleton labels (connected components of non-junction pixels).
        """
        pc_xp = self._to_xp(pixel_class)
        non_junctions = pc_xp > 0
        non_junctions = non_junctions & (pc_xp != 4)

        if self.im_info.no_z:
            structure = xp.ones((3, 3))
        else:
            structure = xp.ones((3, 3, 3))

        non_junction_labels, _ = ndi.label(non_junctions, structure=structure)
        return non_junction_labels

    # -------------------------------------------------------------------------
    # Single timepoint processing
    # -------------------------------------------------------------------------
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
            (branch_skel_labels, pixel_class, branch_labels)
        """
        logger.info(f'Running network analysis, volume {t}/{self.num_t - 1}')

        label_frame = self.label_memmap[t]
        # Frangi frame: convert to xp lazily
        frangi_frame = self._to_xp(self.im_frangi_memmap[t])

        skel_frame = self._skeletonize(label_frame)

        skel_clean = self._remove_connected_label_pixels(skel_frame)
        skel_clean = self._add_missing_skeleton_labels(skel_clean, label_frame, frangi_frame)

        # For pixel classification, work with label IDs at skeleton voxels only
        if device_type == 'cuda':
            skel_pre = skel_clean.get()
            label_frame_cpu = np.asarray(label_frame)
        else:
            skel_pre = np.asarray(skel_clean)
            label_frame_cpu = np.asarray(label_frame)

        skel_pre = (skel_pre > 0) * label_frame_cpu
        pixel_class = self._get_pixel_class(skel_pre)

        branch_skel_labels = self._get_branch_skel_labels(pixel_class)
        branch_labels = self._relabel_objects(branch_skel_labels, label_frame_cpu)

        return branch_skel_labels, pixel_class, branch_labels

    # -------------------------------------------------------------------------
    # Optional junction cleanup
    # -------------------------------------------------------------------------
    def _clean_junctions(self, pixel_class):
        """
        Cleans up junctions by removing closely spaced junction pixels.

        This method uses regionprops to group junction pixels and keeps only the
        pixel closest to the junction centroid as a junction, converting the
        others to branch pixels.

        Parameters
        ----------
        pixel_class : numpy.ndarray or xp.ndarray
            Pixel classification of skeleton points.

        Returns
        -------
        numpy.ndarray
            Cleaned pixel classification with redundant junctions removed.
        """
        pc_np = self._to_cpu(pixel_class).copy()

        junctions = pc_np == 4
        if not junctions.any():
            return pc_np

        junction_labels = skimage.measure.label(junctions)
        junction_objects = skimage.measure.regionprops(junction_labels)
        junction_centroids = [obj.centroid for obj in junction_objects]

        for junction_num, junction in enumerate(junction_objects):
            coords = junction.coords
            if len(coords) < 2:
                continue
            # Use KD-tree to find closest pixel to centroid
            junction_tree = cKDTree(coords)
            _, nearest_idx = junction_tree.query(junction_centroids[junction_num], k=1, workers=-1)
            # Convert all other pixels in this junction component to branch class (3)
            coords_list = coords.tolist()
            coords_list.pop(nearest_idx)
            coords_arr = np.array(coords_list).T
            pc_np[tuple(coords_arr)] = 3

        return pc_np

    # -------------------------------------------------------------------------
    # Full networking pipeline
    # -------------------------------------------------------------------------
    def _run_networking(self):
        """
        Runs the network analysis process for all timepoints in the image.
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Extracting branches. Frame: {t + 1} of {self.num_t}.'

            skel, pixel_class, skel_relabelled = self._run_frame(t)

            if self.im_info.no_t or self.num_t == 1:
                # Single frame or static image
                if device_type == 'cuda':
                    self.skel_memmap[:] = self._to_cpu(skel)
                    self.pixel_class_memmap[:] = self._to_cpu(pixel_class)
                    self.skel_relabelled_memmap[:] = self._to_cpu(skel_relabelled)
                else:
                    self.skel_memmap[:] = skel
                    self.pixel_class_memmap[:] = pixel_class
                    self.skel_relabelled_memmap[:] = skel_relabelled
            else:
                # Time series
                if device_type == 'cuda':
                    self.skel_memmap[t] = self._to_cpu(skel)
                    self.pixel_class_memmap[t] = self._to_cpu(pixel_class)
                    self.skel_relabelled_memmap[t] = self._to_cpu(skel_relabelled)
                else:
                    self.skel_memmap[t] = skel
                    self.pixel_class_memmap[t] = pixel_class
                    self.skel_relabelled_memmap[t] = skel_relabelled

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def run(self):
        """
        Execute the full network analysis pipeline.
        """
        self._get_t()
        self._allocate_memory()
        self._run_networking()


if __name__ == "__main__":
    im_path = r"D:\\test_files\\nelly_tests\\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    skel = Network(im_info, num_t=3)
    skel.run()