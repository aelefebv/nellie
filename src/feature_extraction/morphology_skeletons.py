from src import logger, xp, ndi, device_type
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class MorphologySkeletonFeatures:
    def __init__(self, im_info: ImInfo,
                 num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_memmap = None
        self.label_memmap = None
        self.network_memmap = None
        self.pixel_class_memmap = None
        self.organelle_skeleton_features_path = None

        self.features = {'t': [], 'main_label': [], 'length': [], 'radius_weighted': [], 'tortuosity_weighted': [], 'aspect_ratio_weighted': []}
        self.branch_features = {'t': [], 'label': [], 'main_label': [], 'branch_lengths': [], 'branch_radius': [], 'branch_tortuosities': [], 'branch_aspect_ratio': []}

    def _get_pixel_class(self, skel):
        skel_mask = xp.array(skel > 0).astype('uint8')
        if self.im_info.no_z:
            weights = xp.ones((3, 3))
        else:
            weights = xp.ones((3, 3, 3))
        skel_mask_sum = ndi.convolve(skel_mask, weights=weights, mode='constant', cval=0) * skel_mask
        skel_mask_sum[skel_mask_sum > 4] = 4
        return skel_mask_sum

    def _distance_check(self, mask, check_coords):
        border_mask = ndi.binary_dilation(mask, iterations=1) ^ mask

        border_mask_coords = xp.argwhere(border_mask)
        if device_type == 'cuda':
            border_mask_coords = border_mask_coords.get() * self.spacing
        else:
            border_mask_coords = border_mask_coords * self.spacing

        border_tree = cKDTree(border_mask_coords)
        if device_type == 'cuda':
            check_coords = check_coords.get()
        dist, _ = border_tree.query(check_coords * self.spacing, k=1)
        return dist

    def _get_branches(self, t):
        if self.im_info.no_z:
            structure = xp.ones((3, 3))
        else:
            structure = xp.ones((3, 3, 3))
        skel_labels_gpu = xp.array(self.network_memmap[t])
        main_labels_gpu = xp.array(self.label_memmap[t]) * (skel_labels_gpu>0)
        pixel_class = self._get_pixel_class(skel_labels_gpu)
        # everywhere where the image does not equal 0 or 4
        branch_mask = (pixel_class != 0) * (pixel_class != 4)
        branch_pixel_class = self._get_pixel_class(branch_mask)
        branch_labels, _ = ndi.label(branch_mask, structure=structure)

        branch_px = xp.where(branch_mask)
        px_main_label = main_labels_gpu[branch_px]

        # distance matrix between all branch_px, vectorized
        coord_array_1 = xp.array(branch_px).T
        coord_array_2 = xp.array(branch_px).T[:, None, :]
        dist = xp.linalg.norm(coord_array_1 - coord_array_2, axis=-1)
        dist[dist >= 2] = 0
        dist = xp.tril(dist)
        # get an array where the row and columns have the same main labels
        label_array_1 = px_main_label[:, None]
        label_array_2 = px_main_label[None, :]
        label_array = label_array_1 == label_array_2
        valid_dist = dist > 0
        # coords that are both valid and not in the same label should be set to 0 in the branch_mask

        bad_coords_matrix = valid_dist * ~label_array
        bad_idxs = np.argwhere(bad_coords_matrix).flatten()

        branch_mask[branch_px[0][bad_idxs], branch_px[1][bad_idxs], branch_px[2][bad_idxs]] = 0
        px_class = branch_pixel_class[branch_px]
        px_branch_label = branch_labels[branch_px]
        dist = dist * label_array

        # only keep lower diagonal
        pixel_neighbors = xp.where(dist > 0)
        valid_branch_labels = px_branch_label[pixel_neighbors[0]]

        scaled_coords = coord_array_1 * xp.array(self.spacing)
        scaled_coords_1 = scaled_coords[pixel_neighbors[0]]
        scaled_coords_2 = scaled_coords[pixel_neighbors[1]]

        scaled_coords_dist = xp.linalg.norm(scaled_coords_1 - scaled_coords_2, axis=-1)
        if device_type == 'cuda':
            scaled_coords_dist = scaled_coords_dist.get()

        branch_length_list = {label: [] for label in xp.unique(px_branch_label).tolist()}
        for i, label in enumerate(valid_branch_labels.tolist()):
            branch_length_list[label].append(scaled_coords_dist[i])

        lone_tips = branch_pixel_class == 1
        tips = branch_pixel_class == 2

        lone_tip_coords = xp.argwhere(lone_tips)
        tip_coords = xp.argwhere(tips)

        # match tips to branch labels, and find distance between them (should always be 2 tips)
        tip_branch_labels = branch_labels[tuple(tip_coords.T)]

        # get distance between tips
        gpu_spacing = xp.array(self.spacing)
        tip_coord_labels = {label: [] for label in np.unique(tip_branch_labels).tolist()}
        for i, label in enumerate(tip_branch_labels.tolist()):
            tip_coord_labels[label].append(tip_coords[i] * gpu_spacing)
        tip_coord_distances = {label: [] for label in np.unique(tip_branch_labels).tolist()}
        for label, coords in tip_coord_labels.items():
            tip_coord_distances[label] = xp.linalg.norm(coords[0] - coords[1])
        branch_tortuosities = {label: [] for label in np.unique(px_branch_label).tolist()}
        for label, length_list in branch_length_list.items():
            if len(length_list) == 0:
                branch_tortuosities[label] = 1.0
            elif tip_coord_distances.get(label) is None:
                if device_type == 'cuda':
                    branch_tortuosities[label] = float((xp.sum(xp.array(length_list)) / self.im_info.dim_sizes['X']).get())
                else:
                    branch_tortuosities[label] = float((xp.sum(xp.array(length_list)) / self.im_info.dim_sizes['X']))
            else:
                if device_type == 'cuda':
                    branch_tortuosities[label] = float((xp.sum(xp.array(length_list)) / tip_coord_distances[label]).get())
                else:
                    branch_tortuosities[label] = float((xp.sum(xp.array(length_list)) / tip_coord_distances[label]))


        lone_tip_radii = self._distance_check(xp.array(self.label_memmap[t])>0, lone_tip_coords)
        tip_radii = self._distance_check(xp.array(self.label_memmap[t])>0, tip_coords)

        lone_tip_labels = branch_labels[tuple(lone_tip_coords.T)]
        tip_labels = branch_labels[tuple(tip_coords.T)]
        checked_labels = set(xp.concatenate((lone_tip_labels, tip_labels)).tolist())
        all_labels = set(xp.unique(px_branch_label).tolist())
        unchecked_labels = all_labels - checked_labels
        # check for branches that don't have a tip, and add three random coords in the branch to be checked for a radius
        for label in unchecked_labels:
            branch_coords = xp.argwhere(branch_labels == label)
            random_coords = branch_coords[xp.random.choice(len(branch_coords), 3)]
            random_radii = self._distance_check(xp.array(self.label_memmap[t])>0, random_coords)
            tip_radii = np.concatenate((tip_radii, random_radii))
            tip_labels = xp.concatenate((tip_labels, xp.ones(3) * label))

        for label, radius in zip(lone_tip_labels.tolist(), lone_tip_radii):
            branch_length_list[label].append(radius*2)

        for label, radius in zip(tip_labels.tolist(), tip_radii):
            branch_length_list[label].append(radius)


        # self.branch_features['label'] = [label for label in xp.unique(px_branch_label).tolist()]
        frame_branch_labels = [label for label in xp.unique(px_branch_label).tolist()]
        self.branch_features['label'].extend(frame_branch_labels)
        main_labels = {}
        # self.branch_features['main_label'] =
        for idx, label in enumerate(px_branch_label.tolist()):
            if label not in main_labels:
                if device_type == 'cuda':
                    main_labels[label] = int(px_main_label[idx].get())
                else:
                    main_labels[label] = int(px_main_label[idx])
        frame_branch_main_labels = [main_labels[label] for label in xp.unique(px_branch_label).tolist()]
        self.branch_features['main_label'].extend(frame_branch_main_labels)
        frame_branch_lengths = {label: np.sum(np.array(length_list)) for label, length_list in branch_length_list.items()}
        self.branch_features['branch_lengths'].extend(list(frame_branch_lengths.values()))
        self.branch_features['branch_tortuosities'].extend(list(branch_tortuosities.values()))

        frame_org_main_labels = [label for label in xp.unique(px_main_label).tolist()]
        self.features['main_label'].extend(frame_org_main_labels)

        lengths = {label: [] for label in xp.unique(px_main_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            lengths[main_label].append(np.sum(branch_length_list[branch_label]))
        main_label_lengths = {label: [] for label in xp.unique(px_main_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            main_label_lengths[main_label].append(np.sum(branch_length_list[branch_label]))
        frame_org_lengths = [np.sum(np.array(length_list)) for length_list in lengths.values()]
        self.features['length'].extend(frame_org_lengths)
        # tortuosity weighted by length
        branch_weights = {label: [] for label in xp.unique(px_branch_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            branch_weights[branch_label] = frame_branch_lengths[branch_label]/np.sum(main_label_lengths[main_label])
        tortuosity_weighted = {label: [] for label in xp.unique(px_main_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            tortuosity_weighted[main_label].append(branch_tortuosities[branch_label] * branch_weights[branch_label])
        frame_org_tortuosities = [np.sum(tortuosity_list) for tortuosity_list in tortuosity_weighted.values()]
        self.features['tortuosity_weighted'].extend(frame_org_tortuosities)

        branch_mean_radii = {label: [] for label in xp.unique(px_branch_label).tolist()}
        for label, radius in zip(lone_tip_labels.tolist(), lone_tip_radii):
            branch_mean_radii[label].append(radius)
        for label, radius in zip(tip_labels.tolist(), tip_radii):
            branch_mean_radii[label].append(radius)
        branch_mean_radii = {label: np.mean(np.array(radius_list)) for label, radius_list in branch_mean_radii.items()}
        branch_aspect_ratios = {label: [] for label in xp.unique(px_branch_label).tolist()}
        for branch_idx, (branch_label, branch_mean_radius) in enumerate(branch_mean_radii.items()):
            branch_aspect_ratios[branch_label] = frame_branch_lengths[branch_label]/branch_mean_radius

        frame_branch_radius = [branch_mean_radii[label] for label in xp.unique(px_branch_label).tolist()]
        self.branch_features['branch_radius'].extend(frame_branch_radius)
        frame_branch_aspect_ratio = [branch_aspect_ratios[label] for label in xp.unique(px_branch_label).tolist()]
        self.branch_features['branch_aspect_ratio'].extend(frame_branch_aspect_ratio)

        main_label_radii = {label: [] for label in xp.unique(px_main_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            main_label_radii[main_label].append(branch_mean_radii[branch_label] * branch_weights[branch_label])
        frame_org_radius = [np.sum(radius_list) for radius_list in main_label_radii.values()]
        self.features['radius_weighted'].extend(frame_org_radius)

        aspect_ratio_weighted = {label: [] for label in xp.unique(px_main_label).tolist()}
        for branch_label, main_label in zip(frame_branch_labels, frame_branch_main_labels):
            aspect_ratio_weighted[main_label].append(branch_aspect_ratios[branch_label] * branch_weights[branch_label])
        frame_org_aspect_ratio = [np.sum(aspect_ratio_list) for aspect_ratio_list in aspect_ratio_weighted.values()]
        self.features['aspect_ratio_weighted'].extend(frame_org_aspect_ratio)
        self.features['t'].extend([t for _ in range(len(frame_org_aspect_ratio))])
        self.branch_features['t'].extend([t for _ in range(len(frame_branch_aspect_ratio))])

        # for idx, length in enumerate(self.features['length']):
        #     if length >= 1.74:  # sqrt(3)
        #         continue
        #     # self.features['length'][idx] = 2*self.im_info.dim_sizes['X']
        #     # self.features['radius_weighted'][idx] = self.im_info.dim_sizes['X']
        #     # self.features['tortuosity_weighted'][idx] = 1.0
        #     # self.features['aspect_ratio_weighted'][idx] = 1.0
        #     self.features['length'][idx] = np.nan
        #     self.features['radius_weighted'][idx] = np.nan
        #     self.features['tortuosity_weighted'][idx] = np.nan
        #     self.features['aspect_ratio_weighted'][idx] = np.nan

    def _skeleton_morphology(self):
        for t in range(1, self.num_t-1):
            print(f'Processing frame {t} of {self.num_t-1}')
            self._get_branches(t)

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        network_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.network_memmap = get_reshaped_image(network_memmap, self.num_t, self.im_info)

        pixel_class_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.pixel_class_memmap = get_reshaped_image(pixel_class_memmap, self.num_t, self.im_info)

        # self.im_info.create_output_path('morphology_skeleton_features', ext='.csv')
        self.organelle_skeleton_features_path = self.im_info.pipeline_paths['organelle_skeleton_features']

        self.branch_skeleton_features_path = self.im_info.pipeline_paths['branch_skeleton_features']

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        self.shape = self.network_memmap.shape

    def _save_features(self):
        logger.debug('Saving spatial features.')
        features_df = pd.DataFrame.from_dict(self.features)
        features_df.to_csv(self.organelle_skeleton_features_path, index=False)

        branch_features_df = pd.DataFrame.from_dict(self.branch_features)
        branch_features_df.to_csv(self.branch_skeleton_features_path, index=False)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def run(self):
        self._get_t()
        self._get_memmaps()
        self._skeleton_morphology()
        self._save_features()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)

    morphology_skeleton_features = MorphologySkeletonFeatures(im_info)
    morphology_skeleton_features.run()