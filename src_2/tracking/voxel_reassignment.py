import numpy as np
from scipy.spatial import cKDTree
from tifffile import tifffile

from src_2.io.im_info import ImInfo
from src_2.tracking.flow_interpolation import FlowInterpolator

class VoxelReassigner:
    def __init__(self, im_info: ImInfo,
                 flow_interpolator: FlowInterpolator,
                 num_t=None,):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        self.flow_interpolator = flow_interpolator

        self.debug = None

    def interpolate_coords(self, coords, t):
        vectors = self.flow_interpolator.interpolate_coord(coords, t)
        kept_coords = ~np.isnan(vectors).any(axis=1)
        vectors = vectors[kept_coords]
        if self.flow_interpolator.forward:
            new_coords = coords[kept_coords] + vectors
        else:
            new_coords = coords[kept_coords] - vectors
        return new_coords, kept_coords

    def _match_voxels(self, coords_interpx, coords_real):
        coords_interpx = np.array(coords_interpx) * self.flow_interpolator.scaling
        coords_real = np.array(coords_real) * self.flow_interpolator.scaling
        tree = cKDTree(coords_real)
        dist, idx = tree.query(coords_interpx, k=1)
        return dist, idx

    def get_next_voxels(self, coords, t, next_coords_real):
        next_coords_interpx, kept_idxs = self.interpolate_coords(coords, t)
        _, matched_idx = self._match_voxels(next_coords_interpx, next_coords_real)
        matched_coords = next_coords_real[matched_idx.tolist()]
        distances = np.linalg.norm((coords[kept_idxs] - matched_coords) * self.flow_interpolator.scaling, axis=1)
        matches = matched_coords[distances < self.flow_interpolator.max_distance_um]
        return matches


if __name__ == "__main__":
    import os
    import napari
    viewer = napari.Viewer()
    test_folder = r"D:\test_files\nelly_tests"
    test_skel = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_skel.ome.tif", mode='r')
    test_label = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_instance_label.ome.tif", mode='r')

    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files[:1]:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('flow_vector_array', ext='.npy')
        im_infos.append(im_info)

    flow_interpx = FlowInterpolator(im_infos[0])
    viewer.add_labels(test_label)

    label_nums = list(range(1, np.max(test_label[0])))
    # get 100 random coords
    np.random.seed(0)
    labels = np.random.choice(len(label_nums), 10, replace=False)
    # label_num = 100
    all_mask_coords = [np.argwhere(test_label[t] > 0) for t in range(im_info.shape[0])]
    labels = [735, 629]

    voxel_reassigner = VoxelReassigner(im_infos[0], flow_interpx)
    new_label_im = np.zeros_like(test_label)
    for i, label_num in enumerate(labels):
        print(f'Label {i+1} of {len(labels)}')
        label_coords = np.argwhere(test_label[0] == label_num)
        new_label_im[0][tuple(label_coords.T)] = label_num
        for t in range(im_info.shape[0]-1):
            next_mask_coords = all_mask_coords[t+1]
            if len(label_coords) == 0:
                break
            matches = voxel_reassigner.get_next_voxels(label_coords, t, next_mask_coords)
            label_coords = np.array(matches)
            new_label_im[t+1][tuple(label_coords.T)] = label_num
    viewer.add_image(flow_interpx.im_memmap)
    viewer.add_labels(new_label_im)

    # last_t = 2
    # voxel_reassigner = VoxelReassigner(im_infos[0], flow_interpx)
    # new_label_im = np.zeros_like(test_label)
    # new_label_im[last_t][tuple(np.argwhere(test_label[last_t] == label_num).T)] = label_num
    # inverted_range = np.arange(last_t+1)[::-1][:-1]
    # wanted_coords = np.argwhere(test_label[last_t] == label_num)
    # for t in inverted_range:
    #     # label_coords = np.argwhere(test_label[t] == label_num)
    #     prev_mask_coords = np.argwhere(test_label[t-1] > 0)
    #     # all_coords = np.argwhere(test_label[t] > 0)
    #
    #     # new_labels = voxel_reassigner.get_new_label(label_coords, t, prev_mask_coords, test_label[t-1][test_label[t-1] > 0])
    #     new_labels, wanted_coords = voxel_reassigner.get_new_label(wanted_coords, t, prev_mask_coords, test_label[t-1][test_label[t-1] > 0])
    #
    #     new_label_coords = list(new_labels.keys())
    #     new_label_im[t][tuple(np.array(new_label_coords).T)] = list(new_labels.values())
    # viewer.add_labels(new_label_im)
