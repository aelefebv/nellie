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
        new_coords = []
        kept_coords = []
        for coord in coords:
            vector = self.flow_interpolator.interpolate_coord(coord, t)
            if vector is None:
                kept_coords.append(False)
                continue
            if self.flow_interpolator.forward:
                new_coords.append(coord + vector)
            else:
                new_coords.append(coord - vector)
            kept_coords.append(True)
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
        matched_coords = next_coords_real[matched_idx]
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
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('flow_vector_array', ext='.npy')
        im_infos.append(im_info)

    flow_interpx = FlowInterpolator(im_infos[0])
    viewer.add_labels(test_label)

    label_num = 121

    voxel_reassigner = VoxelReassigner(im_infos[0], flow_interpx)
    new_label_im = np.zeros_like(test_label)
    new_label_im[0][tuple(np.argwhere(test_label[0] == label_num).T)] = label_num
    for t in range(9):
        label_coords = np.argwhere(new_label_im[t] == label_num)
        next_mask_coords = np.argwhere(test_label[t+1] > 0)
        matches = voxel_reassigner.get_next_voxels(label_coords, t, next_mask_coords)

        new_label_im[t+1][tuple(np.array(matches).T)] = label_num
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
