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

    def interpolate_previous_coords(self, coords, t):
        prev_coords = []
        kept_coords = []
        for coord in coords:
            vector = self.flow_interpolator.interpolate_coord(coord, t, forward=False)
            if vector is None:
                kept_coords.append(False)
                continue
            prev_coords.append(coord - vector)
            kept_coords.append(True)
        return prev_coords, kept_coords

    def _match_voxels(self, coords_interpx, coords_real):
        coords_interpx = np.array(coords_interpx) * self.flow_interpolator.scaling
        coords_real = np.array(coords_real) * self.flow_interpolator.scaling
        tree = cKDTree(coords_real)
        dist, idx = tree.query(coords_interpx, k=1)
        return dist, idx

    def get_previous_voxels(self, coords, t, prev_coords_real):
        prev_coords_interpx, kept_idxs = self.interpolate_previous_coords(coords, t)
        _, matched_idx = self._match_voxels(prev_coords_interpx, prev_coords_real)
        matched_coords = prev_coords_real[matched_idx]
        distances = np.linalg.norm((coords[kept_idxs] - matched_coords) * self.flow_interpolator.scaling, axis=1)
        matches = list(zip(
            coords[kept_idxs][distances < self.flow_interpolator.max_distance_um],
            matched_idx[distances < self.flow_interpolator.max_distance_um]
        ))
        return matches

    def get_new_label(self, coords, t, prev_coords_real, prev_labels):
        matches = self.get_previous_voxels(coords, t, prev_coords_real)
        new_labels = [prev_labels[tuple(match[1])] for match in matches]


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

    label_num = 115
    t = 1

    label_coords = np.argwhere(test_label[t] == label_num)
    prev_mask_coords = np.argwhere(test_label[t-1] > 0)

    voxel_reassigner = VoxelReassigner(im_infos[0], flow_interpx)
    voxel_reassigner.get_new_label(label_coords, t, prev_mask_coords, test_label[t-1][test_label[t-1] > 0])
