import numpy as np

from src.case_studies.ionomycin_embeddings.multimesh_GNN import import_data, run_model
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
viewer = None


class Reconstructor:
    def __init__(self, im_info: ImInfo,
                 t=1):
        self.im_info = im_info
        self.t = t
        self.shape = None
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_memmap = None
        self.label_memmap = None
        self.pixel_class = None
        self.distance_memmap = None
        self.preproc_memmap = None
        self.flow_vector_array = None

    def _get_memmaps(self):
        num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if num_t == 1:
            self.t = 0

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, num_t, self.im_info)
        preproc_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.preproc_memmap = get_reshaped_image(preproc_memmap, num_t, self.im_info)
        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, num_t, self.im_info)

        pixel_class = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.pixel_class = get_reshaped_image(pixel_class, num_t, self.im_info)


        rel_ang_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_vel_mag_12'])
        self.rel_ang_vel_mag_12 = get_reshaped_image(rel_ang_vel_mag_12, num_t, self.im_info)
        rel_lin_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_vel_mag_12'])
        self.rel_lin_vel_mag_12 = get_reshaped_image(rel_lin_vel_mag_12, num_t, self.im_info)
        rel_ang_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_acc_mag'])
        self.rel_ang_acc_mag = get_reshaped_image(rel_ang_acc_mag, num_t, self.im_info)
        rel_lin_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_acc_mag'])
        self.rel_lin_acc_mag = get_reshaped_image(rel_lin_acc_mag, num_t, self.im_info)

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)

        if not self.im_info.no_t:
            self.im_memmap = self.im_memmap[self.t]
            self.preproc_memmap = self.preproc_memmap[self.t]
            self.pixel_class = self.pixel_class[self.t]
            self.label_memmap = self.label_memmap[self.t]
            self.rel_ang_vel_mag_12 = self.rel_ang_vel_mag_12[self.t]
            self.rel_lin_vel_mag_12 = self.rel_lin_vel_mag_12[self.t]
            self.rel_ang_acc_mag = self.rel_ang_acc_mag[self.t]
            self.rel_lin_acc_mag = self.rel_lin_acc_mag[self.t]

        self.shape = self.pixel_class.shape

    def run(self):
        self._get_memmaps()

def generate_array(mean, min_val, max_val, cv, num_pxs):
    # Calculate standard deviation
    std_dev = cv * mean

    # Start with an array of values normally distributed around the mean
    array = np.random.normal(mean, std_dev, num_pxs)

    # Ensure values are within the min-max range
    array = np.clip(array, min_val, max_val)

    # # Iteratively adjust the array to meet the mean and CV criteria
    # while not np.isclose(np.mean(array), mean, rtol=0.1) or not np.isclose(np.std(array)/np.mean(array), cv, rtol=0.1):
    #     array = np.random.normal(mean, std_dev, num_pxs)
    #     array = np.clip(array, min_val, max_val)

    return array


def create_sphere(radius):
    # Determine the size of the array
    diameter = 2 * radius + 1

    # Initialize the array
    sphere = np.zeros((diameter, diameter, diameter), dtype=bool)

    # Iterate through each point in the array
    for i in range(diameter):
        for j in range(diameter):
            for k in range(diameter):
                # Check if the point is inside the sphere
                if (i - radius)**2 + (j - radius)**2 + (k - radius)**2 <= radius**2:
                    sphere[i, j, k] = True

    return sphere

if __name__ == '__main__':
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"

    model_path = r"D:\test_files\nelly_tests\autoencoder.pt"
    dataset_0, dataset_0_norm = import_data(im_path)
    datasets = [dataset_0_norm, ]
    reconstruction = run_model(model_path, datasets[0])
    # transform back to unnormalized
    reconstruction = (reconstruction * dataset_0.x.std(dim=0, keepdim=True).cpu().numpy() +
                      dataset_0.x.mean(dim=0, keepdim=True).cpu().numpy())

    im_info = ImInfo(im_path)
    reconstructor = Reconstructor(im_info)
    reconstructor.run()

    skel_idxs = np.argwhere(reconstructor.pixel_class > 0)
    assert len(skel_idxs) == len(reconstruction)


    medians = reconstruction[:, 1]
    maxs = reconstruction[:, 2]
    mins = reconstruction[:, 3]
    covs = reconstruction[:, 4]

    test_sphere = create_sphere(2)
    num_true = test_sphere.sum()

    radii = reconstruction[:, 0] / 2
    new_im = np.zeros_like(reconstructor.im_memmap)
    # generate a sphere of radius radii for each point
    spheres = []
    for idx, skel_idx in enumerate(skel_idxs):
        print(f"Processing {idx} of {len(skel_idxs)}")
        int_radius = int(np.round(radii[idx]+1))
        # sphere = viewer.add_points([skel_idx], size=radii[idx]*2, face_color='red', edge_color='red')
        # spheres.append(sphere)
        sphere = create_sphere(int_radius).astype(np.float32)
        true_locs = np.argwhere(sphere)
        fill_array = generate_array(medians[idx], mins[idx], maxs[idx], covs[idx], len(true_locs))
        sphere[true_locs[:, 0], true_locs[:, 1], true_locs[:, 2]] = fill_array
        sphere = sphere.astype(new_im.dtype)
        spheres.append(sphere)
        min_idx = skel_idx - int_radius
        max_idx = skel_idx + int_radius

        min_x = max(0, min_idx[0])
        min_y = max(0, min_idx[1])
        min_z = max(0, min_idx[2])
        max_x = min(new_im.shape[0], max_idx[0])
        max_y = min(new_im.shape[1], max_idx[1])
        max_z = min(new_im.shape[2], max_idx[2])

        x_len = max_x - min_x
        y_len = max_y - min_y
        z_len = max_z - min_z

        # the new_im coords at that location should be the max of the existing value and the new value
        new_im[min_x:max_x, min_y:max_y, min_z:max_z] = np.maximum(new_im[min_x:max_x, min_y:max_y, min_z:max_z], sphere[:x_len, :y_len, :z_len])


    if viewer is None:
        import napari
        viewer = napari.Viewer()
    viewer.add_image(reconstructor.im_memmap * (reconstructor.label_memmap>0))
    #todo actually in reality, I should reconstruct it with the original data too, and compare those to each other.
    viewer.add_image(reconstructor.pixel_class)
    viewer.add_image(new_im)
#