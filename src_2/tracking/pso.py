from src_2.io.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
import numpy as np


class ParticleSwarmOptimization:
    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']

        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']


        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.im_distance_memmap = None
        self.im_marker_memmap = None

        # constants
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.num_particles = 20

        self.debug = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for mocap marking.')
        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

        im_marker_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_marker'])
        self.im_marker_memmap = get_reshaped_image(im_marker_memmap, self.num_t, self.im_info)

        im_distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance_memmap = get_reshaped_image(im_distance_memmap, self.num_t, self.im_info)

    def _get_particles(self, pre_frame_marker_indices):
        particles = []
        for j in range(len(pre_frame_marker_indices)):
            # todo, scale should be based on max travel radius
            point = xp.random.normal(loc=pre_frame_marker_indices[j], scale=3, size=(self.num_particles, 3))
            # convert each point to an int
            point = xp.rint(point).astype(int)
            particles.append(point)
        particles = xp.array(particles)
        return particles

    def _get_pre_values(self, lmp_indices, pre_intensity, pre_distance, pre_frangi):
        # frangi_im = ndi.gaussian_filter(pre_frangi, sigma=0.5)
        # intensities_im = ndi.gaussian_filter(pre_intensity, sigma=0.5)
        # distances_im = ndi.gaussian_filter(pre_distance, sigma=0.5)

        frangi = pre_frangi[lmp_indices[:, 0], lmp_indices[:, 1], lmp_indices[:, 2]]
        intensities = pre_intensity[lmp_indices[:, 0], lmp_indices[:, 1], lmp_indices[:, 2]]
        distances = pre_distance[lmp_indices[:, 0], lmp_indices[:, 1], lmp_indices[:, 2]]
        return frangi, intensities, distances

    def _evaluate_fitness(self, lmp_values, particle_indices, post_intensity, post_distance, post_frangi):
        frangi = post_frangi[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]]
        intensities = post_intensity[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]]
        distances = post_distance[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]]

        frangi_diff = xp.abs(frangi - lmp_values['frangi'])
        intensity_diff = xp.abs(intensities - lmp_values['intensities'])
        distance_diff = xp.abs(distances - lmp_values['distances'])

        z_score_frangi = xp.abs(frangi_diff - xp.mean(frangi_diff)) / xp.std(frangi_diff)
        z_score_intensity = xp.abs(intensity_diff - xp.mean(intensity_diff)) / xp.std(intensity_diff)
        z_score_distance = xp.abs(distance_diff - xp.mean(distance_diff)) / xp.std(distance_diff)

        fitness = -(z_score_frangi + z_score_intensity + z_score_distance)
        return fitness

    def _run_frame(self, t):
        marker_frame = xp.array(self.im_marker_memmap[t - 1] > 0)
        pre_frame_marker_indices = xp.argwhere(marker_frame)
        particles = self._get_particles(pre_frame_marker_indices)

        pre_frangi_frame = xp.array(ndi.gaussian_filter(self.im_frangi_memmap[t - 1], sigma=0.5))
        pre_intensity_frame = xp.array(ndi.gaussian_filter(self.im_memmap[t - 1], sigma=0.5))
        pre_distance_frame = xp.array(ndi.gaussian_filter(self.im_distance_memmap[t - 1], sigma=0.5))

        frangi, intensities, distances = self._get_pre_values(pre_frame_marker_indices,
                                                              pre_intensity_frame,
                                                              pre_distance_frame,
                                                              pre_frangi_frame)

        post_frangi_frame = xp.array(ndi.gaussian_filter(self.im_frangi_memmap[t], sigma=0.5))
        post_intensity_frame = xp.array(ndi.gaussian_filter(self.im_memmap[t], sigma=0.5))
        post_distance_frame = xp.array(ndi.gaussian_filter(self.im_distance_memmap[t], sigma=0.5))



        # viewer.add_points(particles.get(), size=5)
# remove points that are outside the image

            # z = xp.random.randint(low=pre_frame_marker_indices[j, 0]-10, high=pre_frame_marker_indices[j, 0]+10, size=num_particles)
            # y = xp.random.randint(low=pre_frame_marker_indices[j, 1]-10, high=pre_frame_marker_indices[j, 1]+10, size=num_particles)
            # x = xp.random.randint(low=pre_frame_marker_indices[j, 2]-10, high=pre_frame_marker_indices[j, 2]+10, size=num_particles)
            # for i in range(num_particles):
            #     if z[i] > 0 and y[i] > 0 and x[i] > 0 and z[i] < gpu_frame.shape[0] and y[i] < gpu_frame.shape[1] and x[i] < gpu_frame.shape[2]:
            #         particles.append([z[i], y[i], x[i]])
        # post_frame_marker_indices = xp.argwhere(xp.array(self.im_marker_memmap[t] > 0))

        # pre_frangi_im = ndi.gaussian_filter(xp.array(self.im_frangi_memmap[t - 1]), sigma=0.5)
        # pre_intensities_im = ndi.gaussian_filter(xp.array(self.im_memmap[t - 1]), sigma=0.5)
        # pre_distances_im = ndi.gaussian_filter(xp.array(self.im_distance_memmap[t - 1]), sigma=0.5)
        # post_frangi_im = ndi.gaussian_filter(xp.array(self.im_frangi_memmap[t]), sigma=0.5)
        # post_intensities_im = ndi.gaussian_filter(xp.array(self.im_memmap[t]), sigma=0.5)
        # post_distances_im = ndi.gaussian_filter(xp.array(self.im_distance_memmap[t]), sigma=0.5)
        #
        # pre_distances = pre_distances_im[
        #     pre_frame_marker_indices[:, 0], pre_frame_marker_indices[:, 1], pre_frame_marker_indices[:, 2]
        # ]
        # pre_frangi = np.log10(pre_frangi_im[
        #     pre_frame_marker_indices[:, 0], pre_frame_marker_indices[:, 1], pre_frame_marker_indices[:, 2]
        # ])
        # pre_intensities = pre_intensities_im[
        #     pre_frame_marker_indices[:, 0], pre_frame_marker_indices[:, 1], pre_frame_marker_indices[:, 2]
        # ]
        # post_distances = post_distances_im[
        #     post_frame_marker_indices[:, 0], post_frame_marker_indices[:, 1], post_frame_marker_indices[:, 2]
        # ]
        # post_frangi = np.log10(post_frangi_im[
        #     post_frame_marker_indices[:, 0], post_frame_marker_indices[:, 1], post_frame_marker_indices[:, 2]
        # ])
        # post_intensities = post_intensities_im[
        #     post_frame_marker_indices[:, 0], post_frame_marker_indices[:, 1], post_frame_marker_indices[:, 2]
        # ]
        # # distance between pre and post marker
        # diff_matrix = xp.sqrt(
        #     (pre_frame_marker_indices[:, 0] - post_frame_marker_indices[:, 0]) ** 2 +
        #     (pre_frame_marker_indices[:, 1] - post_frame_marker_indices[:, 1]) ** 2 +
        #     (pre_frame_marker_indices[:, 2] - post_frame_marker_indices[:, 2]) ** 2
        # )
        return None

    def _run_pso(self):
        for t in range(1, self.num_t):
            logger.debug(f'Running particle swarm optimization for frame {t + 1} of {self.num_t}')
            self._run_frame(t)

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_pso()


if __name__ == "__main__":
    import os
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('im_instance_label')
        im_info.create_output_path('im_frangi')
        im_info.create_output_path('im_marker')
        im_info.create_output_path('im_distance')
        im_infos.append(im_info)

    pso_files = []
    for im_info in im_infos:
        pso = ParticleSwarmOptimization(im_info, num_t=2)
        pso.run()
        pso_files.append(pso)
