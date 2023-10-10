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
        self.num_particles = 100

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

    def _get_frame_values(self, indices, intensity, distance, frangi):
        frangi = frangi[indices[:, 0], indices[:, 1], indices[:, 2]]
        intensities = intensity[indices[:, 0], indices[:, 1], indices[:, 2]]
        distances = distance[indices[:, 0], indices[:, 1], indices[:, 2]]
        return frangi, intensities, distances

    def _evaluate_fitness(self, lmp_frangi, lmp_intensity, lmp_distance, particle_indices, post_intensity, post_distance, post_frangi):
        frangi, intensities, distances = self._get_frame_values(particle_indices, post_intensity, post_distance, post_frangi)

        frangi[frangi == 0] = xp.nan
        intensities[intensities == 0] = xp.nan
        distances[distances == 0] = xp.nan
        lmp_frangi[lmp_frangi == 0] = xp.nan
        lmp_intensity[lmp_intensity == 0] = xp.nan
        lmp_distance[lmp_distance == 0] = xp.nan

        frangi_diff = xp.abs(frangi - lmp_frangi)
        intensity_diff = xp.abs(intensities - lmp_intensity)
        distance_diff = xp.abs(distances - lmp_distance)

        z_score_frangi = xp.abs(frangi_diff - xp.mean(frangi_diff)) / xp.std(frangi_diff)
        z_score_intensity = xp.abs(intensity_diff - xp.mean(intensity_diff)) / xp.std(intensity_diff)
        z_score_distance = xp.abs(distance_diff - xp.mean(distance_diff)) / xp.std(distance_diff)

        # fitness = -(z_score_frangi + z_score_distance)
        fitness = -(z_score_frangi + z_score_intensity + z_score_distance)
        return fitness

    def _update_particles(self, lmp_idxs, lmp_frangi, lmp_intensity, lmp_distance,
                          particle_idxs, post_frangi_frame, post_intensity_frame, post_distance_frame):
        self.num_iterations = 100
        particle_list = []
        for lmp_num, lmp_idx in enumerate(lmp_idxs):
            logger.debug(f'Updating particles for marker {lmp_num} of {len(lmp_idxs)}')
            # initialize
            for particle_num, particle_idx in enumerate(particle_idxs[lmp_num]):
                # clip
                particle_idx = xp.clip(particle_idx, 0, xp.array(self.shape[1:]))
                particle_idxs[lmp_num][particle_num] = particle_idx
            lmp_particles = particle_idxs[lmp_num]
            personal_best_idxs = lmp_particles.copy()
            local_fitnesses_last = xp.array(self._evaluate_fitness(
                    lmp_frangi[lmp_num], lmp_intensity[lmp_num], lmp_distance[lmp_num], particle_idxs[lmp_num],
                    post_intensity_frame, post_distance_frame, post_frangi_frame
                ))
            local_particle_list = []
            for i in range(self.num_iterations):
                local_particle_list.append(lmp_particles.copy())
                local_fitnesses = xp.array(self._evaluate_fitness(
                    lmp_frangi[lmp_num], lmp_intensity[lmp_num], lmp_distance[lmp_num], particle_idxs[lmp_num],
                    post_intensity_frame, post_distance_frame, post_frangi_frame
                ))
                personal_best_idxs_comparison = xp.where(local_fitnesses > local_fitnesses_last, 1, 0)
                personal_best_idxs[personal_best_idxs_comparison == 1] = lmp_particles[personal_best_idxs_comparison == 1]
                # test = xp.where(local_fitnesses > local_fitnesses_last, 0, 1)
                global_best_idx = lmp_particles[xp.argmax(local_fitnesses)]
                # lmp_values = lmp_values_all[lmp_num]
                for particle_num, particle_idx in enumerate(lmp_particles):
                    # clip to image bounds
                    new_velocity = self.w * xp.random.rand(3) + \
                                   self.c1 * xp.random.rand() * (personal_best_idxs[particle_num] - particle_idx) + \
                                   self.c2 * xp.random.rand() * (global_best_idx - particle_idx)
                    new_position = particle_idx + new_velocity
                    new_position = xp.rint(new_position).astype(int)
                    lmp_particles[particle_num] = xp.clip(new_position, 0, xp.array(self.shape[1:]))

                particle_idxs[lmp_num] = lmp_particles
                local_fitnesses_last = local_fitnesses

            particle_list.append(local_particle_list)

        return particle_idxs, particle_list


    def _run_frame(self, t):
        marker_frame = xp.array(self.im_marker_memmap[t - 1] > 0)
        pre_frame_marker_indices = xp.argwhere(marker_frame)

        pre_mask = xp.array(self.im_frangi_memmap[t - 1]) > 0
        pre_frangi_frame = ndi.gaussian_filter(xp.array(self.im_frangi_memmap[t - 1]).astype('float'), sigma=0.5) * pre_mask
        pre_intensity_frame = ndi.gaussian_filter(xp.array(self.im_memmap[t - 1]).astype('float'), sigma=0.5) * pre_mask
        pre_distance_frame = ndi.gaussian_filter(xp.array(self.im_distance_memmap[t - 1]).astype('float'), sigma=0.5) * pre_mask
        lmp_frangi, lmp_intensity, lmp_distance = self._get_frame_values(
            pre_frame_marker_indices,
            pre_intensity_frame,
            pre_distance_frame,
            pre_frangi_frame)

        pre_frame_marker_indices = pre_frame_marker_indices[:3]  # todo for testing
        particles = self._get_particles(pre_frame_marker_indices)

        post_mask = xp.array(self.im_frangi_memmap[t]) > 0
        post_frangi_frame = ndi.gaussian_filter(xp.array(self.im_frangi_memmap[t]).astype('float'), sigma=0.5) * post_mask
        post_intensity_frame = ndi.gaussian_filter(xp.array(self.im_memmap[t]).astype('float'), sigma=0.5) * post_mask
        post_distance_frame = ndi.gaussian_filter(xp.array(self.im_distance_memmap[t]).astype('float'), sigma=0.5) * post_mask
        particles2, particle_list = self._update_particles(
            pre_frame_marker_indices, lmp_frangi, lmp_intensity, lmp_distance,
            particles.copy(), post_frangi_frame, post_intensity_frame, post_distance_frame
        )
        new_particle_list = [p.get() for i, p in enumerate(particle_list[1])]
        all_particles = []
        for frame_num, frame in enumerate(new_particle_list[:20]):
            for idx_num, idx in enumerate(frame):
                idx = np.append(frame_num, idx)
                all_particles.append(idx)

        import napari
        viewer = napari.Viewer()
        viewer.add_points(pre_frame_marker_indices.get(), size=2, face_color='green')
        viewer.add_points(particle_list[1][20].get(), size=2, face_color='red')
        viewer.add_points(particle_list[1][-1].get(), size=2, face_color='blue')
        viewer.add_points(all_particles, size=2, face_color='red')
        # viewer.add_points(particles2[1].get(), size=5, face_color='red')
        # viewer.add_points(particles[1].get(), size=5, face_color='blue')
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
