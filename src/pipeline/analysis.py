from tifffile import tifffile

from src.pipeline.organelle_props import OrganellePropertiesConstructor, OrganelleProperties
from src.pipeline.tracking.node_to_node import NodeTrack, Node, NodeConstructor
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
from src import logger
import pandas as pd
import numpy as xp


class StatsNode:
    def __init__(self, node_id, node_track: NodeTrack, region_id):
        self.node_id = node_id
        self.node_track = node_track

        # region and branch identifiers
        self.branch_ids = set()
        self.region_id = region_id

        # this node's properties
        # single frame only
        self.n_distance_from_cell_center = None
        self.n_node_width = None  # the minimum distance to a zero pixel from 3 orthogonal directions
        self.n_intensity_coords = None
        self.n_num_branches = None
        self.n_angles_at_junctions = None
        # previous frame required
        self.n_speed = []
        self.n_distance = []
        self.n_direction = []
        self.n_fission = []
        self.n_fusion = []
    # todo will want to save aggregate values (all stats, mean, median, std, etc.)
    #    self.n_mean_angle_at_junctions = None
    #    self.n_mean_intensity = None
    #    self.n_mean_speed = None
    #    self.n_mean_distance = None
    #    self.n_mean_direction = None
    #    self.n_sum_fission = None
    #    self.n_sum_fusion = None

    def calculate_node_stats(self, frame_num, spacing, intensity_image, cell_centers, mask_image,
                             frame_node_tracks, all_tracks, time_step):
        self.n_distance_from_cell_center = xp.linalg.norm(
            self.node_track.node.centroid_um - cell_centers[frame_num], axis=0)
        self.n_node_width = self._find_node_width(mask_image, spacing)
        node_coords = self.node_track.node.coords
        self.n_intensity_coords = intensity_image[node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]]
        self.n_num_branches = len(self.branch_ids)
        self.n_angles_at_junctions = self._find_angles_at_junctions(frame_node_tracks)
        self._get_motility_metrics(all_tracks, time_step, cell_centers, frame_num)

    def _find_node_width(self, mask_image, spacing):
        # traverse in the given direction from the first coordinate until the intensity image is zero
        def _traverse(coord, direction):
            coord_iteration = xp.array(coord)
            while mask_image[tuple(coord_iteration)] == 1:
                coord_iteration += direction
            return xp.linalg.norm(coord_iteration - xp.array(coord), axis=0)
        # do it for x, y, and z directions
        start_coord = self.node_track.node.coords[0]
        # multiply by 2 to get the diameter in micrometers
        z_dist = _traverse(start_coord, (1, 0, 0)) * 2 * spacing[0]
        y_dist = _traverse(start_coord, (0, 1, 0)) * 2 * spacing[1]
        x_dist = _traverse(start_coord, (0, 0, 1)) * 2 * spacing[2]
        # return the minimum of the three distances
        return min(x_dist, y_dist, z_dist)

    def _find_angles_at_junctions(self, frame_node_tracks):
        track_angles = []
        node = self.node_track.node
        if len(node.connected_nodes) > 1:
            connected_branch_vectors = []
            for connected_node_num in node.connected_nodes:
                connected_node = frame_node_tracks[connected_node_num].node
                branch_vector = xp.array(node.centroid_um) - xp.array(connected_node.centroid_um)
                connected_branch_vectors.append(branch_vector)
            angles = []
            for i in range(len(connected_branch_vectors)):
                for j in range(i + 1, len(connected_branch_vectors)):
                    vec1 = connected_branch_vectors[i]
                    vec2 = connected_branch_vectors[j]
                    cos_theta = xp.dot(vec1, vec2) / (xp.linalg.norm(vec1) * xp.linalg.norm(vec2))
                    angle = xp.arccos(xp.clip(cos_theta, -1, 1)) * 180 / xp.pi
                    angles.append(angle)
            track_angles.extend(angles)
        return track_angles

    def _get_motility_metrics(self, all_tracks, time_step, cell_centers, frame_num):
        for parent_node in self.node_track.parents:
            # get the previous node
            previous_node = all_tracks[parent_node['frame']][parent_node['track']].node

            # get the distance between the previous node and this node
            coord1 = xp.array(self.node_track.node.centroid_um)
            coord2 = xp.array(previous_node.centroid_um)
            self.n_distance.append(xp.linalg.norm(coord1 - coord2, axis=0))
            # get the speed
            self.n_speed.append(self.n_distance[-1] / time_step)
            # get the direction
            coord1 = xp.array(previous_node.centroid_um)
            coord2 = cell_centers[parent_node['frame']]
            cell_vector_1 = xp.linalg.norm(coord1 - coord2)
            coord1 = self.node_track.node.centroid_um
            coord2 = cell_centers[frame_num]
            cell_vector_2 = xp.linalg.norm(coord1 - coord2)
            direction_frame = cell_vector_2 - cell_vector_1
            if direction_frame > 0:
                direction = 1
            elif direction_frame < 0:
                direction = -1
            else:
                direction = 0
            self.n_direction.append(direction)
            # get the fission and fusion
            if len(self.node_track.joins) > 0:
                for join in self.node_track.joins:
                    fusion_frame = 0
                    if join['join_type'] == 'fusion':
                        fusion_frame = 1
                    self.n_fusion.append(fusion_frame)
            else:
                self.n_fusion.append(0)
            if len(self.node_track.splits) > 0:
                for split in self.node_track.splits:
                    fission_frame = 0
                    if split['split_type'] == 'fission':
                        fission_frame = 1
                    self.n_fission.append(fission_frame)
            else:
                self.n_fission.append(0)


class StatsBranch:
    def __init__(self, branch_id, branch_start_coord, region_id):
    # def __init__(self, branch_tuple, branch_info, region_id):
        self.branch_start_coord = branch_start_coord
        self.branch_coords = []
        self.cell_center = None
        self.spacing = None
        self.branch_id = branch_id

        # node and region identifiers
        self.node_ids = set()
        self.region_id = region_id

        # this branch's properties
        self.b_aspect_ratio = None
        self.b_distance_from_cell_center = []
        self.b_circularity = None
        self.b_intensity_coords = None
        self.b_length = None
        self.b_orientation = None
        self.b_orientations = []
        self.b_tortuosity = None
        self.b_widths = None
    # todo will want to save aggregate values (all stats, mean, median, std, etc.)
    #     self.b_distance_from_cell_center_mean = None
    #     self.b_orientations_mean = None
    #     self.b_widths_mean = None

        # todo calculate these
        # node properties for this branch
        self.bn_mean_width = None
        self.bn_mean_speed = None
        self.bn_mean_distance = None
        self.bn_mean_distance_from_cell_center = None
        self.bn_mean_direction = None
        self.bn_mean_fission_num = None
        self.bn_mean_fusion_num = None
        self.bn_mean_num_branches = None
        self.bn_mean_angles_at_junctions = None

    def calculate_branch_stats(self, spacing, intensity_image, cell_center, mask_image, frame_branch_labels):
        self._traverse_and_calculate_length_and_width(frame_branch_labels, mask_image, spacing, cell_center)
        self._calculate_tortuosity(spacing)
        self._calculate_orientation(cell_center, spacing)
        coords = xp.array(self.branch_coords)
        self.b_intensity_coords = intensity_image[coords[:, 0], coords[:, 1], coords[:, 2]]

    def get_node_aggregate_properties(self, tracklets: dict[int: list[NodeTrack]], frame_num: int):
        pass

    def _calculate_tortuosity(self, spacing):
        """Calculate the tortuosity of the branch."""
        coord_1 = self.branch_coords[0]
        coord_2 = self.branch_coords[-1]
        if coord_1 == coord_2:
            self.b_tortuosity = 1
        else:
            self.b_tortuosity = self.b_length / self._euclidean_distance(
                self.branch_coords[0], self.branch_coords[-1], spacing)

    def _euclidean_distance(self, coord1, coord2, spacing):
        """Calculate the euclidean distance between two points."""
        return xp.sqrt(xp.sum((xp.array(coord1) * spacing - xp.array(coord2) * spacing) ** 2))

    def _neighbors_in_volume(self, volume, check_coord):
        """Find the coordinates of the neighboring points within the volume."""
        z, y, x = check_coord
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == dy == dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nx < volume.shape[2]:
                        if volume[nz, ny, nx]:
                            if (nz, ny, nx) not in self.branch_coords:
                                neighbors.append((nz, ny, nx))
                                self.branch_coords.append((nz, ny, nx))
        return neighbors

    def _traverse_and_calculate_length_and_width(self, volume, mask, spacing, cell_center):
        total_length = 0
        widths = []
        current_point = xp.array(self.branch_start_coord)
        self.branch_coords.append(tuple(current_point))
        self.spacing = spacing

        while True:
            neighbor = self._neighbors_in_volume(volume, tuple(current_point))
            if not neighbor:
                # calculate the width for that pixel in any direction
                max_distance1 = self._search_along_direction(mask, current_point, xp.array([1, 0, 0]), spacing)
                max_distance2 = self._search_along_direction(mask, current_point, xp.array([0, 1, 0]), spacing)
                width = max(max_distance1, max_distance2) * 2
                # set the length equal to the width
                total_length += width
                widths.append(width)
                break
            neighbor = neighbor[0]
            # closest_point = min(neighbor, key=lambda x: xp.linalg.norm(current_point - xp.array(x)))

            # Calculate the distance between the current point and the closest point
            segment_length = xp.linalg.norm(current_point * spacing - xp.array(neighbor) * spacing)
            total_length += segment_length

            # Calculate the direction vector of the segment
            direction_vector = xp.array(neighbor) - current_point
            direction_vector = direction_vector / xp.linalg.norm(direction_vector)

            # Find two orthogonal vectors
            orthogonal_vector1 = xp.cross(direction_vector, xp.array([1, 0, 0]))
            if xp.linalg.norm(orthogonal_vector1) == 0:
                orthogonal_vector1 = xp.cross(direction_vector, xp.array([0, 1, 0]))

            orthogonal_vector1 = orthogonal_vector1 / xp.linalg.norm(orthogonal_vector1)
            orthogonal_vector2 = xp.cross(direction_vector, orthogonal_vector1)

            # Find the width of the branch at the current point
            max_distance1 = self._search_along_direction(mask, current_point, orthogonal_vector1, spacing)
            max_distance2 = self._search_along_direction(mask, current_point, orthogonal_vector2, spacing)
            width = max(max_distance1, max_distance2) * 2
            widths.append(width)

            # Calculate the orientation at the current point
            current_orientation, current_distance_from_cell_center = self._calculate_point_orientation(
                current_point * spacing,
                xp.array(neighbor) * spacing,
                cell_center)
            self.b_orientations.append(current_orientation)
            self.b_distance_from_cell_center.append(current_distance_from_cell_center)

            # Set the closest point as the new current point and remove it from the volume
            current_point = xp.array(neighbor)

        self.b_length = total_length
        self.b_widths = widths
        self.b_circularity = xp.mean(widths) / self.b_length
        self.b_aspect_ratio = 1 / self.b_circularity

    def _search_along_direction(self, mask, start_point, direction, spacing):
        max_distance = 0
        i = 0
        while True:
            # Search in positive direction
            search_point_pos = start_point + i * direction
            if not self._is_in_bounds_and_zero(mask, search_point_pos):
                distance_pos = xp.linalg.norm(start_point * spacing - search_point_pos * spacing)
                max_distance = max(max_distance, distance_pos)
                break
            # Search in negative direction
            search_point_neg = start_point - i * direction
            if not self._is_in_bounds_and_zero(mask, search_point_neg):
                distance_neg = xp.linalg.norm(start_point * spacing - search_point_neg * spacing)
                max_distance = max(max_distance, distance_neg)
                break
            i += 1
        return max_distance

    @staticmethod
    def _is_in_bounds_and_zero(mask, search_point):
        nz, ny, nx = tuple(search_point)
        if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]:
            return mask[int(nz), int(ny), int(nx)]
        return False

    def _calculate_orientation(self, cell_center, spacing):
        """Calculate the orientation of the branch with respect to the given cell center."""
        self.cell_center = cell_center
        if len(self.branch_coords) < 2:
            self.b_orientation = xp.nan
        else:
            start_coord = xp.array(self.branch_coords[0]) * spacing
            end_coord = xp.array(self.branch_coords[-1]) * spacing
            cell_center = xp.array(cell_center)

            # Calculate the branch vector
            branch_vector = end_coord - start_coord
            branch_vector_normalized = branch_vector / xp.linalg.norm(branch_vector)

            # Calculate the vector from the cell center to the start point of the branch
            cell_center_to_start_vector = start_coord - cell_center
            cell_center_to_start_vector_normalized = cell_center_to_start_vector / xp.linalg.norm(
                cell_center_to_start_vector)

            # Calculate the orientation as the angle between the two vectors (in radians)
            dot_product = xp.dot(branch_vector_normalized, cell_center_to_start_vector_normalized)
            angle_rad = xp.arccos(dot_product)

            # Convert the angle from radians to degrees and restrict the range to [0, 180]
            angle_deg = xp.degrees(angle_rad)
            self.b_orientation = min(angle_deg, 180 - angle_deg)

    @staticmethod
    def _calculate_point_orientation(start_coord, end_coord, cell_center):
        """Calculate the orientation at the current point with respect to the given cell center."""
        if start_coord is None or end_coord is None:
            return xp.nan

        # Calculate the branch vector
        branch_vector = end_coord - start_coord
        branch_vector_normalized = branch_vector / xp.linalg.norm(branch_vector)

        # Calculate the vector from the cell center to the start point of the branch
        cell_center_to_start_vector = start_coord - cell_center
        cell_center_to_start_vector_normalized = cell_center_to_start_vector / xp.linalg.norm(cell_center_to_start_vector)

        # Calculate the orientation as the angle between the two vectors (in radians)
        dot_product = xp.dot(branch_vector_normalized, cell_center_to_start_vector_normalized)
        angle_rad = xp.arccos(dot_product)

        # Convert the angle from radians to degrees and restrict the range to [0, 180]
        angle_deg = xp.degrees(angle_rad)
        orientation = min(angle_deg, 180 - angle_deg)

        return orientation, xp.abs(cell_center_to_start_vector)


class StatsRegion:
    def __init__(self, organelle: OrganelleProperties, region_id):
        self.r_centroid = organelle.centroid
        self.r_coords = organelle.coords
        self.r_skeleton_coords = organelle.skeleton_coords
        self.r_instance_label = organelle.instance_label
        self.region_id = region_id


        # node and branch identifiers
        self.node_ids = set()
        self.branch_ids = set()

        # region properties for this region
        self.r_scaled_coords = None
        self.r_intensity_coords = None
        self.r_volume = None
        self.r_distance_from_cell_center_coords = None

        # todo make sure everything below here is correct, and calculate them
        # branch properties for this region
        self.rb_num_branches = None
        self.rb_total_branch_length = None
        self.rb_mean_branch_aspect_ratio = None
        self.rb_weighted_branch_aspect_ratio = None
        self.rb_mean_branch_distance_from_cell_center = None
        self.rb_weighted_branch_distance_from_cell_center = None
        self.rb_mean_circularity = None
        self.rb_weighted_circularity = None
        self.rb_mean_length = None
        self.rb_weighted_width = None
        self.rb_mean_width = None
        self.rb_weighted_orientation = None
        self.rb_mean_orientation = None
        self.rb_weighted_tortuosity = None
        self.rb_mean_tortuosity = None

        # node properties for this region
        self.rn_mean_width = None
        self.rn_mean_speed = None
        self.rn_mean_distance = None
        self.rn_mean_distance_from_cell_center = None
        self.rn_mean_direction = None
        self.rn_mean_fission_num = None
        self.rn_mean_fusion_num = None
        self.rn_mean_num_branches = None
        self.rn_mean_angles_at_junctions = None

    def calculate_region_stats(self, spacing, intensity_image, cell_center):
        self.r_scaled_coords = self.r_coords * spacing
        self.r_intensity_coords = intensity_image[self.r_coords[:, 0], self.r_coords[:, 1], self.r_coords[:, 2]]
        self.r_volume = len(self.r_coords) * xp.prod(spacing)
        self.r_distance_from_cell_center_coords = xp.linalg.norm(self.r_scaled_coords - cell_center, axis=1)

    def get_branch_aggregate_properties(self):
        self.rb_num_branches = len(self.branch_objects)
        if self.rb_num_branches == 0:
            return
        self.rb_total_branch_length = sum([branch.b_length for branch in self.branch_objects.values()])
        self.rb_weighted_branch_aspect_ratio = sum([branch.b_aspect_ratio * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_branch_aspect_ratio = sum([branch.b_aspect_ratio for branch in self.branch_objects.values()]) / self.rb_num_branches
        self.rb_weighted_circularity = sum([branch.b_circularity * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_circularity = sum([branch.b_circularity for branch in self.branch_objects.values()]) / self.rb_num_branches
        self.rb_mean_branch_distance_from_cell_center = sum([branch.b_distance_from_cell_center_mean for branch in self.branch_objects.values()]) / self.rb_num_branches
        self.rb_weighted_branch_distance_from_cell_center = sum([branch.b_distance_from_cell_center_mean * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_length = self.rb_total_branch_length / self.rb_num_branches
        self.rb_weighted_width = sum([sum(branch.b_widths) * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_width = sum([branch.b_widths_mean for branch in self.branch_objects.values()]) / self.rb_num_branches
        self.rb_weighted_orientation = sum([branch.b_orientation * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_orientation = sum([branch.b_orientation for branch in self.branch_objects.values()]) / self.rb_num_branches
        self.rb_weighted_tortuosity = sum([branch.b_tortuosity * branch.b_length for branch in self.branch_objects.values()]) / self.rb_total_branch_length
        self.rb_mean_tortuosity = sum([branch.b_tortuosity for branch in self.branch_objects.values()]) / self.rb_num_branches

    def get_node_aggregate_properties(self, tracklets: dict[int: list[NodeTrack]], frame_num: int):
        pass



class AnalysisHierarchyConstructor:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info

        # load in regions (organelles), and tracks, which have node and branch info.
        organelle_props: OrganellePropertiesConstructor = unpickle_object(self.im_info.path_pickle_obj)
        track_props: dict[int: list[NodeTrack]] = unpickle_object(self.im_info.path_pickle_track)
        node_props: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.tracks = track_props
        self.node_props = node_props
        self.organelles = organelle_props.organelles

        # get the voxel spacing
        self.spacing = [im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X']]
        self.time_step = im_info.dim_sizes['T']

        # construct dicts for hierarchy
        self.stats_region = dict()
        self.stats_branches = dict()
        self.stats_nodes = dict()

        # get the cell center for each frame
        self.cell_center = dict()
        self._calculate_cell_center()

    def _calculate_cell_center(self):
        for frame_num, frame_nodes in self.node_props.nodes.items():
            self.cell_center[frame_num] = xp.mean(xp.array([node.centroid_um for node in frame_nodes]), axis=0)

    def get_hierarchy(self):
        # construct hierarchy objects
        self._construct_region_objects()
        self._construct_node_and_branch_objects()
        # calculate stats for each object
        # todo account for when there is no intensity image
        intensity_image = self.im_info.get_im_memmap(self.im_info.im_path)
        mask_image = self.im_info.get_im_memmap(self.im_info.path_im_mask)
        branch_label_image = self.im_info.get_im_memmap(self.im_info.path_im_label_seg)
        self._calculate_region_stats(intensity_image)
        self._calculate_branch_stats(intensity_image, mask_image, branch_label_image)
        self._calculate_node_stats(intensity_image, mask_image)

        # self._calculate_metrics()
        # self._assign_nodes()
        # self._get_branch_stats()
        # self._clean_branch_stats()
        # self._get_mean_stats(tracklets)

    def _construct_region_objects(self):
        for frame_num, organelles in self.organelles.items():
            logger.info(f'Constructing region objects for frame {frame_num}')
            self.stats_region[frame_num] = dict()
            for organelle in organelles:
                region_id = organelle.instance_label
                region = StatsRegion(organelle, region_id)
                self.stats_region[frame_num][region_id] = region

    def _construct_node_and_branch_objects(self):
        for frame_num, frame_tracks in self.tracks.items():
            logger.info(f'Constructing node and branch objects for frame {frame_num}')
            self.stats_nodes[frame_num] = dict()
            self.stats_branches[frame_num] = dict()
            for node_id, track in enumerate(frame_tracks):
                node_object = StatsNode(node_id, track, track.node.skeleton_label)
                self.stats_nodes[frame_num][node_id] = node_object
                for branch_id, branch_start_coord in track.node.connected_branches:
                    if self.stats_branches[frame_num].get(branch_id) is None:
                        self.stats_branches[frame_num][branch_id] = StatsBranch(
                            branch_id, branch_start_coord, track.node.skeleton_label)
                    self.stats_branches[frame_num][branch_id].node_ids.add(node_id)
                    self.stats_nodes[frame_num][node_id].branch_ids.add(branch_id)

    def _calculate_region_stats(self, intensity_image):
        for frame_num, frame_regions in self.stats_region.items():
            logger.info(f'Calculating region stats for frame {frame_num}')
            for region in frame_regions.values():
                region.calculate_region_stats(self.spacing, intensity_image[frame_num], self.cell_center[frame_num])

    def _calculate_branch_stats(self, intensity_image, mask_image, branch_label_image):
        for frame_num, frame_branches in self.stats_branches.items():
            logger.info(f'Calculating branch stats for frame {frame_num}')
            for branch in frame_branches.values():
                branch.calculate_branch_stats(self.spacing, intensity_image[frame_num], self.cell_center[frame_num],
                                              mask_image[frame_num], branch_label_image[frame_num])

    def _calculate_node_stats(self, intensity_image, mask_image):
        for frame_num, frame_nodes in self.stats_nodes.items():
            logger.info(f'Calculating node stats for frame {frame_num}')
            for node in frame_nodes.values():
                node.calculate_node_stats(frame_num, self.spacing, intensity_image[frame_num], self.cell_center,
                                          mask_image[frame_num], self.tracks[frame_num], self.tracks, self.time_step)

    def _get_mean_stats(self, tracklets):
        for frame_num, region_dict in self.regions.items():
            for region_num, region in region_dict.items():
                region.get_branch_aggregate_properties()
                if frame_num == 0:
                    continue
                region.get_node_aggregate_properties(tracklets[frame_num])

    def _clean_branch_stats(self):
        for frame_num, frame_regions in self.regions.items():
            for region_num, region in frame_regions.items():
                for branch_num, branch_list in region.branch_objects.items():
                    node_set = set()
                    for branch in branch_list:
                        node_set.add(branch.branch_tuple[0])
                        node_set.add(branch.branch_tuple[1])
                    for branch in branch_list:
                        branch.node_ids = node_set
                    if len(branch_list) >= 2:
                        longest_branch = max(branch_list, key=lambda x: x.b_length)
                        remove_branches = []
                        for branch in branch_list:
                            if branch is not longest_branch:
                                remove_branches.append(branch)
                        for branch in remove_branches:
                            self.regions[frame_num][region_num].branch_objects[branch_num].remove(branch)
                    region.branch_objects[branch_num] = region.branch_objects[branch_num][0]

    def _get_branch_stats(self):
        branch_labels = tifffile.memmap(self.im_info.path_im_label_seg, mode='r') > 0
        mask_im = tifffile.memmap(self.im_info.path_im_mask, mode='r') > 0
        for frame_num, frame_regions in self.regions.items():
            for region_num, region in frame_regions.items():
                for branch_tuple, branch_infos in region.branch_ids.items():
                    for branch_info in branch_infos:
                        branch_object = StatsBranch(frame_num, branch_tuple, branch_info)
                        branch_label = branch_info[0]
                        branch_object._traverse_and_calculate_length_and_width(
                            branch_labels[frame_num], mask_im[frame_num], self.spacing, self.cell_center[frame_num])
                        branch_object._calculate_tortuosity(self.spacing)
                        branch_object._calculate_orientation(self.cell_center[frame_num], self.spacing)
                        if region.branch_objects.get(branch_label) is None:
                            region.branch_objects[branch_label] = []
                        region.branch_objects[branch_label].append(branch_object)

    def _assign_nodes(self):
        # todo remove frame cap after testing
        for frame_num, nodes in self.node_props.nodes.items():
            if frame_num > 1:
                continue
            logger.debug(f'Assigning nodes to regions for frame {frame_num}/{len(self.node_props.nodes.items())}')
            for node_num, node in enumerate(nodes):
                branches = dict()
                for connected_node_num in node.connected_nodes:
                    connected_node = self.node_props.nodes[frame_num][connected_node_num]
                    node_pair = [node_num, connected_node_num]
                    node_pair.sort()
                    node_pair = tuple(node_pair)
                    if node_pair not in self.regions[frame_num][node.skeleton_label].branch_ids:
                        connected_node_labels = [branch[0] for branch in connected_node.connected_branches]
                        connected_branches = [branch for branch in node.connected_branches
                                              if branch[0] in connected_node_labels]
                        branches[node_pair] = connected_branches
                if node_num not in self.regions[frame_num][node.skeleton_label].node_ids:
                    self.regions[frame_num][node.skeleton_label].node_ids.append(node_num)
                self.regions[frame_num][node.skeleton_label].branch_ids.update(branches)


class StatsDynamics:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.tracklets: dict[int: list[NodeTrack]] = unpickle_object(self.im_info.path_pickle_track)
        self.tracks = []
        self.time_between_volumes = self.im_info.dim_sizes['T']
        self.node_track_dict = None

        root_node_tracks = []
        for frame_num, track_frames in self.tracklets.items():
            for tracklet in track_frames:
                if tracklet.parents:
                    continue
                else:
                    root_node_tracks.append(tracklet)

        for tracklet in root_node_tracks:
            tracks = self.build_tracks(tracklet)
            self.tracks.extend(tracks)

    def build_tracks(self, node_track, current_track=None, all_tracks=None):
        if current_track is None:
            current_track = [node_track]
        else:
            current_track.append(node_track)

        if all_tracks is None:
            all_tracks = []

        if not node_track.children:
            all_tracks.append(current_track)
        else:
            for child in node_track.children:
                child_node = self.tracklets[child['frame']][child['track']]
                self.build_tracks(child_node, current_track.copy(), all_tracks)

        return all_tracks

    # def node_num_track_dict(self):
    #     # create a dictionary where the key is the frame number and the value is a dictionary
    #     # where the key is the node numbers and value is a list of tracks that contain that node
    #     node_num_track_dict = dict()
    #     for frame_num, track_frames in self.tracklets.items():
    #         node_num_track_dict[frame_num] = dict()
    #         for node_num, track in enumerate(track_frames):
    #             if node_num not in node_num_track_dict[frame_num]:
    #                 node_num_track_dict[frame_num][node_num] = []
    #             node_num_track_dict[frame_num][node_num].append(track)
    #     self.node_track_dict = node_num_track_dict


class AnalysisDynamics:
    def __init__(self, im_info: ImInfo, tracks: list[list[NodeTrack]]):
        self.im_info = im_info
        node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: dict[int: list[Node]] = node_constructor.nodes
        self.cell_center = []
        self.calculate_cell_center()
        self.tracks = tracks
        self.metrics = []

    def calculate_cell_center(self):
        for frame_num, frame_nodes in self.nodes.items():
            node_centroids = []
            for node in frame_nodes:
                node_centroids.append(node.centroid_um)
            self.cell_center.append(xp.mean(xp.array(node_centroids), axis=0))

    def calculate_metrics(self):
        for track in self.tracks:
            if len(track) < 2:
                continue
            track_metrics = {}
            track_metrics['frame'] = xp.array([node.frame_num for node in track])
            track_metrics['distance'] = xp.array(self.calculate_distance(track))
            track_metrics['displacement'] = xp.array(self.calculate_displacement(track))
            track_metrics['speed'] = xp.array(self.calculate_speed(track))
            track_metrics['direction'] = xp.array(self.calculate_direction(track))
            track_metrics['fission'], track_metrics['fusion'] = self.calculate_dynamics(track)
            track_metrics['num_branches'] = xp.array(self.calculate_num_branches(track))
            track_metrics['persistance'] = xp.array(self.calculate_persistance(track))
            track_metrics['angles_at_junctions'] = xp.array(self.calculate_angles_at_junctions(track))
            self.metrics.append(track_metrics)

    def save_metrics_to_csv(self, aggregate_output_file, frame_output_file):
        self.save_aggregate_metrics_to_csv(aggregate_output_file)
        self.save_frame_metrics_to_csv(frame_output_file)

    def save_aggregate_metrics_to_csv(self, output_file):
        df = pd.DataFrame(self.metrics)
        aggregate_df = pd.DataFrame()
        aggregate_df['num_frames_tracked'] = df['frame'].apply(lambda x: len(x))
        for metric_name in df.columns:
            metric_metrics = df[metric_name]
            if metric_name == 'frame':
                continue
            if metric_name == 'fission' or metric_name == 'fusion':
                aggregate_df[f'{metric_name}_sum'] = metric_metrics.apply(lambda x: xp.nansum(x))
                continue
            aggregate_df[f'{metric_name}_median'] = metric_metrics.apply(lambda x: xp.nanmedian(x))
            aggregate_df[f'{metric_name}_quartiles25'] = metric_metrics.apply(lambda x: xp.nanquantile(x, 0.25))
            aggregate_df[f'{metric_name}_quartiles75'] = metric_metrics.apply(lambda x: xp.nanquantile(x, 0.75))
            aggregate_df[f'{metric_name}_mean'] = metric_metrics.apply(lambda x: xp.nanmean(x))
            aggregate_df[f'{metric_name}_std'] = metric_metrics.apply(lambda x: xp.nanstd(x))
            aggregate_df[f'{metric_name}_max'] = metric_metrics.apply(lambda x: xp.nanmax(x))
            aggregate_df[f'{metric_name}_min'] = metric_metrics.apply(lambda x: xp.nanmin(x))
            aggregate_df[f'{metric_name}_sum'] = metric_metrics.apply(lambda x: xp.nansum(x))
        aggregate_df.to_csv(output_file)

    def save_frame_metrics_to_csv(self, output_folder):
        df = pd.DataFrame(self.metrics)
        frame_data = []
        output_file = os.path.join(output_folder, f'{self.im_info.filename}_frame_metrics.csv')

        for metric_name in df.columns:
            if metric_name == 'frame':
                continue

            for track_id, frame_metrics in enumerate(df[metric_name]):
                for idx, value in enumerate(frame_metrics):
                    if xp.isnan(value):
                        frame_data.append([track_id, metric_name, 'NaN', df['frame'][track_id][idx]])
                    else:
                        frame_data.append([track_id, metric_name, value, df['frame'][track_id][idx]])

        frame_df = pd.DataFrame(frame_data, columns=['track_id', 'metric_name', 'value', 'frame_num'])
        frame_df = frame_df.pivot_table(index=['track_id', 'frame_num'], columns='metric_name', values='value').reset_index()
        frame_df.to_csv(output_file, index=False)

    def calculate_distance(self, track):
        distance = [xp.nan]
        for i in range(1, len(track)):
            distance.append(self.distance_between_nodes(track[i - 1].node, track[i].node))
        return distance

    def calculate_displacement(self, track):
        start_node = track[0].node
        displacement = [xp.nan]
        if len(track) == 1:
            return displacement
        for frame_node in track[1:]:
            displacement.append(self.distance_between_nodes(start_node, frame_node.node))
        return displacement

    def calculate_speed(self, track):
        distance = self.calculate_distance(track)
        time = (len(track) - 1) * self.im_info.dim_sizes['T']
        if time == 0:
            return [xp.nan]
        speed = xp.array(distance) / time
        speed[0] = xp.nan
        return speed

    def calculate_direction(self, track):
        direction = [xp.nan]
        if len(track) == 1:
            return direction
        for frame_num, node_track in enumerate(track[1:]):
            coord1 = xp.array(track[frame_num-1].node.centroid_um)
            coord2 = self.cell_center[track[frame_num-1].frame_num]
            cell_vector_1 = xp.linalg.norm(coord1 - coord2)
            coord1 = xp.array(track[frame_num].node.centroid_um)
            coord2 = self.cell_center[track[frame_num].frame_num]
            cell_vector_2 = xp.linalg.norm(coord1 - coord2)
            direction_frame = cell_vector_2 - cell_vector_1
            if direction_frame > 0:
                direction.append(1)
            elif direction_frame < 0:
                direction.append(-1)
            else:
                direction.append(0)
        return direction

    def calculate_dynamics(self, track):
        fission = [xp.nan]
        fusion = [xp.nan]
        if len(track) == 1:
            return fission, fusion
        for frame_num, node_track in enumerate(track[1:]):
            fusion_frame = 0
            if len(node_track.joins) > 0:
                for join in node_track.joins:
                    if join['join_type'] == 'fusion':
                        fusion_frame = 1
                        break
            fusion.append(fusion_frame)
            fission_frame = 0
            if len(node_track.splits) > 0:
                for split in node_track.splits:
                    if split['split_type'] == 'fission':
                        fission_frame = 1
                        break
            fission.append(fission_frame)
        return xp.array(fission), xp.array(fusion)

    def calculate_num_branches(self, track):
        connection_list = []
        for node_track in track:
            connected_nodes = set(node_track.node.connected_nodes)
            if node_track.node.instance_label in connected_nodes:
                connected_nodes.remove(node_track.node.instance_label)
            connection_list.append(len(connected_nodes))
        return connection_list

    def calculate_persistance(self, track):
        displacement = self.calculate_displacement(track)
        distance = self.calculate_distance(track)
        sum_distance = 0
        persistence = [xp.nan]
        for i in range(1, len(track)):
            sum_distance += distance[i]
            # first step is always of unit persistance, so no information gained
            if i == 1:
                persistence.append(xp.nan)
            elif sum_distance == 0:
                persistence.append(0)
            else:
                persistence.append(displacement[i] / sum_distance)
        return persistence

    def calculate_angles_at_junctions(self, track):
        angles = []
        for node_track in track:
            track_angles = []
            if len(node_track.node.connected_branches) > 2:
                angle_list = self.calculate_angles_for_junction(node_track)
                track_angles.extend(angle_list)
            angles.append(xp.nanmean(track_angles))
        return angles

    # Helper functions
    def distance_between_nodes(self, node1, node2):
        coord1 = xp.array(node1.centroid_um)
        coord2 = xp.array(node2.centroid_um)
        distance = xp.linalg.norm(coord1 - coord2)
        return distance

    def calculate_angles_for_junction(self, track):
        junction_node = track.node
        # TODO calculates straight angle branch.. maybe should change to closer angle
        connected_branch_vectors = []
        for connected_node_num in junction_node.connected_nodes:
            connected_node = self.nodes[track.frame_num][connected_node_num]
            branch_vector = xp.array(junction_node.centroid_um) - xp.array(connected_node.centroid_um)
            connected_branch_vectors.append(branch_vector)

        angles = []
        for i in range(len(connected_branch_vectors)):
            for j in range(i + 1, len(connected_branch_vectors)):
                angle = self.angle_between_vectors(connected_branch_vectors[i], connected_branch_vectors[j])
                angles.append(angle)
        return angles

    def angle_between_vectors(self, vec1, vec2):
        cos_theta = xp.dot(vec1, vec2) / (xp.linalg.norm(vec1) * xp.linalg.norm(vec2))
        angle = xp.arccos(xp.clip(cos_theta, -1, 1)) * 180 / xp.pi
        return angle


if __name__ == '__main__':
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    track_builder = StatsDynamics(test)
    # track_builder.node_num_track_dict()
    # dynamics = AnalysisDynamics(test, track_builder.tracks)
    # dynamics.calculate_metrics()
    # aggregate_output_file = 'aggregate_metrics.csv'
    # frame_output_folder = test.output_csv_dirpath
    # if not os.path.exists(frame_output_folder):
    #     os.makedirs(frame_output_folder)
    # analysis.save_metrics_to_csv(os.path.join(frame_output_folder, aggregate_output_file), frame_output_folder)
    hierarchy = AnalysisHierarchyConstructor(test)
    hierarchy.get_hierarchy()
    # regions.calculate_metrics()
