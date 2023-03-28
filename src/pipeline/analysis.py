from tifffile import tifffile

from src.pipeline.organelle_props import OrganellePropertiesConstructor, OrganelleProperties
from src.pipeline.tracking.node_to_node import NodeTrack, Node, NodeConstructor
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
from src import logger
import csv
import pandas as pd
import numpy as xp
import os
from skimage import measure
import scipy.ndimage as ndi

# todo something is messing up with tip-tip branches I think

class Branch:
    def __init__(self, frame_num, branch_tuple, branch_info):
        self.frame_num = frame_num
        self.branch_tuple = branch_tuple
        self.branch_label = branch_info[0]
        self.start_coord = branch_info[1]
        self.branch_coords = []
        self.length = None
        self.tortuosity = None
        self.orientation = None
        self.orientations = []
        self.orientations_mean = None
        self.widths = None
        self.widths_mean = None
        self.tortuosity_times_num_voxels = None  # could be useful for summing tortuosity over entire tree
        self.aspect_ratio = None
        self.circularity = None

    def calculate_tortuosity(self, spacing):
        """Calculate the tortuosity of the branch."""
        coord_1 = self.branch_coords[0]
        coord_2 = self.branch_coords[-1]
        if coord_1 == coord_2:
            self.tortuosity = 1
        else:
            self.tortuosity = self.length / self._euclidean_distance(
                self.branch_coords[0], self.branch_coords[-1], spacing)
        self.tortuosity_times_num_voxels = self.tortuosity * len(self.branch_coords)

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

    def traverse_and_calculate_length_and_width(self, volume, mask, spacing, cell_center):
        total_length = 0
        widths = []
        current_point = xp.array(self.start_coord)
        self.branch_coords.append(tuple(current_point))

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
            current_orientation = self._calculate_point_orientation(current_point * spacing,
                                                                    xp.array(neighbor) * spacing,
                                                                    cell_center)
            self.orientations.append(current_orientation)

            # Set the closest point as the new current point and remove it from the volume
            current_point = xp.array(neighbor)

        self.length = total_length
        self.widths = widths
        self.circularity = xp.mean(widths) / self.length
        self.aspect_ratio = 1 / self.circularity
        self.orientations_mean = xp.mean(self.orientations)
        self.widths_mean = xp.mean(widths)

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

    def calculate_orientation(self, cell_center, spacing):
        """Calculate the orientation of the branch with respect to the given cell center."""
        if len(self.branch_coords) < 2:
            self.orientation = xp.nan
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
            self.orientation = min(angle_deg, 180 - angle_deg)

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

        return orientation

class Region:
    def __init__(self, organelle: OrganelleProperties):
        self.centroid = organelle.centroid
        self.coords = organelle.coords
        self.skeleton_coords = organelle.skeleton_coords
        self.instance_label = organelle.instance_label

        self.nodes = []
        self.branches = dict()
        self.branch_objects = dict()

        self.scaled_coords = None
        self.intensity_coords = None
        self.volume = None
        self.distance_from_cell_center_coords = None
        # self.surface_area = None
        # self.sphericity = None
        # self.compactness = None
        #
        # self.surface_mesh = None


class RegionAnalysis:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        organelle_props: OrganellePropertiesConstructor = unpickle_object(self.im_info.path_pickle_obj)
        node_props: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.node_props = node_props
        self.organelles = organelle_props.organelles
        self.spacing = organelle_props.spacing
        self.regions = dict()
        self.cell_center = dict()

    def get_regions(self):
        self._calculate_metrics()
        self._assign_nodes()
        self._get_branch_stats()
        self._clean_branch_stats()

    def _clean_branch_stats(self):
        for frame_num, frame_regions in self.regions.items():
            for region_num, region in frame_regions.items():
                for branch_num, branch_list in region.branch_objects.items():
                    if len(branch_list) < 2:
                        continue
                    longest_branch = max(branch_list, key=lambda x: x.length)
                    remove_branches = []
                    for branch in branch_list:
                        if branch is not longest_branch:
                            remove_branches.append(branch)
                    for branch in remove_branches:
                        self.regions[frame_num][region_num].branch_objects[branch_num].remove(branch)
    def _get_branch_stats(self):
        branch_labels = tifffile.memmap(self.im_info.path_im_label_seg, mode='r') > 0
        mask_im = tifffile.memmap(self.im_info.path_im_mask, mode='r') > 0
        for frame_num, frame_regions in self.regions.items():
            for region_num, region in frame_regions.items():
                for branch_tuple, branch_infos in region.branches.items():
                    for branch_info in branch_infos:
                        branch_object = Branch(frame_num, branch_tuple, branch_info)
                        branch_label = branch_info[0]
                        branch_object.traverse_and_calculate_length_and_width(
                            branch_labels[frame_num], mask_im[frame_num], self.spacing, self.cell_center[frame_num])
                        branch_object.calculate_tortuosity(self.spacing)
                        branch_object.calculate_orientation(self.cell_center[frame_num], self.spacing)
                        if region.branch_objects.get(branch_label) is None:
                            region.branch_objects[branch_label] = []
                        region.branch_objects[branch_label].append(branch_object)

    def _calculate_metrics(self):
        intensity_image = self.im_info.get_im_memmap(self.im_info.im_path)
        for frame_num, organelles in self.organelles.items():
            logger.debug(f'Calculating metrics for frame {frame_num}/{len(self.organelles.items())}')
            self.cell_center[frame_num] = xp.mean(xp.array([organelle.centroid for organelle in organelles]), axis=0)
            self.regions[frame_num] = dict()
            for organelle in organelles:
                logger.debug(f'Calculating metrics for organelle {organelle.instance_label}/{len(organelles)}')
                # binary_image = label_image[frame_num] == organelle.instance_label
                region = Region(organelle)
                region.scaled_coords = region.coords * self.spacing
                region.volume = len(region.coords) * xp.prod(self.spacing)
                region.intensity_coords = intensity_image[frame_num][
                    region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]
                region.distance_from_cell_center = xp.linalg.norm(
                    region.scaled_coords - self.cell_center[frame_num], axis=1)
                # region.surface_mesh = measure.marching_cubes(binary_image, step_size=1, spacing=self.spacing)
                # region.surface_area = measure.mesh_surface_area(region.surface_mesh[0], region.surface_mesh[1])
                # region.sphericity = region.surface_area / region.volume
                # region.compactness = region.volume / xp.sqrt(region.surface_area)
                self.regions[frame_num][region.instance_label] = region

    def _assign_nodes(self):
        for frame_num, nodes in self.node_props.nodes.items():
            logger.debug(f'Assigning nodes to regions for frame {frame_num}/{len(self.node_props.nodes.items())}')
            for node_num, node in enumerate(nodes):
                branches = dict()
                for connected_node_num in node.connected_nodes:
                    connected_node = self.node_props.nodes[frame_num][connected_node_num]
                    node_pair = [node_num, connected_node_num]
                    node_pair.sort()
                    node_pair = tuple(node_pair)
                    if node_pair not in self.regions[frame_num][node.skeleton_label].branches:
                        connected_node_labels = [branch[0] for branch in connected_node.connected_branches]
                        connected_branches = [branch for branch in node.connected_branches
                                              if branch[0] in connected_node_labels]
                        branches[node_pair] = connected_branches
                if node_num not in self.regions[frame_num][node.skeleton_label].nodes:
                    self.regions[frame_num][node.skeleton_label].nodes.append(node_num)
                self.regions[frame_num][node.skeleton_label].branches.update(branches)


class TrackBuilder:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.tracklets: dict[int: list[NodeTrack]] = unpickle_object(self.im_info.path_pickle_track)
        self.tracks = []
        self.time_between_volumes = self.im_info.dim_sizes['T']

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


class NodeAnalysis:
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

# class BranchAnalysis:
#     def __init__(self, label_image, node1, node2, spacing):
#         self.node1 = node1
#         self.node2 = node2
#         self.spacing = spacing
#         self.label_image = label_image
#         self.branch_coords = self.get_branch_coords()
#         self.length = self.calculate_length()
#         self.tortuosity = self.calculate_tortuosity()
#         self.orientation = self.calculate_orientation()
#         self.width = self.calculate_width()
#         self.aspect_ratio = self.calculate_aspect_ratio()
#         self.connected_label = self.get_connected_label()
#
#     def get_branch_coords(self):
#         branch_mask = (self.label_image == self.node1.instance_label) | (self.label_image == self.node2.instance_label)
#         branch_coords = np.argwhere(branch_mask)
#         return branch_coords
#
#     def calculate_length(self):
#         length = 0
#         for i in range(1, len(self.branch_coords)):
#             length += np.linalg.norm(np.array(self.branch_coords[i]) * self.spacing - np.array(self.branch_coords[i-1]) * self.spacing)
#         return length
#
#     def calculate_tortuosity(self):
#         end_to_end_distance = np.linalg.norm(np.array(self.node1.centroid_um) - np.array(self.node2.centroid_um))
#         return self.length / end_to_end_distance
#
#     def calculate_orientation(self, cell_center):
#         v1 = np.array(self.node1.centroid_um) - np.array(cell_center)
#         v2 = np.array(self.node2.centroid_um) - np.array(cell_center)
#         angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#         return angle
#
#     def calculate_width(self):
#         branch_mask = (self.label_image == self.node1.instance_label) | (self.label_image == self.node2.instance_label)
#         dist_map = distance_transform_edt(~branch_mask)
#         width = 2 * np.mean(dist_map[self.branch_coords[:, 0], self.branch_coords[:, 1]])
#         return width
#
#     def calculate_aspect_ratio(self):
#         return self.length / self.width

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
    # track_builder = TrackBuilder(test)
    # analysis = NodeAnalysis(test, track_builder.tracks)
    # analysis.calculate_metrics()
    # aggregate_output_file = 'aggregate_metrics.csv'
    # frame_output_folder = test.output_csv_dirpath
    # if not os.path.exists(frame_output_folder):
    #     os.makedirs(frame_output_folder)
    # analysis.save_metrics_to_csv(os.path.join(frame_output_folder, aggregate_output_file), frame_output_folder)
    regions = RegionAnalysis(test)
    regions.get_regions()
