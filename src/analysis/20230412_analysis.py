from tifffile import tifffile

from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
from src.pipeline.tracking.node_to_node import NodeTrack
import numpy as xp


class MorphologyAnalysis:
    def __init__(self, im_info: ImInfo, track: list[NodeTrack]):
        self.track = track
        self.im_info = im_info
        self.spacing = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])


        # self.node_distance_from_center
        self.node_width = []
        self.branch_length = []
        self.branch_width = []
        self.node_branch_width_ratio = []
        self.branch_tortuosity = []
        self.branch_intensity = []
        self.branch_coords = []

    def run(self):
        for node_track in self.track:
            if len(node_track.node.connected_branches) > 1:
                self.node_width.append(xp.nan)
                self.branch_length.append(xp.nan)
                self.branch_width.append(xp.nan)
                self.node_branch_width_ratio.append(xp.nan)
                self.branch_tortuosity.append(xp.nan)
                self.branch_intensity.append(xp.nan)
                self.branch_coords.append(xp.nan)
                continue
            # mask_image = tifffile.memmap(self.im_info.path_im_mask)[node_track.frame_num, ...]
            # branch_volume = tifffile.memmap(self.im_info.path_im_label_seg)[node_track.frame_num, ...]
            # self.node_width.append(self.calculate_node_width(node_track, mask_image))
            # self.calculate_branch_stats(node_track, branch_volume, mask_image)
            # self.node_branch_width_ratio.append(self.calculate_node_branch_width_ratio(node_track))
            # self.branch_tortuosity.append(self.calculate_branch_tortuosity(node_track))
            # self.branch_intensity.append(self.calculate_branch_intensity(node_track))

    def calculate_node_width(self, node_track: NodeTrack, mask_image):
        # traverse in the given direction from the first coordinate until the intensity image is zero

        def _traverse(coord, direction):
            coord_iteration = xp.array(coord)
            z_max, y_max, x_max = mask_image.shape
            while mask_image[tuple(coord_iteration)] == 1:
                coord_iteration += direction
                if coord_iteration[0] >= z_max or coord_iteration[1] >= y_max or coord_iteration[2] >= x_max:
                    return xp.nan
            return xp.linalg.norm(coord_iteration - xp.array(coord), axis=0)

        # do it for x, y, and z directions
        start_coord = node_track.node.coords[0]
        # multiply by 2 to get the diameter in micrometers
        z_dist = _traverse(start_coord, (1, 0, 0)) * 2 * self.spacing[0]
        y_dist = _traverse(start_coord, (0, 1, 0)) * 2 * self.spacing[1]
        x_dist = _traverse(start_coord, (0, 0, 1)) * 2 * self.spacing[2]
        # return the mean of the smallest two distances
        distances = [z_dist, y_dist, x_dist]
        distances.sort()
        # mean_of_smallest_two = (distances[0] + distances[1]) / 2
        # self.all_widths = distances
        return distances

    def calculate_branch_stats(self, node_track: NodeTrack, branch_volume, mask_image):
        total_length = 0
        widths = []
        start_coord = node_track.node.connected_branches[0][1]
        current_point = xp.array(start_coord)
        branch_coords = []
        branch_coords.append(tuple(current_point))
        self.branch_coords.append(branch_coords)
        while True:
            neighbor = self._neighbors_in_volume(branch_volume, tuple(current_point))
            if not neighbor:
                # calculate the width for that pixel in any direction
                max_distance1 = self._search_along_direction(mask_image, current_point, xp.array([1, 0, 0]), self.spacing)
                max_distance2 = self._search_along_direction(mask_image, current_point, xp.array([0, 1, 0]), self.spacing)
                width = max(max_distance1, max_distance2) * 2
                # set the length equal to the width
                total_length += width
                widths.append(width)
                break
            neighbor = neighbor[0]
            # closest_point = min(neighbor, key=lambda x: xp.linalg.norm(current_point - xp.array(x)))

            # Calculate the distance between the current point and the closest point
            segment_length = xp.linalg.norm(current_point * self.spacing - xp.array(neighbor) * self.spacing)
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
            max_distance1 = self._search_along_direction(mask_image, current_point, orthogonal_vector1, self.spacing)
            max_distance2 = self._search_along_direction(mask_image, current_point, orthogonal_vector2, self.spacing)
            width = max(max_distance1, max_distance2) * 2
            widths.append(width)

            current_point = xp.array(neighbor)

        self.branch_length.append(total_length)
        self.branch_width.append(width)

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
                                self.branch_coords[-1].append((nz, ny, nx))
        return neighbors


    def calculate_branch_tortuosity(self, spacing):
        """Calculate the tortuosity of the branch."""
        coord_1 = self.branch_coords[0]
        coord_2 = self.branch_coords[-1]
        if coord_1 == coord_2:
            self.branch_tortuosity.append(1)
        else:
            self.branch_tortuosity.append(self.branch_length[-1] / self._euclidean_distance(
                self.branch_coords[-1][0], self.branch_coords[-1][-1], self.spacing))

    def _euclidean_distance(self, coord1, coord2):
        """Calculate the euclidean distance between two points."""
        return xp.sqrt(xp.sum((xp.array(coord1) * self.spacing - xp.array(coord2) * self.spacing) ** 2))

class MotilityAnalysis:
    # todo: this should be a class that takes in a track and stores a bunch of motility stats
    #  and then has a method to return a dict of those stats
    def __init__(self, im_info: ImInfo, track: list[NodeTrack]):
        self.track = track
        self.im_info = im_info
        self.morphology_stats = MorphologyAnalysis(im_info, track)
        # self.morphology_stats.run()


        self.distance = self.calculate_distance()
        self.speed = self.calculate_speed()
        self.displacement = self.calculate_displacement()
        self.persistance = self.calculate_persistance()
        self.fission, self.fusion = self.calculate_dynamics()


        self.speed_min = xp.nanmin(self.speed)
        self.speed_max = xp.nanmax(self.speed)
        self.speed_med = xp.nanmedian(self.speed)
        self.speed_q25 = xp.nanpercentile(self.speed, 25)
        self.speed_q75 = xp.nanpercentile(self.speed, 75)
        self.speed_IQR = self.speed_q75 - self.speed_q25
        self.speed_range = self.speed_max - self.speed_min

        self.displacement_final = self.displacement[-1]
        self.displacement_max = xp.nanmax(self.displacement)

        self.persistance_min = xp.nanmin(self.persistance)
        self.persistance_max = xp.nanmax(self.persistance)
        self.persistance_med = xp.nanmedian(self.persistance)
        self.persistance_q25 = xp.nanpercentile(self.persistance, 25)
        self.persistance_q75 = xp.nanpercentile(self.persistance, 75)
        self.persistance_IQR = self.persistance_q75 - self.persistance_q25
        self.persistance_range = self.persistance_max - self.persistance_min

        self.fission_mean = xp.nanmean(self.fission)
        self.fusion_mean = xp.nanmean(self.fusion)

        # self.direction = self.calculate_direction()

    def calculate_distance(self):
        distance = [xp.nan]
        for i in range(1, len(self.track)):
            distance.append(self.distance_between_nodes(self.track[i - 1].node, self.track[i].node))
        return xp.array(distance)

    def calculate_speed(self):
        speed = xp.array(self.distance) / self.im_info.dim_sizes['T']
        speed[0] = xp.nan
        return speed

    @staticmethod
    def distance_between_nodes(node1, node2):
        coord1 = xp.array(node1.centroid_um)
        coord2 = xp.array(node2.centroid_um)
        distance = xp.linalg.norm(coord1 - coord2)
        return distance

    def calculate_displacement(self):
        start_node = self.track[0].node
        displacement = [xp.nan]
        if len(self.track) == 1:
            return displacement
        for frame_node in self.track[1:]:
            displacement.append(self.distance_between_nodes(start_node, frame_node.node))
        return xp.array(displacement)

    def calculate_persistance(self):
        sum_distance = 0
        persistence = [xp.nan]
        for i in range(1, len(self.track)):
            sum_distance += self.distance[i]
            # first step is always of unit persistance, so no information gained
            if i == 1:
                persistence.append(xp.nan)
            elif sum_distance == 0:
                persistence.append(0)
            else:
                persistence.append(self.displacement[i] / sum_distance)
        return xp.array(persistence)

    def calculate_direction(self):
        pass

    def calculate_dynamics(self):
        fission = [xp.nan]
        fusion = [xp.nan]
        if len(self.track) == 1:
            return fission, fusion
        for frame_num, node_track in enumerate(self.track[1:]):
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


class TrackBuilder:
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


if __name__ == "__main__":
    import os

    top_dir = r"D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR"
    file_name = "deskewed-2023-04-06_13-58-58_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-1h.ome.tif"
    im_info = ImInfo(os.path.join(top_dir, file_name), ch=1)
    num_t = im_info.shape[0]
    # tracks = unpickle_object(im_info.path_pickle_track)
    stats = TrackBuilder(im_info)
    tracks = stats.tracks

    remove_tracks = []
    for track_a_num, track_a in enumerate(tracks):
        if len(track_a) < num_t / 2:
            remove_tracks.append(track_a)

    for track in remove_tracks:
        tracks.remove(track)

    track_stats = []
    for track in tracks:
        track_stats.append(MotilityAnalysis(im_info, track))
