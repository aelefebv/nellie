from datetime import datetime

from tifffile import tifffile

from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object, pickle_object
from src.pipeline.tracking.node_to_node import NodeTrack
import numpy as xp
import pandas as pd


class AllBranches:
    def __init__(self, im_info: ImInfo, tracks: list[list[NodeTrack]], mask_im, branch_im):
        self.im_info = im_info
        self.tracks = tracks
        self.stats_branches = dict()
        self.mask_im = mask_im
        self.branch_im = branch_im
        self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        self._construct_branch_objects()

    def _construct_branch_objects(self):
        for track_num, track in enumerate(self.tracks):
            for idx, node_track in enumerate(track):
                if len(node_track.node.connected_branches) != 1:
                    continue
                branch_id, branch_start_coord = node_track.node.connected_branches[0]
                if self.stats_branches.get(node_track.frame_num) is None:
                    self.stats_branches[node_track.frame_num] = dict()
                if self.stats_branches[node_track.frame_num].get(branch_id) is None:
                    # self.stats_branches[node_track.frame_num][branch_id] = 'hi'
                    self.stats_branches[node_track.frame_num][branch_id] = BranchMorphology(
                        self.im_info, self.spacing, branch_start_coord,
                        self.mask_im[node_track.frame_num], self.branch_im[node_track.frame_num])


class BranchMorphology:
    def __init__(self, im_info: ImInfo, spacing, branch_start_coord, mask_frame, branch_frame):
        self.im_info = im_info
        self.start_coord = branch_start_coord
        self.branch_coords = []
        self.branch_length = []
        self.branch_width = []
        self.branch_tortuosity = []
        self.tortuosity_length = []
        self.calculate_branch_stats(mask_frame, branch_frame, spacing)
        self.calculate_branch_tortuosity(spacing)

    def calculate_branch_stats(self, mask_frame, branch_frame, spacing):
        total_length = 0
        widths = []
        start_coord = self.start_coord
        current_point = xp.array(start_coord)
        self.branch_coords.append(tuple(current_point))
        while True:
            neighbor = self._neighbors_in_volume(branch_frame, tuple(current_point))
            if not neighbor:
                # calculate the width for that pixel in any direction
                max_distance1 = self._search_along_direction(mask_frame, current_point, xp.array([1, 0, 0]), spacing)
                max_distance2 = self._search_along_direction(mask_frame, current_point, xp.array([0, 1, 0]), spacing)
                width = max(max_distance1, max_distance2) * 2
                # set the length equal to the width
                self.tortuosity_length = total_length
                total_length += width
                widths.append(width)
                break
            neighbor = neighbor[0]
            # closest_point = min(neighbor, key=lambda x: xp.linalg.norm(current_point - xp.array(x)))

            # Calculate the distance between the current point and the closest point
            segment_length = self._euclidean_distance(current_point, xp.array(neighbor), spacing)
            # segment_length = xp.linalg.norm(current_point * spacing - xp.array(neighbor) * spacing)
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
            max_distance1 = self._search_along_direction(mask_frame, current_point, orthogonal_vector1, spacing)
            max_distance2 = self._search_along_direction(mask_frame, current_point, orthogonal_vector2, spacing)
            width = max(max_distance1, max_distance2) * 2
            widths.append(width)

            current_point = xp.array(neighbor)

        self.branch_length = total_length
        self.branch_width = width

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
                                self.branch_coords.append((nz, ny, nx))
        return neighbors

    def calculate_branch_tortuosity(self, spacing):
        """Calculate the tortuosity of the branch."""
        coord_1 = self.branch_coords[0]
        coord_2 = self.branch_coords[-1]
        if coord_1 == coord_2:
            self.branch_tortuosity = 1
        else:
            self.branch_tortuosity = xp.array(self.tortuosity_length) / self._euclidean_distance(
                coord_1, coord_2, spacing)

    def _euclidean_distance(self, coord1, coord2, spacing):
        """Calculate the euclidean distance between two points."""
        return xp.sqrt(xp.sum((xp.array(coord1) * spacing - xp.array(coord2) * spacing) ** 2))

class MorphologyAnalysis:
    def __init__(self, im_info: ImInfo, track: list[NodeTrack], mask_image, intensity_image, all_branches):
        self.track = track
        self.im_info = im_info
        self.spacing = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])


        # self.node_distance_from_center
        self.node_widths = []

        self.node_width = []
        self.branch_length = []
        self.branch_width = []
        self.node_branch_width_ratio = []
        self.branch_tortuosity = []
        self.branch_intensity = {}

        self.axes = self.im_info.axes
        # get the number of intensity channels and channel dimension position
        self.num_intensity_channels = 0
        self.channel_dim = None
        if self.axes.find('C') != -1:
            self.channel_dim = self.axes.find('C')
            self.num_intensity_channels = intensity_image.shape[self.channel_dim]
        else:
            self.num_intensity_channels = 1

        for i in range(self.num_intensity_channels):
            self.branch_intensity[i] = []

        self.run(mask_image, intensity_image, all_branches)

        self.node_width_min = xp.nanmin(self.node_width)
        self.node_width_max = xp.nanmax(self.node_width)
        self.node_width_med = xp.nanmedian(self.node_width)
        self.node_width_q25 = xp.nanpercentile(self.node_width, 25)
        self.node_width_q75 = xp.nanpercentile(self.node_width, 75)
        self.node_width_IQR = self.node_width_q75 - self.node_width_q25
        self.node_width_range = self.node_width_max - self.node_width_min

        self.branch_length_min = xp.nanmin(self.branch_length)
        self.branch_length_max = xp.nanmax(self.branch_length)
        self.branch_length_med = xp.nanmedian(self.branch_length)
        self.branch_length_q25 = xp.nanpercentile(self.branch_length, 25)
        self.branch_length_q75 = xp.nanpercentile(self.branch_length, 75)
        self.branch_length_IQR = self.branch_length_q75 - self.branch_length_q25
        self.branch_length_range = self.branch_length_max - self.branch_length_min

        self.branch_width_min = xp.nanmin(self.branch_width)
        self.branch_width_max = xp.nanmax(self.branch_width)
        self.branch_width_med = xp.nanmedian(self.branch_width)
        self.branch_width_q25 = xp.nanpercentile(self.branch_width, 25)
        self.branch_width_q75 = xp.nanpercentile(self.branch_width, 75)
        self.branch_width_IQR = self.branch_width_q75 - self.branch_width_q25
        self.branch_width_range = self.branch_width_max - self.branch_width_min

        self.node_branch_width_ratio_min = xp.nanmin(self.node_branch_width_ratio)
        self.node_branch_width_ratio_max = xp.nanmax(self.node_branch_width_ratio)
        self.node_branch_width_ratio_med = xp.nanmedian(self.node_branch_width_ratio)
        self.node_branch_width_ratio_q25 = xp.nanpercentile(self.node_branch_width_ratio, 25)
        self.node_branch_width_ratio_q75 = xp.nanpercentile(self.node_branch_width_ratio, 75)
        self.node_branch_width_ratio_IQR = self.node_branch_width_ratio_q75 - self.node_branch_width_ratio_q25
        self.node_branch_width_ratio_range = self.node_branch_width_ratio_max - self.node_branch_width_ratio_min

        self.branch_tortuosity_min = xp.nanmin(self.branch_tortuosity)
        self.branch_tortuosity_max = xp.nanmax(self.branch_tortuosity)
        self.branch_tortuosity_med = xp.nanmedian(self.branch_tortuosity)
        self.branch_tortuosity_q25 = xp.nanpercentile(self.branch_tortuosity, 25)
        self.branch_tortuosity_q75 = xp.nanpercentile(self.branch_tortuosity, 75)
        self.branch_tortuosity_IQR = self.branch_tortuosity_q75 - self.branch_tortuosity_q25
        self.branch_tortuosity_range = self.branch_tortuosity_max - self.branch_tortuosity_min

        self.branch_intensity_min = []
        self.branch_intensity_max = []
        self.branch_intensity_med = []
        self.branch_intensity_q25 = []
        self.branch_intensity_q75 = []
        self.branch_intensity_IQR = []
        self.branch_intensity_range = []

        for i in range(self.num_intensity_channels):
            self.branch_intensity_min.append(xp.nanmin(self.branch_intensity[i]))
            self.branch_intensity_max.append(xp.nanmax(self.branch_intensity[i]))
            self.branch_intensity_med.append(xp.nanmedian(self.branch_intensity[i]))
            self.branch_intensity_q25.append(xp.nanpercentile(self.branch_intensity[i], 25))
            self.branch_intensity_q75.append(xp.nanpercentile(self.branch_intensity[i], 75))
            self.branch_intensity_IQR.append(self.branch_intensity_q75[i] - self.branch_intensity_q25[i])
            self.branch_intensity_range.append(self.branch_intensity_max[i] - self.branch_intensity_min[i])


    def run(self, mask_image, intensity_image, all_branches):
        for node_track in self.track:
            num_branches = len(node_track.node.connected_branches)
            if num_branches > 1:
                self.node_widths.append(xp.nan)
                self.node_width.append(xp.nan)
                self.branch_length.append(xp.nan)
                self.branch_width.append(xp.nan)
                self.node_branch_width_ratio.append(xp.nan)
                self.branch_tortuosity.append(xp.nan)
                for channel in range(self.num_intensity_channels):
                    self.branch_intensity[channel].append(xp.nan)
                # self.branch_intensity.append(xp.nan)
                continue
            self.node_widths.append(self.calculate_node_width(node_track, mask_image[node_track.frame_num]))
            if num_branches == 0:
                self.node_width.append(xp.nanmean(self.node_widths[-1]))
                self.branch_length.append(xp.nanmax(self.node_widths[-1]))
                self.branch_width.append(xp.nanmean(self.node_widths[-1][:2]))
                self.branch_tortuosity.append(1.0)
                for channel in range(self.num_intensity_channels):
                    channel_intensity_image = xp.take(intensity_image, channel, axis=self.channel_dim)
                    branch_intensities = []
                    for coord in node_track.node.coords:
                        # print(coord)
                        # print(channel_intensity_image.shape)
                        branch_intensities.append(xp.nanmean(channel_intensity_image[node_track.frame_num][tuple(coord)]))
                    branch_intensity_mean = xp.nanmean(branch_intensities)
                    self.branch_intensity[channel].append(branch_intensity_mean)
                self.node_branch_width_ratio.append(self.node_width[-1] / self.branch_width[-1])
                continue
            branch = all_branches.stats_branches[node_track.frame_num][node_track.node.connected_branches[0][0]]
            self.get_branch_stats(branch, intensity_image[node_track.frame_num])
            # todo don't do this, get branch stats for branches first then pull from there

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

    def get_branch_stats(self, branch, intensity_image):
        self.node_width.append(xp.nanmean(self.node_widths[-1][:2]))
        self.branch_length.append(branch.branch_length + xp.nanmin(self.node_widths[-1]))
        self.branch_width.append(branch.branch_width)
        self.branch_tortuosity.append(branch.branch_tortuosity)
        for channel in range(self.num_intensity_channels):
            channel_intensity_image = xp.take(intensity_image, channel, axis=self.channel_dim-1)
            branch_intensities = []
            for coord in branch.branch_coords:
                # print(coord)
                # print(intensity_image.shape)
                branch_intensities.append(channel_intensity_image[tuple(coord)])
            branch_intensity_mean = xp.nanmean(branch_intensities)
            self.branch_intensity[channel].append(branch_intensity_mean)
        self.node_branch_width_ratio.append(self.node_width[-1] / self.branch_width[-1])


class MotilityAnalysis:
    # todo: this should be a class that takes in a track and stores a bunch of motility stats
    #  and then has a method to return a dict of those stats
    def __init__(self, im_info: ImInfo, track: list[NodeTrack]):
        self.track = track
        self.im_info = im_info


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
        # self.speed_max_min_ratio = self.speed_max / self.speed_min
        # self.speed_max_med_ratio = self.speed_max / self.speed_med

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
    def __init__(self, im_info: ImInfo, num_t=None):
        self.im_info = im_info
        self.tracklets: dict[int: list[NodeTrack]] = unpickle_object(self.im_info.path_pickle_track)
        # only get tracklets from first num_t frames
        if num_t is not None:
            self.tracklets = {frame_num: tracklets for frame_num, tracklets in self.tracklets.items() if frame_num < num_t}
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
                if child['frame'] not in self.tracklets:
                    continue
                child_node = self.tracklets[child['frame']][child['track']]
                # print(child_node)
                # if child_node.frame_num >= num_t:
                #     continue
                self.build_tracks(child_node, current_track.copy(), all_tracks)

        return all_tracks


class TrackStats:
    def __init__(self):
        self.morphology = None
        self.motility = None
        self.track_num = None
        self.concentration = None
        self.file_name = None
        self.file_num = None


    # def gather_attributes(self):
    #     EXCLUDED_ATTRIBUTES = [
    #         'track', 'im_info', 'spacing',
    #         'node_widths', 'node_width',
    #         'branch_length', 'branch_width', 'node_branch_width_ratio',
    #         'branch_tortuosity', 'branch_intensity',
    #         'distance', 'speed', 'displacement', ''
    #     ]
    #     attributes = {}
    #     for attr_name in dir(self.morphology):
    #         if not attr_name.startswith("__") and attr_name not in EXCLUDED_ATTRIBUTES:
    #             attributes[attr_name] = getattr(self.morphology, attr_name)
    #     for attr_name in dir(self.motility):
    #         if not attr_name.startswith("__") and attr_name not in EXCLUDED_ATTRIBUTES:
    #             attributes[attr_name] = getattr(self.motility, attr_name)
    #     return attributes

    def gather_attributes(self):
        attributes = {}
        for analysis_instance in [self.morphology, self.motility]:
            for attr_name in dir(analysis_instance):
                attr_value = getattr(analysis_instance, attr_name)
                if (
                    not attr_name.startswith("__")
                    and isinstance(attr_value, (float, int))
                ):
                    attributes[attr_name] = attr_value
                elif attr_name.startswith("branch_intensity"):
                    for ch, attr in enumerate(attr_value):
                        attributes[f"{attr_name}_ch{ch}"] = attr

        attributes['track_num'] = self.track_num
        attributes['file_num'] = self.file_num
        attributes['file_name'] = self.file_name
        attributes['concentration'] = self.concentration

        # num_intensity_channels = self.morphology.num_intensity_channels
        # for ch in range(num_intensity_channels):
        #     attributes[f'branch_intensity_ch{ch}'] = self.morphology.branch_intensity[ch]
        return attributes

if __name__ == "__main__":
    import os
    import re

    ch_to_analyze = 0
    num_t = None
    if num_t is not None:
        num_t_threshold = num_t // 2


    top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"
    # find all files that end with 0-1h.ome.tif
    files = [file for file in os.listdir(top_dir) if file.endswith(".ome.tif")]
    files.sort()



    # Set up the CSV file
    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(top_dir, f"{date_now}-ch_{ch_to_analyze}-track_stats.csv")

    # file_name = "deskewed-2023-04-06_13-58-58_000_AELxKL-dmr_PERK-lipid_droplets_mtDR"#-5000-1h.ome.tif"
    data = []

    for file_num, file_name in enumerate(files[1:2]):
        try:
            print(file_num, len(files))
            match = re.search(r'-(\d+)-\d+h', file_name)

            if match:
                output = match.group(1)
            else:
                output = xp.nan
            print('Loading image info...')
            im_info = ImInfo(os.path.join(top_dir, file_name), ch=ch_to_analyze)
            if num_t is None:
                num_t = im_info.shape[0]
                num_t_threshold = num_t // 2
            # tracks = unpickle_object(im_info.path_pickle_track)
            print('Creating tracks...')
            stats = TrackBuilder(im_info, num_t=num_t)
            tracks = stats.tracks

            print('Finding short tracks and junction tracks...')
            # remove_tracks = []
            keep_tracks = []
            for track_a_num, track_a in enumerate(tracks):
                print(track_a_num, len(tracks))
                if len(track_a) < num_t_threshold:
                    # remove_tracks.append(track_a)
                    continue
                num_junctions = 0
                for node_track in track_a:
                    if node_track.node.node_type == 'junction':
                        num_junctions += 1
                if num_junctions >= (num_t / 2) - 1:
                    # remove_tracks.append(track_a)
                    continue
                keep_tracks.append(track_a)
            # break

            # print('Removing short tracks and junction tracks...')
            # for track_num, track in enumerate(remove_tracks):
            #     print(track_num, len(remove_tracks))
            #     if track in tracks:
            #         tracks.remove(track)

            print('Pickling tracks...')
            # pickle the tracks list
            pickle_object(im_info.path_pickle_seg, keep_tracks)

            print('Loading images...')
            mask_image = im_info.get_im_memmap(im_info.path_im_mask)
            branch_image = im_info.get_im_memmap(im_info.path_im_label_seg)
            intensity_image = im_info.get_im_memmap(im_info.im_path, ch='all')

            print('Getting branches...')
            all_branches = AllBranches(im_info, keep_tracks, mask_image, branch_image)

            print('Analyzing tracks...')
            track_stats = {}
            for track_num, track in enumerate(keep_tracks[:10]):
                print(track_num, len(keep_tracks))
                track_stats[track_num] = TrackStats()
                track_stats[track_num].motility = MotilityAnalysis(im_info, track)
                track_stats[track_num].morphology = MorphologyAnalysis(im_info, track, mask_image, intensity_image, all_branches)
                track_stats[track_num].track_num = track_num
                track_stats[track_num].concentration = output
                track_stats[track_num].file_name = file_name
                track_stats[track_num].file_num = file_num

            # Gather track attributes
            data = []
            print('Gathering track attributes...')
            for track_stats in track_stats.values():
                attributes = track_stats.gather_attributes()
                data.append(attributes)

            # Save the attributes to the CSV for this file
            df = pd.DataFrame(data)
            mode = 'a' if os.path.exists(csv_path) else 'w'
            with open(csv_path, mode) as f:
                df.to_csv(f, header=f.tell() == 0, index=False)


        except:
            print('beep boop something went wrong')
            continue

    # print('Saving track attributes...')
    # df = pd.DataFrame(data)
    # date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # df.to_csv(os.path.join(top_dir, f"{date_now}-ch_{ch_to_analyze}-track_stats.csv"))

   # need a good way to visualize tracks from dataframe
