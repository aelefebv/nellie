from src.pipeline.tracking.node_to_node import NodeTrack, Node, NodeConstructor
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
from src import xp, logger
import csv
import pandas as pd
from scipy import stats

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

    # def save_metrics_to_csv(self, output_file):
    #     fieldnames = ['frame', 'distance', 'displacement', 'speed', 'direction', 'fission', 'fusion', 'num_branches/node', 'persistance', 'angles_at_junctions']
    #     with open(output_file, 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for metric in self.metrics:
    #             writer.writerow(metric)

    def save_metrics_to_csv(self, aggregate_output_file, frame_output_file):
        self.save_aggregate_metrics_to_csv(aggregate_output_file)
        self.save_frame_metrics_to_csv(frame_output_file)

    def save_aggregate_metrics_to_csv(self, output_file):
        df = pd.DataFrame(self.metrics)
        df.to_csv(output_file)
        # create an empty df that I will append to
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
        for metric_name in df.columns:
            frame_output_file = os.path.join(output_folder, f'{metric_name}_frame_metrics.csv')

            with open(frame_output_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write the header
                csv_writer.writerow(['track_id', metric_name, 'frame_num'])

                # Write the data
                for track_id, frame_metrics in enumerate(df[metric_name]):
                    for idx, value in enumerate(frame_metrics):
                        if xp.isnan(value):
                            csv_writer.writerow([track_id, 'NaN', df['frame'][track_id][idx]])
                        else:
                            csv_writer.writerow([track_id, value, df['frame'][track_id][idx]])

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
    track_builder = TrackBuilder(test)
    analysis = NodeAnalysis(test, track_builder.tracks)
    analysis.calculate_metrics()
    aggregate_output_file = 'aggregate_metrics.csv'
    frame_output_folder = test.output_csv_dirpath
    if not os.path.exists(frame_output_folder):
        os.makedirs(frame_output_folder)
    analysis.save_metrics_to_csv(os.path.join(frame_output_folder, aggregate_output_file), frame_output_folder)
