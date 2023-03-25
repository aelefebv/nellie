from src.pipeline.tracking.node_to_node import NodeTrack, Node, NodeConstructor
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
from src import xp, logger
import csv

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
        self.nodes: list[list[Node]] = node_constructor.nodes
        self.tracks = tracks
        self.metrics = []

    def calculate_metrics(self):
        for track in self.tracks:
            track_metrics = {}
            track_metrics['frame'] = [node.frame_num for node in track]
            track_metrics['distance'] = self.calculate_distance(track)
            track_metrics['displacement'] = self.calculate_displacement(track)
            track_metrics['speed'] = self.calculate_speed(track)
            track_metrics['direction'] = self.calculate_direction(track)
            track_metrics['dynamics'] = self.calculate_dynamics(track)
            track_metrics['num_branches/node'] = self.calculate_num_branches_per_node(track)
            track_metrics['persistance'] = self.calculate_persistance(track)
            track_metrics['angles_at_junctions'] = self.calculate_angles_at_junctions(track)
            self.metrics.append(track_metrics)

    def save_metrics_to_csv(self, output_file):
        fieldnames = ['frame', 'distance', 'displacement', 'speed', 'direction', 'dynamics', 'num_branches/node', 'persistance', 'angles_at_junctions']
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

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
            return 0
        speed = xp.array(distance) / time
        return speed

    def calculate_direction(self, track):
        # Your implementation for calculating the direction (retro vs antero wrt cell center)
        # This depends on the specific definition of retrograde and anterograde movement in your context
        pass

    def calculate_dynamics(self, track):
        # Your implementation for calculating dynamics
        # This depends on the specific definition of dynamics in your context
        pass

    def calculate_num_branches_per_node(self, track):
        num_branches = sum([len(node_track.node.connected_branches) for node_track in track])
        num_nodes = len(track)
        if num_nodes == 0:
            return 0
        return num_branches / num_nodes

    def calculate_persistance(self, track):
        displacement = self.calculate_displacement(track)
        distance = self.calculate_distance(track)
        sum_distance = 0
        persistence = [xp.nan]
        for i in range(1, len(track)):
            sum_distance += distance[i]
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
            if len(node_track.node.connected_branches) > 2:
                angle_list = self.calculate_angles_for_junction(node_track)
                angles.extend(angle_list)
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
    analysis.save_metrics_to_csv('metrics.csv')
