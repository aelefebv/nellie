import glob

from src.io.im_info import ImInfo
from src.io.pickle_jar import pickle_object
from src.pipeline.frangi_filtration import FrangiFilter
from src.pipeline.segmentation import Segment
from src.pipeline.skeletonization import Skeleton
from src.pipeline.organelle_props import OrganellePropertiesConstructor
from src.pipeline.networking import Neighbors
from src.pipeline.tracking.node_to_node import NodeTrackConstructor
from src.pipeline.node_props import NodeConstructor
from src.pipeline.analysis import AnalysisHierarchyConstructor
from src import logger
from src.pipeline.analysis import StatsDynamics, AnalysisDynamics


def run(input_path: str, num_t: int = None, ch: int = 0, dimension_order = ''):
    im_info = ImInfo(input_path, ch=ch, dimension_order=dimension_order)
    # todo: idea, go backwards from last pipeline step, see if path is populated with a valid file.
    #  If invalid, keep going backwards until a valid one is found, then start pipeline from there.
    frangi = FrangiFilter(im_info)
    frangi.run_filter(num_t)
    segmentation = Segment(im_info)
    segmentation.semantic(num_t)
    segmentation.instance(num_t)
    skeleton = Skeleton(im_info)
    skeleton.skeletonize(num_t)
    organelle_props = OrganellePropertiesConstructor(im_info)
    organelle_props.get_organelle_properties(num_t)
    pickle_object(im_info.path_pickle_obj, organelle_props)
    network = Neighbors(im_info)
    network.find_neighbors(num_t)
    nodes = NodeConstructor(im_info)
    nodes.get_node_properties(num_t)
    pickle_object(im_info.path_pickle_node, nodes)
    nodes_test = NodeTrackConstructor(im_info, distance_thresh_um_per_sec=1)
    nodes_test.populate_tracks(num_t)
    pickle_object(im_info.path_pickle_track, nodes_test.tracks)
    hierarchy = AnalysisHierarchyConstructor(im_info)
    hierarchy.get_hierarchy()
    hierarchy.save_stat_attributes()
    # track_builder = StatsDynamics(im_info)
    # todo I might want to reconnect disconnected/short tracks here


if __name__ == "__main__":
    import os
    top_dir = r"/Users/austin/test_files/nelly_Alireza"
    files = glob.glob(os.path.join(top_dir, '*.tif*'))
    files.sort()
    # file_name = "deskewed-2023-03-23_13-02-09_000_20230323-AELxKL-dmr-lipid_droplets-1.ome.tif"
    for file_num, filepath in enumerate(files[1:]):
        print(file_num, len(files), filepath)
        # filepath = os.path.join(top_dir, file_name)
        # if not os.path.isfile(filepath):
        #     filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
        try:
            run(filepath, num_t=None, dimension_order='ZYX')
        except FileNotFoundError:
            logger.error("File not found.")
            exit(1)
    print('hi')
