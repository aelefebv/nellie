from src.io.im_info import ImInfo
from src.io.pickle_jar import pickle_object
from src.pipeline.frangi_filtration import FrangiFilter
from src.pipeline.segmentation import Segment
from src.pipeline.skeletonization import Skeleton
from src.pipeline.organelle_props import OrganellePropertiesConstructor
from src.pipeline.networking import Neighbors
from src.pipeline.branch_labeling import BranchSegments
from src.pipeline.node_props import NodeConstructor
from src import logger


def run(input_path: str, num_t: int = 0):
    im_info = ImInfo(input_path)
    # todo: idea, go backwards from last pipeline step, see if path is populated with a valid file.
    #  If invalid, keep going backwards until a valid one is found, then start pipeline from there.
    # frangi = FrangiFilter(im_info)
    # frangi.run_filter(num_t)
    # segmentation = Segment(im_info)
    # segmentation.semantic(num_t)
    # segmentation.instance(num_t)
    # skeleton = Skeleton(im_info)
    # skeleton.skeletonize(num_t)
    # organelle_props = OrganellePropertiesConstructor(im_info)
    # organelle_props.measure_organelles(num_t)
    # pickle_object(im_info.path_pickle_obj, organelle_props)
    # network = Neighbors(im_info)
    # network.find_neighbors(num_t)
    # branches = BranchSegments(im_info)
    # branches.segment_branches(num_t)
    nodes = NodeConstructor(im_info)
    nodes.get_node_properties(num_t)
    pickle_object(im_info.path_pickle_node, nodes)


if __name__ == "__main__":
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        run(filepath, 2)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    print('hi')
