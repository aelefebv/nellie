from src.io.im_info import ImInfo
from src.io.pickle_jar import pickle_object
from src.pipeline.frangi_filtration import FrangiFilter
from src.pipeline.segmentation import Segment
from src.pipeline.skeletonization import Skeleton
from src.pipeline.organelle_props import OrganellePropertiesConstructor
from src.pipeline.networking import Neighbors
from src.pipeline.branching import BranchSegments


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
    branches = BranchSegments(im_info)
    branches.segment_branches(num_t)


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    run(filepath, 2)
    print('hi')
