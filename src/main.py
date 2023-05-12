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


def run(input_path: str, min_radius_um = 0.25, max_radius_um = 0.5,
        num_t: int = None, ch: int = 0, dimension_order = '',
        intensity_threshold = None, max_size_threshold = None,):
    im_info = ImInfo(input_path, ch=ch, dimension_order=dimension_order)
    # todo: idea, go backwards from last pipeline step, see if path is populated with a valid file.
    #  If invalid, keep going backwards until a valid one is found, then start pipeline from there.
    frangi = FrangiFilter(im_info, min_radius_um=min_radius_um, max_radius_um=max_radius_um,
                          intensity_threshold=intensity_threshold)
    frangi.run_filter(num_t)
    segmentation = Segment(im_info, min_radius_um=min_radius_um,
                           min_mean_intensity=intensity_threshold, max_size_threshold=max_size_threshold)
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
    nodes_test = NodeTrackConstructor(im_info, distance_thresh_um_per_sec=1, min_radius_um=min_radius_um)
    nodes_test.populate_tracks(num_t)
    pickle_object(im_info.path_pickle_track, nodes_test.tracks)
    # hierarchy = AnalysisHierarchyConstructor(im_info)
    # hierarchy.get_hierarchy()
    # hierarchy.save_stat_attributes()
    # track_builder = StatsDynamics(im_info)
    # todo I might want to reconnect disconnected/short tracks here
    # todo should have some way to pass it in a different intensity image (i.e. for tmre signal)


if __name__ == "__main__":
    import os

    top_dir = r"D:\test_files\nelly\20230413_AELxES-good-dmr_lipid_droplets_mt_DR-activate_deactivate"
    specific_file = None
    # specific_file = r"D:\test_files\nelly\20230413_AELxES-good-dmr_lipid_droplets_mt_DR-activate_deactivate\deskewed-2023-04-13_16-26-04_000_AELxES-stress_granules-dmr_perk-activate_deactivate-0p1nM-activate.ome.tif"
    min_radius_um = 0.2
    max_radius_um = 1
    ch_to_analyze = 0

    # intensity_threshold = 160
    intensity_threshold = None
    max_size_thresh = None

    files = glob.glob(os.path.join(top_dir, '*.tif*'))
    files.sort()
    if specific_file:
        files = [specific_file]
    # file_name = "deskewed-2023-03-23_13-02-09_000_20230323-AELxKL-dmr-lipid_droplets-1.ome.tif"
    for file_num, filepath in enumerate(files):
        print(file_num, len(files), filepath)
        # if 0p1 in filepath, skip
        # if '0p1' in filepath:
        #     continue
        # filepath = os.path.join(top_dir, file_name)
        # if not os.path.isfile(filepath):
        #     filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
        # try:
        #     run(filepath, num_t=None, dimension_order='ZYX')
        run(filepath, ch=ch_to_analyze,
            min_radius_um=min_radius_um, max_radius_um=max_radius_um,
            intensity_threshold=intensity_threshold, max_size_threshold=max_size_thresh)#, num_t=2)
            # run(filepath, ch=0)#, num_t=2)
        # except:
        # except FileNotFoundError:
        #     logger.error("Error, skipping.")
            # logger.error("File not found.")
            # continue
            # exit(1)
    print('hi')
