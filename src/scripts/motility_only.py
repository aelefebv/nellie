import pandas as pd

from src import logger
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
import tifffile
import os
import numpy as np
import skimage.measure
import scipy.ndimage


top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"

# find all .ome.tif files in the top_dir
ome_tif_files = [os.path.join(top_dir, file) for file in os.listdir(top_dir) if file.endswith('.ome.tif')]

length_cutoff = 6  # greater than half the frames means no duplicates

for file_num, filepath in enumerate(ome_tif_files[:1]):
    print(f'Processing file {file_num + 1} of {len(ome_tif_files)}')
    try:
        test = ImInfo(filepath, ch=0, dimension_order='')
    except FileNotFoundError:
        logger.error("File not found.")
        continue

    file_info = filepath.split('_acquire')[0].split('_')
    scale = [test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']]
    condition = file_info[-2]
    concentration = file_info[-1]

    tracks = unpickle_object(test.path_pickle_track)
    track_list = []
    track_idx = 0
    for frame_num, frame_tracks in tracks.items():
        print(f'Processing frame {frame_num} of {len(tracks)}')
        for track_num, track in enumerate(frame_tracks):
            print(f'Processing track {track_num} of {len(frame_tracks)}')
            num_parents = len(track.parents)
            num_children = len(track.children)
            if num_children > 1:
                continue
            if num_parents != 1:
                # start new track
                new_track = [track.node]
                check_track = track
                # iterate through children until stopping criteria is met
                while num_children == 1:
                    child_dict = check_track.children[0]
                    child_track = tracks[child_dict['frame']][child_dict['track']]
                    num_parents = len(child_track.parents)
                    if num_parents != 1:
                        break
                    new_track.append(child_track.node)
                    num_children = len(child_track.children)
                    check_track = child_track
                if len(new_track) < length_cutoff:
                    continue
                track_list.append(new_track)
                track_idx += 1

            else:
                continue

    for track_num, track in enumerate(track_list[:1]):
        speeds = []
        total_distance = 0
        displacements = []
        first_node = track[0]
        for node_num, node in enumerate(track[1:]):
            node_last = track[node_num - 1]
            distance = np.linalg.norm(np.array(node_last.centroid_um) - np.array(node.centroid_um))
            displacement = np.linalg.norm(np.array(first_node.centroid_um) - np.array(node.centroid_um))
            time_diff = node.time_point_sec - node_last.time_point_sec
            speed = distance / time_diff

            speeds.append(speed)
            total_distance += distance
            displacements.append(displacement)

        speed_mean = np.mean(speeds)
        speed_median = np.median(speeds)
        speed_std = np.std(speeds)
        final_displacement = displacements[-1]
        max_displacement = np.max(displacements)

        # concat the region_df to the df
        df = pd.concat([df, region_df], ignore_index=True)

# save df as csv in the top dir with top dir name
df.to_csv(os.path.join(top_dir, f'{top_dir.split(os.sep)[-1]}_first_frame_morphology.csv'), index=False)

# import napari
# viewer = napari.Viewer()
# viewer.add_labels(branches, name='branches')
# viewer.add_image(og_im[0], name='mt_green')
# viewer.add_image(mask_im, name='mask')
# viewer.add_image(distance_transform_mask, name='distance_transform_mask')
