from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.im_info import ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.all_tracks_for_label import LabelTracks
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner


def run(im_path, num_t=None, remove_edges=True, ch=0):
    im_info = ImInfo(im_path, ch=ch)

    preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
    preprocessing.run()

    segmenting = Label(im_info, num_t)
    segmenting.run()

    networking = Network(im_info, num_t)
    networking.run()

    mocap_marking = Markers(im_info, num_t)
    mocap_marking.run()

    hu_tracking = HuMomentTracking(im_info, num_t)
    hu_tracking.run()

    vox_reassign = VoxelReassigner(im_info, num_t)
    vox_reassign.run()

    # hierarchy = Hierarchy(im_info, num_t)
    # hierarchy.run()

    return im_info


if __name__ == "__main__":
    # Single file run
    im_path = r"/Users/austin/GitHub/nellie-simulations/motion/angular/angular-length_32-std_512-t_1.ome.tif"
    num_t = None
    im_info = run(im_path, num_t=num_t, remove_edges=False)
    label_tracks = LabelTracks(im_info, num_t=num_t)
    label_tracks.initialize()
    # tracks, track_properties = label_tracks.run(label_num=None, skip_coords=1)

    all_tracks = []
    all_props = {}
    max_track_num = 0
    if num_t is None:
        num_t = im_info.shape[0]
    for frame in range(num_t):
        tracks, track_properties = label_tracks.run(label_num=None, start_frame=frame, end_frame=num_t,
                                                    min_track_num=max_track_num,
                                                    skip_coords=1)
        all_tracks += tracks
        for property in track_properties.keys():
            if property not in all_props.keys():
                all_props[property] = []
            all_props[property] += track_properties[property]
        if len(tracks) == 0:
            break
        max_track_num = max([track[0] for track in tracks]) + 1

    import napari
    viewer = napari.Viewer()

    raw_im = im_info.get_im_memmap(im_info.im_path)
    frangi_im = im_info.get_im_memmap(im_info.pipeline_paths['im_frangi'])
    marker_im = im_info.get_im_memmap(im_info.pipeline_paths['im_marker']) > 0
    instance_label = im_info.get_im_memmap(im_info.pipeline_paths['im_instance_label'])
    reassigned = im_info.get_im_memmap(im_info.pipeline_paths['im_obj_label_reassigned'])
    branch_reassigned = im_info.get_im_memmap(im_info.pipeline_paths['im_branch_label_reassigned'])
    viewer.add_image(raw_im, name='raw_im')
    # viewer.add_image(frangi_im, name='frangi_im')
    # viewer.add_labels(marker_im, name='marker im')
    # viewer.add_labels(instance_label, name='instance im')
    viewer.add_labels(reassigned, name='instance im')
    viewer.add_labels(branch_reassigned, name='branch instance im')
    viewer.add_tracks(all_tracks, properties=all_props, name='tracks')
    napari.run()
