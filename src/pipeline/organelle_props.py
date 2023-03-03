from src.io.im_info import ImInfo
from src import xp, ndi, measure, is_gpu
import tifffile


class OrganelleProperties:
    def __init__(self, label_region, skel_region):
        self.instance_label = label_region.label
        self.centroid = label_region.centroid
        self.coords = label_region.coords
        self.skeleton_coords = skel_region.coords



class OrganellePropertiesConstructor:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.organelles = []

    def measure_organelles(self, num_t: int = None):
        # could potentially include intensity-weighted centroid, but not sure it's necessary
        label_im = tifffile.memmap(self.im_info.path_im_label_obj, mode='r')
        skel_im = tifffile.memmap(self.im_info.path_im_mask, mode='r+')

        # Load only a subset of frames if num_t is not None
        if num_t is not None:
            num_t = min(num_t, semantic_mask.shape[0])
            label_im = label_im[:num_t, ...]
            skel_im = skel_im[:num_t, ...]

        for frame_num, frame in enumerate(label_im):
            label_frame = xp.asarray(label_im[frame_num])
            skel_frame = xp.asarray(skel_im[frame_num])
            skel_label_set = set(skel_frame)  # get all the unique labels

            label_props = measure.regionprops(label_im, spacing=self.spacing)
            skel_props = measure.regionprops(skel_frame, spacing=self.spacing)

            organelles_frame = []
            for organelle in label_props:

                # For some reason, skeletonization misses some small ones. This assigns centroid as skeleton.
                if label_region.label not in skel_label_set:
                    skel_region = None
                    # todo need to make sure spacing still allows setting proper coordinates.
                    skel_region.coords = label_region.centroid

                    # labels skeleton image as well.
                    if is_gpu:
                        skel_im[skel_region.coords] = region.label.get()
                    else:
                        skel_im[skel_region.coords] = region.label

                organelles_frame.append(OrganelleProperties(label_region, skel_region))

            self.organelles.append(organelles_frame)
