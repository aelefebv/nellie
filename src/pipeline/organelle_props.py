from src.io.im_info import ImInfo
from src import xp, measure, logger
import tifffile


class OrganelleProperties:
    def __init__(self, organelle, skel_coords):
        self.instance_label = organelle.label
        self.centroid = organelle.centroid
        self.coords = organelle.coords
        self.skeleton_coords = skel_coords


class OrganellePropertiesConstructor:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.organelles = []

    def measure_organelles(self, num_t: int = None):
        # could potentially include intensity-weighted centroid, but not sure that it's necessary
        label_im = tifffile.memmap(self.im_info.path_im_label_obj, mode='r')
        skel_im = tifffile.memmap(self.im_info.path_im_skeleton, mode='r+')

        # Load only a subset of frames if num_t is not None
        if num_t is not None:
            num_t = min(num_t, label_im.shape[0])
            label_im = label_im[:num_t, ...]
            skel_im = skel_im[:num_t, ...]

        for frame_num, frame in enumerate(label_im):
            logger.info(f'Getting organelle properties, volume {frame_num}/{len(label_im)}')
            label_frame = xp.asarray(label_im[frame_num])
            skel_frame = xp.asarray(skel_im[frame_num])

            label_props = measure.regionprops(label_frame, spacing=self.spacing)
            label_dict = {props.label: props for props in label_props}
            skel_props = measure.regionprops(skel_frame, spacing=self.spacing)
            skel_dict = {props.label: props for props in skel_props}

            organelles_frame = []
            for label_num, label_prop in label_dict.items():

                # For some reason, skeletonization misses some small ones. This assigns centroid as skeleton.
                if label_num not in skel_dict.keys():
                    # current gpu implementation does not give unscaled centroid, so I rederive them here
                    unscaled_centroid = [round(label_prop.centroid[i] / self.spacing[i])
                                         for i in range(len(self.spacing))]
                    unscaled_centroid.insert(0, frame_num)
                    skel_im[tuple(unscaled_centroid)] = label_num
                    skel_coords = label_prop.centroid
                else:
                    skel_coords = skel_dict[label_num].coords

                organelles_frame.append(OrganelleProperties(label_prop, skel_coords))

            self.organelles.append(organelles_frame)


if __name__ == "__main__":
    from src.io.pickle_jar import pickle_object
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath, ch=0)
    organelle_props = OrganellePropertiesConstructor(test)
    organelle_props.measure_organelles(2)
    pickle_object(test.path_pickle_obj, organelle_props)
    print('hi')
