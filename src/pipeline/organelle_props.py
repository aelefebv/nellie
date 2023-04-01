from src.io.im_info import ImInfo
from src import xp, measure, logger
import tifffile
from src.utils.general import get_reshaped_image


class OrganelleProperties:
    """
    A container for storing properties of an organelle, including its instance label, centroid, coordinates, and
    skeleton coordinates.
    """
    def __init__(self, organelle, skel_coords):
        """
        Initializes the OrganelleProperties instance.

        Args:
            organelle: An object returned by `skimage.measure.regionprops` that contains properties of an organelle.
            skel_coords: The coordinates of the organelle's skeleton in the original image.
        """
        # self.organelle = organelle
        self.instance_label = organelle.label
        self.centroid = organelle.centroid
        self.coords = organelle.coords
        self.skeleton_coords = skel_coords


class OrganellePropertiesConstructor:
    """
    A class for constructing OrganelleProperties instances from labeled images and skeleton images.
    """
    def __init__(self, im_info: ImInfo):
        """
        Initializes the OrganellePropertiesConstructor instance.

        Args:
            im_info: An ImInfo instance that provides information about the images to be processed and their paths.
        """
        self.im_info = im_info
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.organelles = dict()

    def get_organelle_properties(self, num_t: int = None):
        """
        Measures organelle properties for all frames of the labeled image and stores the results in the
        `organelles` attribute.

        Args:
            num_t: The number of frames to process. If not provided, all frames will be processed.
        """
        # could potentially include intensity-weighted centroid, but not sure that it's necessary
        label_im = tifffile.memmap(self.im_info.path_im_label_obj, mode='r')
        skel_im = tifffile.memmap(self.im_info.path_im_skeleton, mode='r+')
        label_im = get_reshaped_image(label_im, num_t, self.im_info)
        skel_im = get_reshaped_image(skel_im, num_t, self.im_info)
        shape = label_im.shape
        if len(skel_im.shape) == len(shape)-1:
            skel_im = skel_im[None, ...]

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
                    # current gpu implementation does not give unscaled centroid, so I re-derive them here
                    unscaled_centroid = [round(label_prop.centroid[i] / self.spacing[i])
                                         for i in range(len(self.spacing))]
                    unscaled_centroid.insert(0, frame_num)
                    skel_im[tuple(unscaled_centroid)] = label_num
                    skel_coords = label_prop.centroid
                else:
                    skel_coords = skel_dict[label_num].coords

                organelles_frame.append(OrganelleProperties(label_prop, skel_coords))
            self.organelles[frame_num] = organelles_frame


if __name__ == "__main__":
    from src.io.pickle_jar import pickle_object, unpickle_object
    windows_filepath = (r"D:\test_files\nelly\deskewed-single.ome.tif", '')
    mac_filepath = ("/Users/austin/Documents/Transferred/deskewed-single.ome.tif", '')

    custom_filepath = (r"/Users/austin/test_files/nelly_Alireza/1.tif", 'ZYX')

    filepath = custom_filepath
    try:
        test = ImInfo(filepath[0], ch=0, dimension_order=filepath[1])
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    organelle_props = OrganellePropertiesConstructor(test)
    organelle_props.get_organelle_properties()
    pickle_object(test.path_pickle_obj, organelle_props)
    organelle_props_unpickled = unpickle_object(test.path_pickle_obj)
    print('hi')
