import os

from qtpy.QtWidgets import QGridLayout, QWidget, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout
from tifffile import tifffile

from nellie import logger
from nellie.tracking.all_tracks_for_label import LabelTracks


class NellieVisualizer(QWidget):
    """
    The NellieVisualizer class provides an interface for visualizing different stages of the Nellie pipeline, such as raw images,
    preprocessed images, segmentation labels, mocap markers, and reassigned labels. It also enables track visualization for specific
    labels and all frame labels using napari.

    Attributes
    ----------
    nellie : object
        Reference to the Nellie instance managing the pipeline.
    viewer : napari.viewer.Viewer
        Reference to the napari viewer instance.
    scale : tuple of floats
        Scaling factors for visualizing the images, based on the resolution of the image dimensions.
    im_memmap : ndarray
        Memory-mapped array for the raw image.
    raw_layer : ImageLayer
        The napari image layer for displaying the raw image.
    im_branch_label_reassigned_layer : LabelsLayer
        The napari labels layer for displaying reassigned branch labels.
    im_branch_label_reassigned : ndarray
        Memory-mapped array for reassigned branch labels.
    im_obj_label_reassigned_layer : LabelsLayer
        The napari labels layer for displaying reassigned object labels.
    im_obj_label_reassigned : ndarray
        Memory-mapped array for reassigned object labels.
    im_marker_layer : ImageLayer
        The napari image layer for displaying mocap markers.
    im_marker : ndarray
        Memory-mapped array for mocap marker images.
    im_skel_relabelled_layer : LabelsLayer
        The napari labels layer for displaying skeleton relabeled images.
    im_instance_label_layer : LabelsLayer
        The napari labels layer for displaying instance labels.
    frangi_layer : ImageLayer
        The napari image layer for displaying preprocessed images (Frangi-filtered).
    im_skel_relabelled : ndarray
        Memory-mapped array for skeleton relabeled images.
    im_instance_label : ndarray
        Memory-mapped array for instance labels.
    im_frangi : ndarray
        Memory-mapped array for the preprocessed (Frangi-filtered) images.
    initialized : bool
        Flag indicating whether the visualizer has been initialized.

    Methods
    -------
    set_ui()
        Initializes and sets the layout and UI components for the NellieVisualizer.
    post_init()
        Initializes the visualizer by setting the scale and making the scale bar visible in napari.
    check_3d()
        Ensures that the napari viewer is in 3D mode if the dataset contains Z-dimension data.
    set_scale()
        Sets the scale for image display based on the image resolution in Z, Y, and X dimensions.
    open_preprocess_image()
        Opens and displays the preprocessed (Frangi-filtered) image in the napari viewer.
    open_segment_image()
        Opens and displays the segmentation labels (skeleton relabeled and instance labels) in the napari viewer.
    on_track_selected()
        Visualizes the tracks for the currently selected label in the napari viewer.
    track_all()
        Visualizes tracks for all labels across frames in the napari viewer.
    open_mocap_image()
        Opens and displays the mocap marker image in the napari viewer.
    open_reassign_image()
        Opens and displays the reassigned branch and object labels in the napari viewer.
    open_raw()
        Opens and displays the raw image in the napari viewer.
    check_file_existence()
        Checks for the existence of files related to different steps of the pipeline, enabling or disabling buttons accordingly.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the NellieVisualizer class, setting up buttons and layout for opening and visualizing images and tracks.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the Nellie instance managing the pipeline.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.scale = (1, 1, 1)

        self.im_memmap = None
        self.raw_layer = None
        self.im_branch_label_reassigned_layer = None
        self.im_branch_label_reassigned = None
        self.im_obj_label_reassigned_layer = None
        self.im_obj_label_reassigned = None
        self.im_marker_layer = None
        self.im_marker = None
        self.im_skel_relabelled_layer = None
        self.im_instance_label_layer = None
        self.frangi_layer = None
        self.im_skel_relabelled = None
        self.im_instance_label = None
        self.im_frangi = None

        # Raw im button
        self.raw_button = QPushButton(text="Open raw image")
        self.raw_button.clicked.connect(self.open_raw)
        self.raw_button.setEnabled(True)

        # Preprocess open button
        self.open_preprocess_button = QPushButton(text="Open preprocessed image")
        self.open_preprocess_button.clicked.connect(self.open_preprocess_image)
        self.open_preprocess_button.setEnabled(False)

        # Segment open button
        self.open_segment_button = QPushButton(text="Open segmentation images")
        self.open_segment_button.clicked.connect(self.open_segment_image)
        self.open_segment_button.setEnabled(False)

        # open im button
        self.open_mocap_button = QPushButton(text="Open mocap marker image")
        self.open_mocap_button.clicked.connect(self.open_mocap_image)
        self.open_mocap_button.setEnabled(False)

        # open reassign button
        self.open_reassign_button = QPushButton(text="Open reassigned labels image")
        self.open_reassign_button.clicked.connect(self.open_reassign_image)
        self.open_reassign_button.setEnabled(False)

        self.track_button = QPushButton(text="Visualize selected label's tracks")
        self.track_button.clicked.connect(self.on_track_selected)
        self.track_button.setEnabled(False)

        self.track_all_button = QPushButton(text="Visualize all frame labels' tracks")
        self.track_all_button.clicked.connect(self.track_all)
        self.track_all_button.setEnabled(False)

        self.set_ui()

        # self.layout = QGridLayout()
        # self.setLayout(self.layout)
        #
        # self.layout.addWidget(self.raw_button, 1, 0, 1, 1)
        # self.layout.addWidget(self.open_preprocess_button, 2, 0, 1, 1)
        # self.layout.addWidget(self.open_segment_button, 3, 0, 1, 1)
        # self.layout.addWidget(self.open_mocap_button, 4, 0, 1, 1)
        # self.layout.addWidget(self.open_reassign_button, 5, 0, 1, 1)

        self.initialized = False

    def set_ui(self):
        """
        Initializes and sets the layout and UI components for the NellieVisualizer. It groups the buttons for image
        and track visualization into separate sections and arranges them within a vertical layout.
        """
        main_layout = QVBoxLayout()

        # visualization group
        visualization_group = QGroupBox("Image visualization")
        visualization_layout = QVBoxLayout()
        visualization_layout.addWidget(self.raw_button)
        visualization_layout.addWidget(self.open_preprocess_button)
        visualization_layout.addWidget(self.open_segment_button)
        visualization_layout.addWidget(self.open_mocap_button)
        visualization_layout.addWidget(self.open_reassign_button)
        visualization_group.setLayout(visualization_layout)

        # tracking group
        tracking_group = QGroupBox("Track visualization")
        tracking_layout = QVBoxLayout()
        tracking_layout.addWidget(self.track_button)
        tracking_layout.addWidget(self.track_all_button)
        tracking_group.setLayout(tracking_layout)

        main_layout.addWidget(visualization_group)
        main_layout.addWidget(tracking_group)
        self.setLayout(main_layout)

    def post_init(self):
        """
        Post-initialization method that sets the image scale based on the image resolution and makes the scale bar visible.
        """
        self.set_scale()
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'
        self.initialized = True

    def check_3d(self):
        """
        Ensures that the napari viewer is in 3D mode if the dataset contains Z-dimension data.
        """
        if not self.nellie.im_info.no_z and self.viewer.dims.ndim != 3:
            # ndimensions should be 3 for viewer
            self.viewer.dims.ndim = 3
            self.viewer.dims.ndisplay = 3

    def set_scale(self):
        """
        Sets the scale for image display based on the resolution of the Z, Y, and X dimensions of the image.
        """
        dim_res = self.nellie.im_info.dim_res
        if self.nellie.im_info.no_z:
            self.scale = (dim_res['Y'], dim_res['X'])
        else:
            self.scale = (dim_res['Z'], dim_res['Y'], dim_res['X'])

    def open_preprocess_image(self):
        """
        Opens and displays the preprocessed (Frangi-filtered) image in the napari viewer.
        """
        self.im_frangi = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_preprocessed'])
        self.check_3d()
        self.frangi_layer = self.viewer.add_image(self.im_frangi, name='Pre-processed', colormap='turbo', scale=self.scale)
        self.frangi_layer.interpolation = 'nearest'
        self.viewer.layers.selection.active = self.frangi_layer

    def open_segment_image(self):
        """
        Opens and displays the segmentation labels (skeleton relabeled and instance labels) in the napari viewer.
        """
        self.im_instance_label = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_instance_label'])
        self.im_skel_relabelled = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_skel_relabelled'])

        self.check_3d()

        self.im_skel_relabelled_layer = self.viewer.add_labels(self.im_skel_relabelled, name='Labels: Branches',
                                                               opacity=1, scale=self.scale, visible=False)
        self.im_instance_label_layer = self.viewer.add_labels(self.im_instance_label, name='Labels: Organelles',
                                                              opacity=1, scale=self.scale)
        self.viewer.layers.selection.active = self.im_instance_label_layer

    def on_track_selected(self):
        """
        Visualizes the tracks for the currently selected label in the napari viewer, based on the active image layer.
        """
        # if flow_vector_array path does not point to an existing file, return
        if not os.path.exists(self.nellie.im_info.pipeline_paths['flow_vector_array']):
            return

        # pos = None
        # label = 0
        layer = self.viewer.layers.selection.active
        t = int(self.viewer.dims.current_step[0])
        # if event.button == 1 and event.is_dragging is False and 'Alt' in event.modifiers:
        #     scaled_pos = self.viewer.cursor.position
        #     pos = [scaled_pos[i+1] / self.scale[i] for i in range(len(scaled_pos)-1)]
        #     pos = (scaled_pos[0], *pos)
        #     pos = tuple(int(pos_dim) for pos_dim in pos)
        #     label = layer.selected_label
        # if pos is None:
        #     return


        if layer == self.im_instance_label_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_instance_label']
        elif layer == self.im_skel_relabelled_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_skel_relabelled']
        elif layer == self.im_branch_label_reassigned_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_branch_label_reassigned']
        elif layer == self.im_obj_label_reassigned_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_obj_label_reassigned']
        else:
            return

        label = layer.selected_label
        if label == 0:
            return

        label_tracks = LabelTracks(im_info=self.nellie.im_info, label_im_path=label_path)
        label_tracks.initialize()
        all_tracks = []
        all_props = {}
        max_track_num = 0
        if self.nellie.settings.track_all_frames.isChecked() and (
                layer == self.im_branch_label_reassigned_layer or layer == self.im_obj_label_reassigned_layer):
            for frame in range(self.nellie.im_info.shape[0]):
                tracks, track_properties = label_tracks.run(label_num=label, start_frame=frame, end_frame=None,
                                                            min_track_num=max_track_num,
                                                            skip_coords=self.nellie.settings.skip_vox.value())
                all_tracks += tracks
                for property in track_properties.keys():
                    if property not in all_props.keys():
                        all_props[property] = []
                    all_props[property] += track_properties[property]
                if len(tracks) == 0:
                    break
                max_track_num = max([track[0] for track in tracks])+1
        else:
            all_tracks, all_props = label_tracks.run(label_num=label, start_frame=t, end_frame=None,
                                                     skip_coords=self.nellie.settings.skip_vox.value())
        if len(all_tracks) == 0:
            return
        self.viewer.add_tracks(all_tracks, properties=all_props, name=f'Tracks: Label {label}', scale=self.scale)
        self.viewer.layers.selection.active = layer
        self.check_file_existence()

    def track_all(self):
        """
        Visualizes tracks for all labels across frames in the napari viewer, based on the active image layer.
        """
        layer = self.viewer.layers.selection.active
        if layer == self.im_instance_label_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_instance_label']
        elif layer == self.im_skel_relabelled_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_skel_relabelled']
        elif layer == self.im_branch_label_reassigned_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_branch_label_reassigned']
        elif layer == self.im_obj_label_reassigned_layer:
            label_path = self.nellie.im_info.pipeline_paths['im_obj_label_reassigned']
        else:
            return

        t = int(self.viewer.dims.current_step[0])
        label_tracks = LabelTracks(im_info=self.nellie.im_info, label_im_path=label_path)
        label_tracks.initialize()
        all_tracks = []
        all_props = {}
        max_track_num = 0
        if self.nellie.settings.track_all_frames.isChecked() and (
                layer == self.im_branch_label_reassigned_layer or layer == self.im_obj_label_reassigned_layer):
            for frame in range(self.nellie.im_info.shape[0]):
                tracks, track_properties = label_tracks.run(start_frame=frame, end_frame=None,
                                                            min_track_num=max_track_num,
                                                            skip_coords=self.nellie.settings.skip_vox.value())
                all_tracks += tracks
                for property in track_properties.keys():
                    if property not in all_props.keys():
                        all_props[property] = []
                    all_props[property] += track_properties[property]
                if len(tracks) == 0:
                    break
                max_track_num = max([track[0] for track in tracks]) + 1
        else:
            all_tracks, all_props = label_tracks.run(start_frame=t, end_frame=None,
                                                     skip_coords=self.nellie.settings.skip_vox.value())
        if len(all_tracks) == 0:
            return
        self.viewer.add_tracks(all_tracks, properties=all_props, name=f'Tracks: All labels', scale=self.scale)
        self.viewer.layers.selection.active = layer
        self.check_file_existence()

    def open_mocap_image(self):
        """
        Opens and displays the mocap marker image in the napari viewer.
        """
        self.check_3d()

        self.im_marker = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_marker'])
        self.im_marker_layer = self.viewer.add_image(self.im_marker, name='Mocap Markers', colormap='red',
                                                     blending='additive', contrast_limits=[0, 1], scale=self.scale)
        self.im_marker_layer.interpolation = 'nearest'
        self.viewer.layers.selection.active = self.im_marker_layer

    def open_reassign_image(self):
        """
        Opens and displays the reassigned branch and object labels in the napari viewer.
        """
        self.check_3d()

        self.im_branch_label_reassigned = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_branch_label_reassigned'])
        self.im_branch_label_reassigned_layer = self.viewer.add_labels(self.im_branch_label_reassigned, name='Reassigned px: Branches', scale=self.scale, visible=False)

        self.im_obj_label_reassigned = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_obj_label_reassigned'])
        self.im_obj_label_reassigned_layer = self.viewer.add_labels(self.im_obj_label_reassigned, name='Reassigned px: Organelles', scale=self.scale)
        self.viewer.layers.selection.active = self.im_obj_label_reassigned_layer

    def open_raw(self):
        """
        Opens and displays the raw image in the napari viewer.
        """
        self.check_3d()
        self.im_memmap = tifffile.memmap(self.nellie.im_info.im_path)
        self.raw_layer = self.viewer.add_image(self.im_memmap, name='raw', colormap='gray',
                                               blending='additive', scale=self.scale)
        # make 3d interpolation to nearest
        self.raw_layer.interpolation = 'nearest'
        self.viewer.layers.selection.active = self.raw_layer

    def check_file_existence(self):
        """
        Checks for the existence of files related to different steps of the pipeline, enabling or disabling buttons accordingly.
        """
        # set all other buttons to disabled first
        self.raw_button.setEnabled(False)
        self.open_preprocess_button.setEnabled(False)
        self.open_segment_button.setEnabled(False)
        self.open_mocap_button.setEnabled(False)
        self.open_reassign_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.track_all_button.setEnabled(False)

        if os.path.exists(self.nellie.im_info.im_path):
            self.raw_button.setEnabled(True)
        else:
            self.raw_button.setEnabled(False)

        frangi_path = self.nellie.im_info.pipeline_paths['im_preprocessed']
        if os.path.exists(frangi_path):
            self.open_preprocess_button.setEnabled(True)
        else:
            self.open_preprocess_button.setEnabled(False)

        im_instance_label_path = self.nellie.im_info.pipeline_paths['im_instance_label']
        im_skel_relabelled_path = self.nellie.im_info.pipeline_paths['im_skel_relabelled']
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.open_segment_button.setEnabled(True)
            self.nellie.settings.skip_vox.setEnabled(True)
            self.nellie.settings.track_all_frames.setEnabled(True)
            self.track_button.setEnabled(True)
            self.track_all_button.setEnabled(True)
            # self.viewer.help = 'Alt + click a label to see its tracks'
        else:
            self.open_segment_button.setEnabled(False)
            self.nellie.settings.skip_vox.setEnabled(False)
            self.nellie.settings.track_all_frames.setEnabled(False)

        im_marker_path = self.nellie.im_info.pipeline_paths['im_marker']
        if os.path.exists(im_marker_path):
            self.open_mocap_button.setEnabled(True)
        else:
            self.open_mocap_button.setEnabled(False)

        im_branch_label_path = self.nellie.im_info.pipeline_paths['im_branch_label_reassigned']
        if os.path.exists(im_branch_label_path):
            self.open_reassign_button.setEnabled(True)
            self.track_button.setEnabled(True)
            self.track_all_button.setEnabled(True)
            # self.viewer.help = 'Alt + click a label to see its tracks'
        else:
            self.open_reassign_button.setEnabled(False)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
