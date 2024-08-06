import os

from qtpy.QtWidgets import QGridLayout, QWidget, QPushButton
from tifffile import tifffile

from nellie import logger
from nellie.tracking.all_tracks_for_label import LabelTracks


class NellieVisualizer(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
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

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.raw_button, 1, 0, 1, 1)
        self.layout.addWidget(self.open_preprocess_button, 2, 0, 1, 1)
        self.layout.addWidget(self.open_segment_button, 3, 0, 1, 1)
        self.layout.addWidget(self.open_mocap_button, 4, 0, 1, 1)
        self.layout.addWidget(self.open_reassign_button, 5, 0, 1, 1)

        self.initialized = False

    def post_init(self):
        self.set_scale()
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'
        self.initialized = True

    def check_3d(self):
        if not self.nellie.im_info.no_z and self.viewer.dims.ndim != 3:
            # ndimensions should be 3 for viewer
            self.viewer.dims.ndim = 3
            self.viewer.dims.ndisplay = 3

    def set_scale(self):
        dim_res = self.nellie.im_info.dim_res
        if self.nellie.im_info.no_z:
            self.scale = (dim_res['Y'], dim_res['X'])
        else:
            self.scale = (dim_res['Z'], dim_res['Y'], dim_res['X'])

    def open_preprocess_image(self):
        self.im_frangi = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_frangi'])
        self.check_3d()
        self.frangi_layer = self.viewer.add_image(self.im_frangi, name='Pre-processed', colormap='turbo', scale=self.scale)
        self.frangi_layer.interpolation = 'nearest'

    def open_segment_image(self):
        self.im_instance_label = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_instance_label'])
        self.im_skel_relabelled = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_skel_relabelled'])

        self.check_3d()

        self.im_instance_label_layer = self.viewer.add_labels(self.im_instance_label, name='Labels: Organelles',
                                                              opacity=1, scale=self.scale)
        self.im_skel_relabelled_layer = self.viewer.add_labels(self.im_skel_relabelled, name='Labels: Branches',
                                                               opacity=1, scale=self.scale)
        self.im_instance_label_layer.mouse_drag_callbacks.append(self.on_label_click)
        self.im_skel_relabelled_layer.mouse_drag_callbacks.append(self.on_label_click)

    def on_label_click(self, layer, event):
        # if flow_vector_array path does not point to an existing file, return
        if not os.path.exists(self.nellie.im_info.pipeline_paths['flow_vector_array']):
            return

        pos = None
        if event.button == 1 and event.is_dragging is False and 'Alt' in event.modifiers:
            scaled_pos = self.viewer.cursor.position
            pos = [scaled_pos[i+1] / self.scale[i] for i in range(len(scaled_pos)-1)]
            pos = (scaled_pos[0], *pos)
        if pos is None:
            return

        try:
            pos = tuple(int(pos_dim) for pos_dim in pos)
            label = layer.data[pos]
        except Exception as e:
            logger.error(e)
            return

        if label == 0:
            return

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
            all_tracks, all_props = label_tracks.run(label_num=label, start_frame=pos[0], end_frame=None,
                                                     skip_coords=self.nellie.settings.skip_vox.value())
        if len(all_tracks) == 0:
            return
        self.viewer.add_tracks(all_tracks, properties=all_props, name=f'Tracks: Label {label}', scale=self.scale)
        self.viewer.layers.selection.active = layer
        self.check_file_existence()

    def open_mocap_image(self):
        self.check_3d()

        self.im_marker = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_marker'])
        self.im_marker_layer = self.viewer.add_image(self.im_marker, name='Mocap Markers', colormap='red',
                                                     blending='additive', contrast_limits=[0, 1], scale=self.scale)
        self.im_marker_layer.interpolation = 'nearest'

    def open_reassign_image(self):
        self.check_3d()

        self.im_branch_label_reassigned = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_branch_label_reassigned'])
        self.im_branch_label_reassigned_layer = self.viewer.add_labels(self.im_branch_label_reassigned, name='Reassigned px: Branches', scale=self.scale)
        self.im_branch_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

        self.im_obj_label_reassigned = tifffile.memmap(self.nellie.im_info.pipeline_paths['im_obj_label_reassigned'])
        self.im_obj_label_reassigned_layer = self.viewer.add_labels(self.im_obj_label_reassigned, name='Reassigned px: Organelles', scale=self.scale)
        self.im_obj_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

    def open_raw(self):
        self.check_3d()
        self.im_memmap = tifffile.memmap(self.nellie.im_info.im_path)
        self.raw_layer = self.viewer.add_image(self.im_memmap, name='raw', colormap='gray',
                                               blending='additive', scale=self.scale)
        # make 3d interpolation to nearest
        self.raw_layer.interpolation = 'nearest'

    def check_file_existence(self):
        # set all other buttons to disabled first
        self.open_preprocess_button.setEnabled(False)
        self.open_segment_button.setEnabled(False)
        self.open_mocap_button.setEnabled(False)
        self.open_reassign_button.setEnabled(False)

        frangi_path = self.nellie.im_info.pipeline_paths['im_frangi']
        if os.path.exists(frangi_path):
            self.open_preprocess_button.setEnabled(True)
        else:
            self.open_preprocess_button.setEnabled(False)
            return

        im_instance_label_path = self.nellie.im_info.pipeline_paths['im_instance_label']
        im_skel_relabelled_path = self.nellie.im_info.pipeline_paths['im_skel_relabelled']
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.open_segment_button.setEnabled(True)
            self.nellie.settings.skip_vox.setEnabled(True)
            self.nellie.settings.track_all_frames.setEnabled(True)
            self.viewer.help = 'Alt + click a label to see its tracks'
        else:
            self.open_segment_button.setEnabled(False)
            self.nellie.settings.skip_vox.setEnabled(False)
            self.nellie.settings.track_all_frames.setEnabled(False)
            return

        im_marker_path = self.nellie.im_info.pipeline_paths['im_marker']
        if os.path.exists(im_marker_path):
            self.open_mocap_button.setEnabled(True)
        else:
            self.open_mocap_button.setEnabled(False)
            return

        im_branch_label_path = self.nellie.im_info.pipeline_paths['im_branch_label_reassigned']
        if os.path.exists(im_branch_label_path):
            self.open_reassign_button.setEnabled(True)
            self.viewer.help = 'Alt + click a label to see its tracks'
        else:
            self.open_reassign_button.setEnabled(False)
            return


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
