import os

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QGridLayout, QWidget, QPushButton, QLabel, QSpinBox, QCheckBox

from nellie import logger
from nellie.tracking.all_tracks_for_label import LabelTracks
from nellie.utils.general import get_reshaped_image


class NellieVisualizer(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.scale = (1, 1, 1)

        self.im_memmap = None
        self.num_t = None

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
        self.raw_button.setEnabled(False)

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

        # Label above the spinner box
        self.skip_vox_label = QLabel("Track every N voxel. N=")

        self.skip_vox = QSpinBox()
        self.skip_vox.setRange(1, 10000)
        self.skip_vox.setValue(5)
        self.skip_vox.setEnabled(False)

        # Track all frames
        self.track_all_frames = QCheckBox("Track all frames' voxels")
        self.track_all_frames.setChecked(True)
        self.track_all_frames.setEnabled(False)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.raw_button, 1, 0, 1, 1)
        self.layout.addWidget(self.open_preprocess_button, 2, 0, 1, 1)
        self.layout.addWidget(self.open_segment_button, 3, 0, 1, 1)
        self.layout.addWidget(self.open_mocap_button, 4, 0, 1, 1)
        self.layout.addWidget(self.open_reassign_button, 5, 0, 1, 1)
        self.layout.addWidget(self.skip_vox_label, 6, 0, 1, 1)
        self.layout.addWidget(self.skip_vox, 6, 1, 1, 1)
        self.layout.addWidget(self.track_all_frames, 7, 0, 1, 1)

    def post_init(self):
        if not self.check_for_raw():
            return
        self.num_t = self.nellie.processor.time_input.value()
        self.set_scale()
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'

    def check_3d(self):
        if not self.nellie.im_info.no_z and self.viewer.dims.ndim != 3:
            # ndimensions should be 3 for viewer
            self.viewer.dims.ndim = 3
            self.viewer.dims.ndisplay = 3

    def check_for_raw(self):
        self.raw_button.setEnabled(False)
        try:
            # todo implement lazy loading if wanted
            im_memmap = self.nellie.im_info.get_im_memmap(self.nellie.im_info.im_path)
            self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.nellie.im_info)
            self.raw_button.setEnabled(True)
            return True
        except Exception as e:
            logger.error(e)
            show_info(f"Could not open raw image: {e}")
            self.im_memmap = None
            return False

    def set_scale(self):
        dim_sizes = self.nellie.im_info.dim_sizes
        if self.nellie.im_info.no_z:
            self.scale = (dim_sizes['Y'], dim_sizes['X'])
        else:
            self.scale = (dim_sizes['Z'], dim_sizes['Y'], dim_sizes['X'])

    def open_preprocess_image(self):
        im_frangi = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_frangi'])
        self.im_frangi = get_reshaped_image(im_frangi, self.num_t, self.nellie.im_info)

        self.check_3d()

        self.frangi_layer = self.viewer.add_image(self.im_frangi, name='pre processed', colormap='turbo', scale=self.scale)
        self.frangi_layer.interpolation = 'nearest'

    def open_segment_image(self):
        im_instance_label = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_instance_label'])
        self.im_instance_label = get_reshaped_image(im_instance_label, self.num_t, self.nellie.im_info)
        im_skel_relabelled = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_skel_relabelled'])
        self.im_skel_relabelled = get_reshaped_image(im_skel_relabelled, self.num_t, self.nellie.im_info)

        self.check_3d()

        self.im_instance_label_layer = self.viewer.add_labels(self.im_instance_label, name='segmentation: objects',
                                                              opacity=1, scale=self.scale)
        self.im_skel_relabelled_layer = self.viewer.add_labels(self.im_skel_relabelled, name='segmentation: branches',
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

        label_tracks = LabelTracks(im_info=self.nellie.im_info, num_t=self.num_t, label_im_path=label_path)
        label_tracks.initialize()
        all_tracks = []
        all_props = {}
        max_track_num = 0
        if self.track_all_frames.isChecked() and (
                layer == self.im_branch_label_reassigned_layer or layer == self.im_obj_label_reassigned_layer):
            for frame in range(self.num_t):
                tracks, track_properties = label_tracks.run(label_num=label, start_frame=frame, end_frame=None,
                                                            min_track_num=max_track_num,
                                                            skip_coords=self.skip_vox.value())
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
                                                     skip_coords=self.skip_vox.value())
        if len(all_tracks) == 0:
            return
        self.viewer.add_tracks(all_tracks, properties=all_props, name=f'tracks: {label}', scale=self.scale)
        self.viewer.layers.selection.active = layer
        self.check_file_existence()

    def open_mocap_image(self):
        self.check_3d()

        im_marker = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_marker'])
        self.im_marker = get_reshaped_image(im_marker, self.num_t, self.nellie.im_info)
        self.im_marker_layer = self.viewer.add_image(self.im_marker, name='mocap markers', colormap='red',
                                                     blending='additive', contrast_limits=[0, 1], scale=self.scale)
        self.im_marker_layer.interpolation = 'nearest'

    def open_reassign_image(self):
        self.check_3d()

        im_branch_label_reassigned = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_branch_label_reassigned'])
        self.im_branch_label_reassigned = get_reshaped_image(im_branch_label_reassigned, self.num_t, self.nellie.im_info)
        self.im_branch_label_reassigned_layer = self.viewer.add_labels(self.im_branch_label_reassigned, name='reassigned branch voxels', scale=self.scale)
        self.im_branch_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

        im_obj_label_reassigned = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_obj_label_reassigned'])
        self.im_obj_label_reassigned = get_reshaped_image(im_obj_label_reassigned, self.num_t, self.nellie.im_info)
        self.im_obj_label_reassigned_layer = self.viewer.add_labels(self.im_obj_label_reassigned, name='reassigned object voxels', scale=self.scale)
        self.im_obj_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

    def open_raw(self):
        self.check_3d()
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
            self.skip_vox.setEnabled(True)
            self.track_all_frames.setEnabled(True)
            self.viewer.help = 'Alt + click a label to see its tracks'
        else:
            self.open_segment_button.setEnabled(False)
            self.skip_vox.setEnabled(False)
            self.track_all_frames.setEnabled(False)
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
        else:
            self.open_reassign_button.setEnabled(False)
            return


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
