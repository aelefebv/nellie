import os

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QSpinBox, QCheckBox
from nellie import logger
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from nellie.utils.general import get_reshaped_image


class NellieProcessor(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        # Label above the spinner box
        self.channel_label = QLabel("Channel to analyze:")

        self.channel_input = QSpinBox()
        self.channel_input.setRange(0, 0)
        self.channel_input.setValue(0)
        self.channel_input.setEnabled(False)
        self.channel_input.valueChanged.connect(self.change_channel)

        # Label above the spinner box
        self.time_label = QLabel("Number of temporal frames:")

        self.time_input = QSpinBox()
        self.time_input.setRange(1, 1)
        self.time_input.setValue(1)
        self.time_input.setEnabled(False)
        self.time_input.valueChanged.connect(self.change_t)
        self.num_t = 1

        # Checkbox for 'Remove edges'
        self.remove_edges_checkbox = QCheckBox("Remove image edges")
        self.remove_edges_checkbox.setEnabled(False)
        self.remove_edges_checkbox.setToolTip(
            "Originally for Snouty deskewed images. If you see weird image edge artifacts, enable this.")

        # Run im button
        self.run_button = QPushButton(text="Run Nellie")
        self.run_button.clicked.connect(self.run_nellie)
        self.run_button.setEnabled(False)

        # Preprocess im button
        self.preprocess_button = QPushButton(text="Run preprocessing")
        self.preprocess_button.clicked.connect(self.run_preprocessing)
        self.preprocess_button.setEnabled(False)

        # Segment im button
        self.segment_button = QPushButton(text="Run segmentation")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setEnabled(False)

        # Run mocap button
        self.mocap_button = QPushButton(text="Run mocap marking")
        self.mocap_button.clicked.connect(self.run_mocap)
        self.mocap_button.setEnabled(False)


        # Run tracking button
        self.track_button = QPushButton(text="Run tracking")
        self.track_button.clicked.connect(self.run_tracking)
        self.track_button.setEnabled(False)

        # Run reassign button
        self.reassign_button = QPushButton(text="Run voxel reassignment")
        self.reassign_button.clicked.connect(self.run_reassign)

        self.reassign_button.setEnabled(False)

        # Run feature extraction button
        self.feature_export_button = QPushButton(text="Run feature export")
        self.feature_export_button.clicked.connect(self.run_feature_export)
        self.feature_export_button.setEnabled(False)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add buttons
        self.layout.addWidget(self.channel_label, 0, 0)
        self.layout.addWidget(self.channel_input, 0, 1)
        self.layout.addWidget(self.time_label, 1, 0)
        self.layout.addWidget(self.time_input, 1, 1)
        self.layout.addWidget(self.remove_edges_checkbox, 2, 1)

        self.layout.addWidget(QLabel("Run full pipeline"), 42, 0, 1, 2)
        self.layout.addWidget(self.run_button, 43, 0, 1, 2)

        self.layout.addWidget(QLabel("Run individual steps / Visualize"), 44, 0, 1, 2)
        self.layout.addWidget(self.preprocess_button, 45, 0, 1, 2)
        self.layout.addWidget(self.segment_button, 46, 0, 1, 2)
        self.layout.addWidget(self.mocap_button, 47, 0, 1, 2)
        self.layout.addWidget(self.track_button, 48, 0, 1, 2)
        self.layout.addWidget(self.feature_export_button, 49, 0, 1, 2)
        self.layout.addWidget(self.reassign_button, 50, 0, 1, 2)

        self.im_memmap = None
        self.raw_layer = None
        self.ray_layer = None
        self.ray_near_far = None
        self.point_layer = None

        self.animation_timer = None
        self.animation_start_time = None
        self.current_angles = None
        self.start_angle = None
        self.tracked_coords = None
        self.track_layer = None
        self.tracks = []

    def post_init(self):
        if not self.check_for_raw():
            return
        self.check_file_existence()
        self.remove_edges_checkbox.setEnabled(True)
        
    def check_file_existence(self):
        self.nellie.visualizer.check_file_existence()

        # set all other buttons to disabled first
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)

        frangi_path = self.nellie.im_info.pipeline_paths['im_frangi']
        if os.path.exists(frangi_path):
            self.segment_button.setEnabled(True)
        else:
            self.segment_button.setEnabled(False)
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_instance_label_path = self.nellie.im_info.pipeline_paths['im_instance_label']
        im_skel_relabelled_path = self.nellie.im_info.pipeline_paths['im_skel_relabelled']
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.mocap_button.setEnabled(True)
        else:
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_marker_path = self.nellie.im_info.pipeline_paths['im_marker']
        if os.path.exists(im_marker_path):
            self.track_button.setEnabled(True)
        else:
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        track_path = self.nellie.im_info.pipeline_paths['flow_vector_array']
        if os.path.exists(track_path):
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)
        else:
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            # return

        analysis_path = self.nellie.im_info.pipeline_paths['adjacency_maps']
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
            self.nellie.analyzer.post_init()
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)


    def run_preprocessing(self):
        preprocessing = Filter(im_info=self.nellie.im_info, num_t=self.num_t,
                               remove_edges=self.remove_edges_checkbox.isChecked())
        preprocessing.run()

        self.check_file_existence()

    def run_segmentation(self):
        segmenting = Label(im_info=self.nellie.im_info, num_t=self.num_t)
        segmenting.run()
        networking = Network(im_info=self.nellie.im_info, num_t=self.num_t)
        networking.run()

        self.check_file_existence()

    def run_mocap(self):
        mocap_marking = Markers(im_info=self.nellie.im_info, num_t=self.num_t)
        mocap_marking.run()

        self.check_file_existence()

    def run_tracking(self):
        hu_tracking = HuMomentTracking(im_info=self.nellie.im_info, num_t=self.num_t)
        hu_tracking.run()

        self.check_file_existence()

    def run_reassign(self):
        vox_reassign = VoxelReassigner(im_info=self.nellie.im_info, num_t=self.num_t)
        vox_reassign.run()

        self.check_file_existence()

    def run_feature_export(self):
        hierarchy = Hierarchy(im_info=self.nellie.im_info, num_t=self.num_t)
        hierarchy.run()

        self.check_file_existence()

    def run_nellie(self):
        self.run_preprocessing()
        self.run_segmentation()
        self.run_mocap()
        self.run_tracking()
        self.run_reassign()
        self.run_feature_export()

        self.check_file_existence()

    def check_for_raw(self):
        self.preprocess_button.setEnabled(False)
        self.run_button.setEnabled(False)
        try:
            # todo implement lazy loading if wanted
            im_memmap = self.nellie.im_info.get_im_memmap(self.nellie.im_info.im_path)
            self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.nellie.im_info)
            self.preprocess_button.setEnabled(True)
            self.run_button.setEnabled(True)
            return True
        except Exception as e:
            logger.error(e)
            show_info(f"Could not open raw image: {e}")
            self.im_memmap = None
            return False

    def change_channel(self):
        if self.nellie.file_select.single:
            self.nellie.file_select.initialize_single_file()
        else:
            self.nellie.file_select.initialize_folder(self.nellie.file_select.filepath)

    def change_t(self):
        self.num_t = self.time_input.value()

    # def find_closest_voxel(self, near2, far2):
    #     near1, far1 = self.ray_near_far
    #     time_point = near1[0]
    #     near1 = near1[1:]
    #     far1 = far1[1:]
    #     near2 = near2[1:]
    #     far2 = far2[1:]
    #
    #     p1 = np.array(near1)
    #     d1 = np.array(far1) - p1
    #
    #     p2 = np.array(near2)
    #     d2 = np.array(far2) - p2
    #
    #     show_info(f"{p1}, {d1}, {p2}, {d2}")
    #
    #     # normalize
    #     d1 /= np.linalg.norm(d1)
    #     d2 /= np.linalg.norm(d2)
    #     show_info(f"{p1}, {d1}, {p2}, {d2}")
    #
    #     # cross
    #     cross_d1_d2 = np.cross(d1, d2)
    #
    #     # check non-parallel
    #     if np.linalg.norm(cross_d1_d2) < 1e-6:
    #         logger.error("Rays are parallel or coincident.")
    #         return
    #
    #     # Calculate the line segment connecting the two lines that is perpendicular to both
    #     # This is done by solving the system of linear equations formed by the two lines
    #     a = np.array([d1, -d2, cross_d1_d2]).T
    #     b = p2 - p1
    #     t, s, _ = np.linalg.solve(a, b)
    #
    #     # Calculate the closest points on the original lines
    #     closest_point_on_line1 = p1 + d1 * t
    #     closest_point_on_line2 = p2 + d2 * s
    #
    #     # Find the midpoint of the shortest line connecting the two rays
    #     midpoint = (closest_point_on_line1 + closest_point_on_line2) / 2
    #     # add time point to midpoint coords
    #     midpoint = np.insert(midpoint, 0, time_point)
    #     # Convert the midpoint to voxel coordinates (assumes the raw layer is the image)
    #     voxel_coords = np.round(self.raw_layer.world_to_data(midpoint)).astype(int)
    #     show_info(f"Midpoint: {midpoint}, Voxel coords: {voxel_coords}")
    #     # self.point_layer = self.viewer.add_points(voxel_coords, name='midpoint', face_color='red', size=2)
    #     self.tracked_coords = voxel_coords
    #
    #     # Check if the voxel_coords are within the bounds of the image
    #     if np.all(voxel_coords >= 0) and np.all(voxel_coords < self.im_memmap.shape):
    #         # Here you can do something with the voxel coordinates, like highlighting the voxel
    #         logger.info(f"Closest voxel at: {voxel_coords}")
    #     else:
    #         logger.error("Calculated voxel coordinates are out of bounds.")

    # def animate_camera_angle(self):
    #     # Calculate the time elapsed since the start of the animation
    #     time_elapsed = time.time() - self.animation_start_time
    #     duration = 0.5  # Total duration of the animation in seconds
    #
    #     if time_elapsed < duration:
    #         # Calculate the intermediate angle based on the elapsed time
    #         fraction = time_elapsed / duration
    #         delta_angle = 45 * fraction  # Change in angle should be 45 degrees
    #         new_angle = self.start_angle + delta_angle
    #
    #         # Update the camera angle
    #         self.viewer.camera.angles = (self.current_angles[0], self.current_angles[1], new_angle)
    #     else:
    #         # Animation is complete, set the final angle
    #         self.viewer.camera.angles = (self.current_angles[0], self.current_angles[1], self.start_angle + 45)
    #         self.animation_timer.stop()  # Stop the timer

    # def get_tracks(self):
    #     if self.tracked_coords is None:
    #         return
    #     t_current = self.tracked_coords[0]
    #     coord_to_track = np.asarray([self.tracked_coords[1:]])
    #     tracks_forward, track_props_forward = interpolate_all_forward(coord_to_track, t_current,
    #                                                                   self.num_t - 1, self.nellie.im_info)
    #     tracks_back, track_props_back = interpolate_all_backward(coord_to_track, t_current,
    #                                                             1, self.nellie.im_info)
    #     self.track_layer = self.viewer.add_tracks(tracks_forward, properties=track_props_forward, name='tracks', scale=self.scale)
    #     self.track_layer = self.viewer.add_tracks(tracks_back, properties=track_props_back, name='tracks', scale=self.scale)

    # def on_click(self, layer, event):  # https://napari.org/dev/guides/3D_interactivity.html
    #     near, far = None, None
    #     if event.button == 1 and event.is_dragging is False and 'Alt' in event.modifiers:
    #         near, far = layer.get_ray_intersections(event.position, event.view_direction, event.dims_displayed)
    #     if near is None or far is None:
    #         return
    #
    #     # add line to viewer
    #     if self.ray_layer is None:
    #         self.ray_near_far = (near, far)
    #         self.ray_layer = self.viewer.add_shapes(
    #             np.asarray([[*near], [*far]]),
    #             shape_type='line',
    #             edge_color='red',
    #             edge_width=1,
    #             name='ray',
    #             opacity=0.1, scale=self.scale
    #         )
    #         self.viewer.layers.selection.active = self.viewer.layers[0]
    #
    #         # Store the current angles and start angle for the animation
    #         self.current_angles = self.viewer.camera.angles
    #         self.start_angle = self.current_angles[2]
    #
    #         # Start the animation
    #         self.animation_start_time = time.time()
    #         self.animation_timer = QTimer()
    #         self.animation_timer.timeout.connect(self.animate_camera_angle)
    #         self.animation_timer.start(1000 // 60)  # Update at 60 FPS
    #     else:
    #         self.find_closest_voxel(near, far)
    #         # remove ray layer
    #         self.viewer.layers.remove(self.ray_layer)
    #         self.ray_layer = None
    #         self.ray_near_far = None
    #         self.get_tracks()


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
