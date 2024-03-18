import os
import time

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QCheckBox, QGridLayout
from napari.utils.notifications import show_info
from nellie import logger
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.all_tracks_for_label import LabelTracks
from nellie.tracking.flow_interpolation import interpolate_all_forward, interpolate_all_backward
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from nellie.utils.general import get_reshaped_image
import datetime
import tifffile

from nellie_napari.nellie_analysis import NellieAnalysis


class NellieProcessor(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer
        self.im_info = None
        self.num_t = None
        self.remove_edges = False
        self.nellie_analyzer = None

        self.scale = (1, 1, 1)

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

        # self.setLayout(QVBoxLayout())

        # Raw im button
        self.raw_button = QPushButton(text="Open raw image")
        self.raw_button.clicked.connect(self.open_raw)
        self.raw_button.setEnabled(False)

        # Run im button
        self.run_button = QPushButton(text="Run Nellie")
        self.run_button.clicked.connect(self.run_nellie)
        self.run_button.setEnabled(False)

        # Preprocess im button
        self.preprocess_button = QPushButton(text="Run preprocessing")
        self.preprocess_button.clicked.connect(self.run_preprocessing)
        self.preprocess_button.setEnabled(False)

        # Preprocess open button
        self.open_preprocess_button = QPushButton(text="Open preprocessed image")
        self.open_preprocess_button.clicked.connect(self.open_preprocess_image)
        self.open_preprocess_button.setEnabled(False)

        # Segment im button
        self.segment_button = QPushButton(text="Run segmentation")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setEnabled(False)
        # Segment open button
        self.open_segment_button = QPushButton(text="Open segmentation images")
        self.open_segment_button.clicked.connect(self.open_segment_image)
        self.open_segment_button.setEnabled(False)

        # Run mocap button
        self.mocap_button = QPushButton(text="Run mocap marking")
        self.mocap_button.clicked.connect(self.run_mocap)
        self.mocap_button.setEnabled(False)
        # open im button
        self.open_mocap_button = QPushButton(text="Open mocap marker image")
        self.open_mocap_button.clicked.connect(self.open_mocap_image)
        self.open_mocap_button.setEnabled(False)

        # Run tracking button
        self.track_button = QPushButton(text="Run tracking")
        self.track_button.clicked.connect(self.run_tracking)
        self.track_button.setEnabled(False)

        # Run reassign button
        self.reassign_button = QPushButton(text="Run voxel reassignment")
        self.reassign_button.clicked.connect(self.run_reassign)
        # open reassign button
        self.open_reassign_button = QPushButton(text="Open reassigned labels image")
        self.open_reassign_button.clicked.connect(self.open_reassign_image)
        self.open_reassign_button.setEnabled(False)
        self.reassign_button.setEnabled(False)

        # Run feature extraction button
        self.feature_export_button = QPushButton(text="Run feature export")
        self.feature_export_button.clicked.connect(self.run_feature_export)
        self.feature_export_button.setEnabled(False)

        # analysis button
        self.analysis_button = QPushButton(text="Open analysis")
        self.analysis_button.clicked.connect(self.open_analysis)
        self.analysis_button.setEnabled(False)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add buttons
        self.layout.addWidget(QLabel("Run full pipeline"), 0, 0, 1, 2)
        self.layout.addWidget(self.run_button, 1, 0, 1, 1)
        self.layout.addWidget(self.analysis_button, 1, 1, 1, 1)
        self.layout.addWidget(QLabel("Run individual steps / Visualize"), 2, 0, 1, 2)
        self.layout.addWidget(self.raw_button, 3, 0, 1, 2)
        self.layout.addWidget(self.preprocess_button, 4, 0, 1, 1)
        self.layout.addWidget(self.open_preprocess_button, 4, 1, 1, 1)
        self.layout.addWidget(self.segment_button, 5, 0, 1, 1)
        self.layout.addWidget(self.open_segment_button, 5, 1, 1, 1)
        self.layout.addWidget(self.mocap_button, 6, 0, 1, 1)
        self.layout.addWidget(self.open_mocap_button, 6, 1, 1, 1)
        self.layout.addWidget(self.track_button, 7, 0, 1, 2)
        self.layout.addWidget(self.feature_export_button, 8, 0, 1, 2)
        self.layout.addWidget(self.reassign_button, 9, 0, 1, 1)
        self.layout.addWidget(self.open_reassign_button, 9, 1, 1, 1)

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
        self.check_for_raw()
        self.check_file_existence()
        if self.im_info.no_z:
            self.scale = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.scale = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'
        self.viewer.help = 'Alt + click a label to see its tracks'

    def check_3d(self):
        if not self.im_info.no_z and self.viewer.dims.ndim != 3:
            # ndimensions should be 3 for viewer
            self.viewer.dims.ndim = 3
            self.viewer.dims.ndisplay = 3

    def open_analysis(self):
        self.nellie_analyzer = NellieAnalysis()
        self.viewer.window.add_dock_widget(self.nellie_analyzer, name='Nellie Analyzer', area='right')
        self.nellie_analyzer.im_info = self.im_info
        self.nellie_analyzer.viewer = self.viewer
        self.nellie_analyzer.post_init()
        
    def check_file_existence(self):
        self.preprocess_button.setEnabled(True)
        # set all other buttons to disabled first
        self.open_preprocess_button.setEnabled(False)
        self.open_segment_button.setEnabled(False)
        self.open_mocap_button.setEnabled(False)
        self.open_reassign_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)
        self.analysis_button.setEnabled(False)

        frangi_path = self.im_info.pipeline_paths['im_frangi']
        if os.path.exists(frangi_path):
            self.open_preprocess_button.setEnabled(True)
            self.segment_button.setEnabled(True)
        else:
            self.open_preprocess_button.setEnabled(False)
            self.segment_button.setEnabled(False)
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            self.analysis_button.setEnabled(False)
            return

        im_instance_label_path = self.im_info.pipeline_paths['im_instance_label']
        im_skel_relabelled_path = self.im_info.pipeline_paths['im_skel_relabelled']
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.open_segment_button.setEnabled(True)
            self.mocap_button.setEnabled(True)
        else:
            self.open_segment_button.setEnabled(False)
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            self.analysis_button.setEnabled(False)
            return

        im_marker_path = self.im_info.pipeline_paths['im_marker']
        if os.path.exists(im_marker_path):
            self.open_mocap_button.setEnabled(True)
            self.track_button.setEnabled(True)
        else:
            self.open_mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            self.analysis_button.setEnabled(False)
            return

        track_path = self.im_info.pipeline_paths['flow_vector_array']
        if os.path.exists(track_path):
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)
        else:
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            self.analysis_button.setEnabled(False)
            return

        adjacency_maps_path = self.im_info.pipeline_paths['adjacency_maps']
        if os.path.exists(adjacency_maps_path):
            self.analysis_button.setEnabled(True)
        else:
            self.analysis_button.setEnabled(False)

        im_branch_label_path = self.im_info.pipeline_paths['im_branch_label_reassigned']
        if os.path.exists(im_branch_label_path):
            self.open_reassign_button.setEnabled(True)
        else:
            self.open_reassign_button.setEnabled(False)

    def open_preprocess_image(self):
        im_frangi = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi = get_reshaped_image(im_frangi, self.num_t, self.im_info)

        self.check_3d()

        self.frangi_layer = self.viewer.add_image(self.im_frangi, name='pre processed', colormap='turbo', scale=self.scale)
        self.frangi_layer.interpolation = 'nearest'

    def open_segment_image(self):
        im_instance_label = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.im_instance_label = get_reshaped_image(im_instance_label, self.num_t, self.im_info)
        im_skel_relabelled = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.im_skel_relabelled = get_reshaped_image(im_skel_relabelled, self.num_t, self.im_info)

        self.check_3d()

        self.im_instance_label_layer = self.viewer.add_labels(self.im_instance_label, name='segmentation: objects',
                                                              opacity=1, scale=self.scale)
        self.im_skel_relabelled_layer = self.viewer.add_labels(self.im_skel_relabelled, name='segmentation: branches',
                                                               opacity=1, scale=self.scale)
        self.im_instance_label_layer.mouse_drag_callbacks.append(self.on_label_click)
        self.im_skel_relabelled_layer.mouse_drag_callbacks.append(self.on_label_click)

    def on_label_click(self, layer, event):
        # if flow_vector_array path does not point to an existing file, return
        if not os.path.exists(self.im_info.pipeline_paths['flow_vector_array']):
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
            label_path = self.im_info.pipeline_paths['im_instance_label']
        elif layer == self.im_skel_relabelled_layer:
            label_path = self.im_info.pipeline_paths['im_skel_relabelled']
        elif layer == self.im_branch_label_reassigned_layer:
            label_path = self.im_info.pipeline_paths['im_branch_label_reassigned']
        elif layer == self.im_obj_label_reassigned_layer:
            label_path = self.im_info.pipeline_paths['im_obj_label_reassigned']
        else:
            label_path = None

        label_tracks = LabelTracks(im_info=self.im_info, num_t=self.num_t, label_im_path=label_path)
        label_tracks.initialize()
        all_tracks = []
        all_props = {}
        max_track_num = 0
        if layer == self.im_branch_label_reassigned_layer or layer == self.im_obj_label_reassigned_layer:
            for frame in range(self.num_t):
                tracks, track_properties = label_tracks.run(label_num=label, start_frame=frame, end_frame=None,
                                                            min_track_num=max_track_num)
                all_tracks += tracks
                for property in track_properties.keys():
                    if property not in all_props.keys():
                        all_props[property] = []
                    all_props[property] += track_properties[property]
                max_track_num = max([track[0] for track in tracks])+1
        else:
            all_tracks, all_props = label_tracks.run(label_num=label, start_frame=pos[0], end_frame=None)
        self.viewer.add_tracks(all_tracks, properties=all_props, name=f'tracks: {label}', scale=self.scale)
        self.viewer.layers.selection.active = layer
        self.check_file_existence()

    def run_preprocessing(self):
        preprocessing = Filter(im_info=self.im_info, num_t=self.num_t, remove_edges=self.remove_edges)
        preprocessing.run()

        self.check_file_existence()

    def run_segmentation(self):
        segmenting = Label(im_info=self.im_info, num_t=self.num_t)
        segmenting.run()
        networking = Network(im_info=self.im_info, num_t=self.num_t)
        networking.run()

        self.check_file_existence()

    def open_mocap_image(self):
        self.check_3d()

        im_marker = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_marker'])
        self.im_marker = get_reshaped_image(im_marker, self.num_t, self.im_info)
        self.im_marker_layer = self.viewer.add_image(self.im_marker, name='mocap markers', colormap='red',
                                                     blending='additive', contrast_limits=[0, 1], scale=self.scale)
        self.im_marker_layer.interpolation = 'nearest'

    def run_mocap(self):
        mocap_marking = Markers(im_info=self.im_info, num_t=self.num_t)
        mocap_marking.run()

        self.check_file_existence()

    def run_tracking(self):
        hu_tracking = HuMomentTracking(im_info=self.im_info, num_t=self.num_t)
        hu_tracking.run()

        self.check_file_existence()

    def open_reassign_image(self):
        self.check_3d()

        im_branch_label_reassigned = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_branch_label_reassigned'])
        self.im_branch_label_reassigned = get_reshaped_image(im_branch_label_reassigned, self.num_t, self.im_info)
        self.im_branch_label_reassigned_layer = self.viewer.add_labels(self.im_branch_label_reassigned, name='reassigned branch voxels', scale=self.scale)
        self.im_branch_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

        im_obj_label_reassigned = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_obj_label_reassigned'])
        self.im_obj_label_reassigned = get_reshaped_image(im_obj_label_reassigned, self.num_t, self.im_info)
        self.im_obj_label_reassigned_layer = self.viewer.add_labels(self.im_obj_label_reassigned, name='reassigned object voxels', scale=self.scale)
        self.im_obj_label_reassigned_layer.mouse_drag_callbacks.append(self.on_label_click)

    def run_reassign(self):
        vox_reassign = VoxelReassigner(im_info=self.im_info, num_t=self.num_t)
        vox_reassign.run()

        self.check_file_existence()

    def run_feature_export(self):
        hierarchy = Hierarchy(im_info=self.im_info, num_t=self.num_t)
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
        try:
            # todo implement lazy loading if wanted
            im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
            self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)
        except Exception as e:
            logger.error(e)
            self.im_memmap = None
        if self.im_memmap is not None:
            self.run_button.setEnabled(True)
            self.raw_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)
            self.segment_button.setEnabled(True)
            self.mocap_button.setEnabled(True)
            self.track_button.setEnabled(True)
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)

    def open_raw(self):
        self.check_3d()
        self.raw_layer = self.viewer.add_image(self.im_memmap, name='raw', colormap='gray',
                                               blending='additive', scale=self.scale)
        # make 3d interpolation to nearest
        self.raw_layer.interpolation = 'nearest'
        # self.raw_layer.mouse_drag_callbacks.append(self.on_click)

    def find_closest_voxel(self, near2, far2):
        near1, far1 = self.ray_near_far
        time_point = near1[0]
        near1 = near1[1:]
        far1 = far1[1:]
        near2 = near2[1:]
        far2 = far2[1:]

        p1 = np.array(near1)
        d1 = np.array(far1) - p1

        p2 = np.array(near2)
        d2 = np.array(far2) - p2

        show_info(f"{p1}, {d1}, {p2}, {d2}")

        # normalize
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)
        show_info(f"{p1}, {d1}, {p2}, {d2}")

        # cross
        cross_d1_d2 = np.cross(d1, d2)

        # check non-parallel
        if np.linalg.norm(cross_d1_d2) < 1e-6:
            logger.error("Rays are parallel or coincident.")
            return

        # Calculate the line segment connecting the two lines that is perpendicular to both
        # This is done by solving the system of linear equations formed by the two lines
        a = np.array([d1, -d2, cross_d1_d2]).T
        b = p2 - p1
        t, s, _ = np.linalg.solve(a, b)

        # Calculate the closest points on the original lines
        closest_point_on_line1 = p1 + d1 * t
        closest_point_on_line2 = p2 + d2 * s

        # Find the midpoint of the shortest line connecting the two rays
        midpoint = (closest_point_on_line1 + closest_point_on_line2) / 2
        # add time point to midpoint coords
        midpoint = np.insert(midpoint, 0, time_point)
        # Convert the midpoint to voxel coordinates (assumes the raw layer is the image)
        voxel_coords = np.round(self.raw_layer.world_to_data(midpoint)).astype(int)
        show_info(f"Midpoint: {midpoint}, Voxel coords: {voxel_coords}")
        # self.point_layer = self.viewer.add_points(voxel_coords, name='midpoint', face_color='red', size=2)
        self.tracked_coords = voxel_coords

        # Check if the voxel_coords are within the bounds of the image
        if np.all(voxel_coords >= 0) and np.all(voxel_coords < self.im_memmap.shape):
            # Here you can do something with the voxel coordinates, like highlighting the voxel
            logger.info(f"Closest voxel at: {voxel_coords}")
        else:
            logger.error("Calculated voxel coordinates are out of bounds.")

    def animate_camera_angle(self):
        # Calculate the time elapsed since the start of the animation
        time_elapsed = time.time() - self.animation_start_time
        duration = 0.5  # Total duration of the animation in seconds

        if time_elapsed < duration:
            # Calculate the intermediate angle based on the elapsed time
            fraction = time_elapsed / duration
            delta_angle = 45 * fraction  # Change in angle should be 45 degrees
            new_angle = self.start_angle + delta_angle

            # Update the camera angle
            self.viewer.camera.angles = (self.current_angles[0], self.current_angles[1], new_angle)
        else:
            # Animation is complete, set the final angle
            self.viewer.camera.angles = (self.current_angles[0], self.current_angles[1], self.start_angle + 45)
            self.animation_timer.stop()  # Stop the timer

    def get_tracks(self):
        if self.tracked_coords is None:
            return
        t_current = self.tracked_coords[0]
        coord_to_track = np.asarray([self.tracked_coords[1:]])
        tracks_forward, track_props_forward = interpolate_all_forward(coord_to_track, t_current,
                                                                      self.num_t - 1, self.im_info)
        tracks_back, track_props_back = interpolate_all_backward(coord_to_track, t_current,
                                                                1, self.im_info)
        self.track_layer = self.viewer.add_tracks(tracks_forward, properties=track_props_forward, name='tracks', scale=self.scale)
        self.track_layer = self.viewer.add_tracks(tracks_back, properties=track_props_back, name='tracks', scale=self.scale)

    def on_click(self, layer, event):  # https://napari.org/dev/guides/3D_interactivity.html
        near, far = None, None
        if event.button == 1 and event.is_dragging is False and 'Alt' in event.modifiers:
            near, far = layer.get_ray_intersections(event.position, event.view_direction, event.dims_displayed)
        if near is None or far is None:
            return

        # add line to viewer
        if self.ray_layer is None:
            self.ray_near_far = (near, far)
            self.ray_layer = self.viewer.add_shapes(
                np.asarray([[*near], [*far]]),
                shape_type='line',
                edge_color='red',
                edge_width=1,
                name='ray',
                opacity=0.1, scale=self.scale
            )
            self.viewer.layers.selection.active = self.viewer.layers[0]

            # Store the current angles and start angle for the animation
            self.current_angles = self.viewer.camera.angles
            self.start_angle = self.current_angles[2]

            # Start the animation
            self.animation_start_time = time.time()
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.animate_camera_angle)
            self.animation_timer.start(1000 // 60)  # Update at 60 FPS
        else:
            self.find_closest_voxel(near, far)
            # remove ray layer
            self.viewer.layers.remove(self.ray_layer)
            self.ray_layer = None
            self.ray_near_far = None
            self.get_tracks()

    def screenshot(self, event=None):
        # # if there's no layer, return
        # if self.viewer.layers is None:
        #     return

        # easy no prompt screenshot
        dt = datetime.datetime.now() #year, month, day, hour, minute, second, millisecond up to 3 digits
        dt = dt.strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.im_info.screenshot_dir
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        im_name = f'{dt}-{self.im_info.basename_no_ext}.png'
        file_path = os.path.join(screenshot_folder, im_name)

        # Take screenshot
        screenshot = self.viewer.screenshot(canvas_only=True)

        # Save the screenshot
        try:
            # save as png to file_path using tifffile
            tifffile.imwrite(file_path, screenshot)
            print(f"Screenshot saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to save screenshot: {str(e)}")
            raise e


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
