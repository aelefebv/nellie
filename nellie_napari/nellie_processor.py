import os
import time

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QGroupBox, QLabel
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt, QTimer

from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from napari.qt.threading import thread_worker


class NellieProcessor(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.im_info_list = None
        self.current_im_info = None

        self.status_label = QLabel("Awaiting your input")
        self.status = None
        self.num_ellipses = 1

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)

        # Run im button
        self.run_button = QPushButton(text="Run Nellie")
        self.run_button.clicked.connect(self.run_nellie)
        self.run_button.setEnabled(True)
        self.run_button.setFixedWidth(200)
        self.run_button.setFixedHeight(100)
        self.run_button.setStyleSheet("border-radius: 10px;")
        self.run_button.setFont(QFont("Arial", 20))

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

        self.set_ui()

        self.initialized = False
        self.pipeline = False

    def set_ui(self):
        main_layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        status_group.setMaximumHeight(100)
        status_group.setLayout(status_layout)

        # Run full pipeline
        full_pipeline_group = QGroupBox("Run full pipeline")
        full_pipeline_layout = QVBoxLayout()
        full_pipeline_layout.addWidget(self.run_button, alignment=Qt.AlignCenter)
        full_pipeline_group.setLayout(full_pipeline_layout)

        # Run partial pipeline
        partial_pipeline_group = QGroupBox("Run individual steps")
        partial_pipeline_layout = QVBoxLayout()
        partial_pipeline_layout.addWidget(self.preprocess_button)
        partial_pipeline_layout.addWidget(self.segment_button)
        partial_pipeline_layout.addWidget(self.mocap_button)
        partial_pipeline_layout.addWidget(self.track_button)
        partial_pipeline_layout.addWidget(self.reassign_button)
        partial_pipeline_layout.addWidget(self.feature_export_button)
        partial_pipeline_group.setLayout(partial_pipeline_layout)

        main_layout.addWidget(status_group)
        main_layout.addWidget(full_pipeline_group)
        main_layout.addWidget(partial_pipeline_group)

        self.setLayout(main_layout)

    def post_init(self):
        if self.nellie.im_info_list is None:
            self.im_info_list = [self.nellie.im_info]
            self.current_im_info = self.nellie.im_info
        else:
            self.im_info_list = self.nellie.im_info_list
            self.current_im_info = self.nellie.im_info_list[0]
        self.check_file_existence()
        self.initialized = True
        
    def check_file_existence(self):
        self.nellie.visualizer.check_file_existence()

        # set all other buttons to disabled first
        self.run_button.setEnabled(False)
        self.preprocess_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)

        analysis_path = self.current_im_info.pipeline_paths['features_components']
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)

        if os.path.exists(self.current_im_info.im_path):
            self.run_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)
        else:
            return

        frangi_path = self.current_im_info.pipeline_paths['im_frangi']
        if os.path.exists(frangi_path):
            self.segment_button.setEnabled(True)
        else:
            self.segment_button.setEnabled(False)
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_instance_label_path = self.current_im_info.pipeline_paths['im_instance_label']
        im_skel_relabelled_path = self.current_im_info.pipeline_paths['im_skel_relabelled']
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.mocap_button.setEnabled(True)
        else:
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_marker_path = self.current_im_info.pipeline_paths['im_marker']
        if os.path.exists(im_marker_path):
            self.track_button.setEnabled(True)
        else:
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        track_path = self.current_im_info.pipeline_paths['flow_vector_array']
        if os.path.exists(track_path):
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)
        else:
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(True)
            # if im_info's 'T' axis has more than 1 timepoint, disable the feature export button
            if self.current_im_info.shape[0] > 1:
                self.feature_export_button.setEnabled(False)
                return



    @thread_worker
    def _run_preprocessing(self):
        self.status = "preprocessing"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Preprocessing file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            preprocessing = Filter(im_info=self.current_im_info,
                                   remove_edges=self.nellie.settings.remove_edges_checkbox.isChecked(),
                                   viewer=self.viewer)
            preprocessing.run()

    def run_preprocessing(self):
        worker = self._run_preprocessing()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_segmentation)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_segmentation(self):
        self.status = "segmentation"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Segmentation file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            segmenting = Label(im_info=self.current_im_info, viewer=self.viewer)
            segmenting.run()
            networking = Network(im_info=self.current_im_info, viewer=self.viewer)
            networking.run()

    def run_segmentation(self):
        worker = self._run_segmentation()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_mocap)
        worker.finished.connect(self.check_file_existence)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()

    @thread_worker
    def _run_mocap(self):
        self.status = "mocap marking"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Mocap Marking file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            mocap_marking = Markers(im_info=self.current_im_info, viewer=self.viewer)
            mocap_marking.run()

    def run_mocap(self):
        worker = self._run_mocap()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_tracking)
        worker.finished.connect(self.check_file_existence)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()


    @thread_worker
    def _run_tracking(self):
        self.status = "tracking"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Tracking file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            hu_tracking = HuMomentTracking(im_info=self.current_im_info, viewer=self.viewer)
            hu_tracking.run()

    def run_tracking(self):
        worker = self._run_tracking()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            if self.nellie.settings.voxel_reassign.isChecked():
                worker.finished.connect(self.run_reassign)
            else:
                worker.finished.connect(self.run_feature_export)
        worker.finished.connect(self.check_file_existence)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()

    @thread_worker
    def _run_reassign(self):
        self.status = "voxel reassignment"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Voxel Reassignment file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            vox_reassign = VoxelReassigner(im_info=self.current_im_info, viewer=self.viewer)
            vox_reassign.run()

    def run_reassign(self):
        worker = self._run_reassign()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_feature_export)
        worker.finished.connect(self.check_file_existence)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()


    @thread_worker
    def _run_feature_export(self):
        self.status = "feature export"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Feature export file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            hierarchy = Hierarchy(im_info=self.current_im_info,
                                  skip_nodes=not bool(self.nellie.settings.analyze_node_level.isChecked()),
                                  viewer=self.viewer)
            hierarchy.run()
            if self.nellie.settings.remove_intermediates_checkbox.isChecked():
                try:
                    self.current_im_info.remove_intermediates()
                except Exception as e:
                    show_info(f"Error removing intermediates: {e}")
        if self.nellie.analyzer.initialized:
            self.nellie.analyzer.rewrite_dropdown()

    def run_feature_export(self):
        worker = self._run_feature_export()
        worker.started.connect(self.turn_off_buttons)
        worker.finished.connect(self.check_file_existence)
        worker.finished.connect(self.turn_off_pipeline)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()

    def turn_off_pipeline(self):
        self.pipeline = False

    def run_nellie(self):
        self.pipeline = True
        self.run_preprocessing()

    def set_status(self):
        self.running = True
        self.status_timer.start(500)  # Update every 250 ms

    def update_status(self):
        if self.running:
            self.status_label.setText(f"Running {self.status}{'.' * (self.num_ellipses % 4)}")
            self.num_ellipses += 1
        else:
            self.status_timer.stop()

    def reset_status(self):
        self.running = False
        self.status_label.setText("Awaiting your input")
        self.num_ellipses = 1
        self.status_timer.stop()

    def turn_off_buttons(self):
        self.run_button.setEnabled(False)
        self.preprocess_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
