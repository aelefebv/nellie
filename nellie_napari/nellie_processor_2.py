import os

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QGroupBox
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt

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

        main_layout.addWidget(full_pipeline_group)
        main_layout.addWidget(partial_pipeline_group)

        self.setLayout(main_layout)

    def post_init(self):
        self.check_file_existence()
        self.initialized = True
        
    def check_file_existence(self):
        self.nellie.visualizer.check_file_existence()
        self.run_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)

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
            self.feature_export_button.setEnabled(True)
            # if im_info's 'T' axis has more than 1 timepoint, disable the feature export button
            if self.nellie.im_info.shape[0] > 1:
                self.feature_export_button.setEnabled(False)
                return

        analysis_path = self.nellie.im_info.pipeline_paths['adjacency_maps']
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)

    @thread_worker
    def _run_preprocessing(self):
        show_info("Nellie is running: Preprocessing")
        preprocessing = Filter(im_info=self.nellie.im_info,
                               remove_edges=self.nellie.settings.remove_edges_checkbox.isChecked(),
                               viewer=self.viewer)
        preprocessing.run()
        return None

    def run_preprocessing(self):
        worker = self._run_preprocessing()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_segmentation)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_segmentation(self):
        show_info("Nellie is running: Segmentation")
        segmenting = Label(im_info=self.nellie.im_info, viewer=self.viewer)
        segmenting.run()
        networking = Network(im_info=self.nellie.im_info, viewer=self.viewer)
        networking.run()

    def run_segmentation(self):
        worker = self._run_segmentation()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_mocap)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_mocap(self):
        show_info("Nellie is running: Mocap Marking")
        mocap_marking = Markers(im_info=self.nellie.im_info, viewer=self.viewer)
        mocap_marking.run()

    def run_mocap(self):
        worker = self._run_mocap()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_tracking)
        worker.finished.connect(self.check_file_existence)
        worker.start()


    @thread_worker
    def _run_tracking(self):
        show_info("Nellie is running: Tracking")
        hu_tracking = HuMomentTracking(im_info=self.nellie.im_info, viewer=self.viewer)
        hu_tracking.run()

    def run_tracking(self):
        worker = self._run_tracking()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_reassign)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_reassign(self):
        show_info("Nellie is running: Voxel Reassignment")
        vox_reassign = VoxelReassigner(im_info=self.nellie.im_info, viewer=self.viewer)
        vox_reassign.run()

    def run_reassign(self):
        worker = self._run_reassign()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_feature_export)
        worker.finished.connect(self.check_file_existence)
        worker.start()


    @thread_worker
    def _run_feature_export(self):
        show_info("Nellie is running: Feature export")
        hierarchy = Hierarchy(im_info=self.nellie.im_info,
                              skip_nodes=not bool(self.nellie.settings.analyze_node_level.isChecked()),
                              viewer=self.viewer)
        hierarchy.run()

    def run_feature_export(self):
        worker = self._run_feature_export()
        worker.started.connect(self.turn_off_buttons)
        worker.finished.connect(self.check_file_existence)
        worker.finished.connect(self.turn_off_pipeline)
        worker.start()

    def turn_off_pipeline(self):
        self.pipeline = False

    def run_nellie(self):
        self.pipeline = True
        self.run_preprocessing()

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
