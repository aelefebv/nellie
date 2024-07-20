import os

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QSpinBox
from nellie import logger
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from nellie.utils.general import get_reshaped_image
from napari.qt.threading import thread_worker


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

        self.initialized = False

    def post_init(self):
        if not self.check_for_raw():
            return
        self.check_file_existence()
        self.initialized = True
        
    def check_file_existence(self):
        self.pipeline = False
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
            self.feature_export_button.setEnabled(True)
            if self.num_t > 1:
                self.feature_export_button.setEnabled(False)
                return

        analysis_path = self.nellie.im_info.pipeline_paths['adjacency_maps']
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)

    @thread_worker
    def _run_preprocessing(self):
        show_info("Nellie in running: Preprocessing")
        preprocessing = Filter(im_info=self.nellie.im_info, num_t=self.num_t,
                               remove_edges=self.nellie.settings.remove_edges_checkbox.isChecked(),
                               viewer=self.viewer)
        preprocessing.run()
        return None

    def run_preprocessing(self):
        worker = self._run_preprocessing()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_segmentation)
        # else:
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_segmentation(self):
        show_info("Nellie in running: Segmentation")
        segmenting = Label(im_info=self.nellie.im_info, num_t=self.num_t, viewer=self.viewer)
        segmenting.run()
        networking = Network(im_info=self.nellie.im_info, num_t=self.num_t, viewer=self.viewer)
        networking.run()

    def run_segmentation(self):
        worker = self._run_segmentation()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_mocap)
        # else:
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()


    @thread_worker
    def _run_mocap(self):
        show_info("Nellie in running: Mocap Marking")
        mocap_marking = Markers(im_info=self.nellie.im_info, num_t=self.num_t, viewer=self.viewer)
        mocap_marking.run()

    def run_mocap(self):
        worker = self._run_mocap()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_tracking)
        # else:
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()


    @thread_worker
    def _run_tracking(self):
        show_info("Nellie in running: Tracking")
        hu_tracking = HuMomentTracking(im_info=self.nellie.im_info, num_t=self.num_t, viewer=self.viewer)
        hu_tracking.run()

    def run_tracking(self):
        worker = self._run_tracking()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_reassign)
        # else:
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    @thread_worker
    def _run_reassign(self):
        show_info("Nellie in running: Voxel Reassignment")
        vox_reassign = VoxelReassigner(im_info=self.nellie.im_info, num_t=self.num_t, viewer=self.viewer)
        vox_reassign.run()

    def run_reassign(self):
        worker = self._run_reassign()
        worker.started.connect(self.turn_off_buttons)
        if self.pipeline:
            worker.finished.connect(self.run_feature_export)
        # else:
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()


    @thread_worker
    def _run_feature_export(self):
        show_info("Nellie in running: Feature export")
        hierarchy = Hierarchy(im_info=self.nellie.im_info, num_t=self.num_t,
                              skip_nodes=not bool(self.nellie.settings.analyze_node_level.isChecked()),
                              viewer=self.viewer)
        hierarchy.run()

    def run_feature_export(self):
        worker = self._run_feature_export()
        worker.started.connect(self.turn_off_buttons)
        worker.finished.connect(self.check_for_raw)
        worker.finished.connect(self.check_file_existence)
        worker.start()

    # @thread_worker
    # def _run_nellie(self):
    #     self.run_preprocessing(pipeline=True)
    #     self.run_segmentation()
    #     self.run_mocap()
    #     self.run_tracking()
    #     if self.nellie.settings.voxel_reassign.isChecked():
    #         self.run_reassign()
    #     self.run_feature_export()
    #
    #     # self.check_file_existence()

    def run_nellie(self):
        self.pipeline = True
        self.run_preprocessing()
        # worker = self.run_preprocessing()
        # worker.started.connect(self.turn_off_buttons)
        # worker.finished.connect(self.check_for_raw)
        # worker.finished.connect(self.check_file_existence)
        # worker.start()

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
