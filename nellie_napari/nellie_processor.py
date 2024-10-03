import os
import subprocess
import time

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QGroupBox, QLabel, QHBoxLayout
from qtpy.QtGui import QFont, QIcon
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
    """
    The NellieProcessor class manages the different steps of the Nellie pipeline such as preprocessing, segmentation,
    mocap marking, tracking, voxel reassignment, and feature extraction. It provides an interface to run each step
    individually or as part of a full pipeline within a napari viewer.

    Attributes
    ----------
    nellie : object
        Reference to the Nellie instance managing the pipeline.
    viewer : napari.viewer.Viewer
        Reference to the napari viewer instance.
    im_info_list : list of ImInfo or None
        List of ImInfo objects for the selected files. Contains metadata and file information.
    current_im_info : ImInfo or None
        The current image's information and metadata object.
    status_label : QLabel
        Label displaying the current status of the process.
    status : str or None
        The current status of the process (e.g., "preprocessing", "segmentation").
    num_ellipses : int
        Counter to manage the ellipsis effect on the status label during execution.
    status_timer : QTimer
        Timer that periodically updates the status label during pipeline execution.
    open_dir_button : QPushButton
        Button to open the output directory of the current image file.
    run_button : QPushButton
        Button to run the full Nellie pipeline.
    preprocess_button : QPushButton
        Button to run only the preprocessing step of the pipeline.
    segment_button : QPushButton
        Button to run only the segmentation step of the pipeline.
    mocap_button : QPushButton
        Button to run only the mocap marking step of the pipeline.
    track_button : QPushButton
        Button to run only the tracking step of the pipeline.
    reassign_button : QPushButton
        Button to run only the voxel reassignment step of the pipeline.
    feature_export_button : QPushButton
        Button to run only the feature extraction step of the pipeline.
    initialized : bool
        Flag indicating whether the processor has been initialized.
    pipeline : bool
        Flag indicating whether the full pipeline is being run.

    Methods
    -------
    set_ui()
        Initializes and sets the layout and UI components for the NellieProcessor.
    post_init()
        Post-initialization method to load image information and check file existence.
    check_file_existence()
        Checks the existence of necessary files for each step of the pipeline and enables/disables buttons accordingly.
    run_nellie()
        Runs the entire Nellie pipeline starting from preprocessing to feature extraction.
    run_preprocessing()
        Runs the preprocessing step of the Nellie pipeline.
    run_segmentation()
        Runs the segmentation step of the Nellie pipeline.
    run_mocap()
        Runs the mocap marking step of the Nellie pipeline.
    run_tracking()
        Runs the tracking step of the Nellie pipeline.
    run_reassign()
        Runs the voxel reassignment step of the Nellie pipeline.
    run_feature_export()
        Runs the feature extraction step of the Nellie pipeline.
    set_status()
        Sets the status to indicate that a process has started, and starts the status update timer.
    update_status()
        Updates the status label with the current process and an ellipsis effect while the process is running.
    reset_status()
        Resets the status label to indicate that no process is running.
    turn_off_buttons()
        Disables all buttons to prevent multiple processes from running simultaneously.
    open_directory()
        Opens the output directory where the current image results are saved.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the NellieProcessor class, setting up the user interface and preparing for running various steps of the pipeline.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the Nellie instance that manages the pipeline.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
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

        self.open_dir_button = QPushButton(text="Open output directory")
        self.open_dir_button.clicked.connect(self.open_directory)
        self.open_dir_button.setEnabled(False)

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
        """
        Initializes and sets the layout and user interface components for the NellieProcessor. This includes the status label,
        buttons for running individual steps, and the button for running the entire pipeline.
        """
        main_layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout()  # Changed to QHBoxLayout
        status_layout.addWidget(self.status_label, alignment=Qt.AlignLeft)
        status_layout.addWidget(self.open_dir_button, alignment=Qt.AlignCenter)
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
        """
        Post-initialization method that checks the state of the selected images. It determines whether the pipeline
        steps have already been completed and enables/disables the corresponding buttons accordingly.
        """
        if self.nellie.im_info_list is None:
            self.im_info_list = [self.nellie.im_info]
            self.current_im_info = self.nellie.im_info
        else:
            self.im_info_list = self.nellie.im_info_list
            self.current_im_info = self.nellie.im_info_list[0]
        self.check_file_existence()
        self.initialized = True
        self.open_dir_button.setEnabled(os.path.exists(self.current_im_info.file_info.output_dir))

    def check_file_existence(self):
        """
        Checks the existence of files required for each step of the pipeline (e.g., preprocessed images, segmented labels).
        Enables or disables buttons based on the existence of these files.
        """
        self.nellie.visualizer.check_file_existence()

        # set all other buttons to disabled first
        self.run_button.setEnabled(False)
        self.preprocess_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)

        analysis_path = self.current_im_info.pipeline_paths['features_organelles']
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)

        self.open_dir_button.setEnabled(os.path.exists(self.current_im_info.file_info.output_dir))

        if os.path.exists(self.current_im_info.im_path):
            self.run_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)
        else:
            return

        frangi_path = self.current_im_info.pipeline_paths['im_preprocessed']
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
        """
        Runs the preprocessing step in a separate thread. Filters the image to remove noise or unwanted edges before segmentation.
        """
        self.status = "preprocessing"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Preprocessing file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            preprocessing = Filter(im_info=self.current_im_info,
                                   remove_edges=self.nellie.settings.remove_edges_checkbox.isChecked(),
                                   viewer=self.viewer)
            preprocessing.run()

    def run_preprocessing(self):
        """
        Starts the preprocessing step and updates the UI to reflect that preprocessing is running.
        If the full pipeline is running, it automatically proceeds to segmentation after preprocessing is finished.
        """
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
        """
        Runs the segmentation step in a separate thread. Labels and segments regions of interest in the preprocessed image.
        """
        self.status = "segmentation"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Segmentation file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            segmenting = Label(im_info=self.current_im_info, viewer=self.viewer)
            segmenting.run()
            networking = Network(im_info=self.current_im_info, viewer=self.viewer)
            networking.run()

    def run_segmentation(self):
        """
        Starts the segmentation step and updates the UI to reflect that segmentation is running.
        If the full pipeline is running, it automatically proceeds to mocap marking after segmentation is finished.
        """
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
        """
        Runs the mocap marking step in a separate thread. Marks the motion capture points within the segmented regions.
        """
        self.status = "mocap marking"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Mocap Marking file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            mocap_marking = Markers(im_info=self.current_im_info, viewer=self.viewer)
            mocap_marking.run()

    def run_mocap(self):
        """
        Starts the mocap marking step and updates the UI to reflect that mocap marking is running.
        If the full pipeline is running, it automatically proceeds to tracking after mocap marking is finished.
        """
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
        """
        Runs the tracking step in a separate thread. Tracks the motion of the marked points over time.
        """
        self.status = "tracking"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Tracking file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            hu_tracking = HuMomentTracking(im_info=self.current_im_info, viewer=self.viewer)
            hu_tracking.run()

    def run_tracking(self):
        """
        Starts the tracking step and updates the UI to reflect that tracking is running.
        If the full pipeline is running, it automatically proceeds to voxel reassignment or feature extraction depending on the settings.
        """
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
        """
        Runs the voxel reassignment step in a separate thread. Reassigns voxel labels based on the tracked motion.
        """
        self.status = "voxel reassignment"
        for im_num, im_info in enumerate(self.im_info_list):
            show_info(f"Nellie is running: Voxel Reassignment file {im_num + 1}/{len(self.im_info_list)}")
            self.current_im_info = im_info
            vox_reassign = VoxelReassigner(im_info=self.current_im_info, viewer=self.viewer)
            vox_reassign.run()

    def run_reassign(self):
        """
        Starts the voxel reassignment step and updates the UI to reflect that voxel reassignment is running.
        If the full pipeline is running, it automatically proceeds to feature extraction after voxel reassignment is finished.
        """
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
        """
        Runs the feature extraction step in a separate thread. Extracts various features from the processed image data for analysis.
        """
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
        """
        Starts the feature extraction step and updates the UI to reflect that feature extraction is running.
        """
        worker = self._run_feature_export()
        worker.started.connect(self.turn_off_buttons)
        worker.finished.connect(self.check_file_existence)
        worker.finished.connect(self.turn_off_pipeline)
        worker.started.connect(self.set_status)
        worker.finished.connect(self.reset_status)
        worker.start()

    def turn_off_pipeline(self):
        """
        Turns off the pipeline flag to indicate that the full pipeline is no longer running.
        """
        self.pipeline = False

    def run_nellie(self):
        """
        Starts the entire Nellie pipeline from preprocessing to feature extraction.
        """
        self.pipeline = True
        self.run_preprocessing()

    def set_status(self):
        """
        Sets the status of the processor to indicate that a process is running, and starts the status update timer.
        """
        self.running = True
        self.status_timer.start(500)  # Update every 250 ms

    def update_status(self):
        """
        Updates the status label with an ellipsis effect to indicate ongoing processing.
        """
        if self.running:
            self.status_label.setText(f"Running {self.status}{'.' * (self.num_ellipses % 4)}")
            self.num_ellipses += 1
        else:
            self.status_timer.stop()

    def reset_status(self):
        """
        Resets the status label to indicate that no process is running.
        """
        self.running = False
        self.status_label.setText("Awaiting your input")
        self.num_ellipses = 1
        self.status_timer.stop()

    def turn_off_buttons(self):
        """
        Disables all buttons to prevent multiple processes from running simultaneously.
        """
        self.run_button.setEnabled(False)
        self.preprocess_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.mocap_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.reassign_button.setEnabled(False)
        self.feature_export_button.setEnabled(False)

    def open_directory(self):
        """
        Opens the output directory of the current image in the system file explorer.
        """
        directory = self.current_im_info.file_info.output_dir
        if os.path.exists(directory):
            if os.name == 'nt':  # For Windows
                os.startfile(directory)
            elif os.name == 'posix':  # For macOS and Linux
                subprocess.call(['open', directory])
        else:
            show_info("Output directory does not exist.")


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
