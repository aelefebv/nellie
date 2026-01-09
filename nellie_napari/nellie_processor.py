import os
import subprocess
import sys

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QGroupBox, QLabel, QHBoxLayout
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
        Runs the mocap marking step of the pipeline.
    run_tracking()
        Runs the tracking step of the pipeline.
    run_reassign()
        Runs the voxel reassignment step of the pipeline.
    run_feature_export()
        Runs the feature extraction step of the pipeline.
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

    def __init__(self, napari_viewer: "napari.viewer.Viewer", nellie, parent=None):
        """
        Initialize the NellieProcessor class, setting up the user interface and preparing for running various steps of the pipeline.

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
        self.status_label.setToolTip("Processing status for the current dataset.")
        self.output_dir_label = QLabel("Output directory: none")
        self.output_dir_label.setWordWrap(True)
        self.status = None
        self.num_ellipses = 1

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)

        self.open_dir_button = QPushButton(text="Open output directory")
        self.open_dir_button.setToolTip("Open the output directory for the current dataset.")
        self.open_dir_button.clicked.connect(self.open_directory)
        self.open_dir_button.setEnabled(False)

        # Run im button
        self.run_button = QPushButton(text="Run Nellie")
        self.run_button.setToolTip("Run the full pipeline in order.")
        self.run_button.clicked.connect(self.run_nellie)
        self.run_button.setEnabled(True)
        self.run_button.setFixedWidth(200)
        self.run_button.setFixedHeight(100)
        self.run_button.setStyleSheet("border-radius: 10px;")
        run_font = QFont()
        run_font.setPointSize(18)
        run_font.setBold(True)
        self.run_button.setFont(run_font)

        # Preprocess im button
        self.preprocess_button = QPushButton(text="Run preprocessing")
        self.preprocess_button.setToolTip("Enhance and denoise the input image.")
        self.preprocess_button.clicked.connect(self.run_preprocessing)
        self.preprocess_button.setEnabled(False)

        # Segment im button
        self.segment_button = QPushButton(text="Run segmentation")
        self.segment_button.setToolTip("Create organelle and branch labels.")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setEnabled(False)

        # Run mocap button
        self.mocap_button = QPushButton(text="Run mocap marking")
        self.mocap_button.setToolTip("Detect mocap markers used for tracking.")
        self.mocap_button.clicked.connect(self.run_mocap)
        self.mocap_button.setEnabled(False)

        # Run tracking button
        self.track_button = QPushButton(text="Run tracking")
        self.track_button.setToolTip("Track labels over time using flow vectors.")
        self.track_button.clicked.connect(self.run_tracking)
        self.track_button.setEnabled(False)

        # Run reassign button
        self.reassign_button = QPushButton(text="Run voxel reassignment")
        self.reassign_button.setToolTip("Reassign voxels to improve track consistency.")
        self.reassign_button.clicked.connect(self.run_reassign)
        self.reassign_button.setEnabled(False)

        # Run feature extraction button
        self.feature_export_button = QPushButton(text="Run feature export")
        self.feature_export_button.setToolTip("Compute features and export CSV files.")
        self.feature_export_button.clicked.connect(self.run_feature_export)
        self.feature_export_button.setEnabled(False)

        self.pipeline_hint_label = QLabel(
            "Step order: Preprocess -> Segment -> Mocap markers -> Track -> "
            "Reassign voxels -> Feature export."
        )
        self.pipeline_hint_label.setWordWrap(True)

        self.set_ui()

        self.initialized = False
        self.pipeline = False

        # status / error tracking
        self.running = False
        self._last_error_message = ""
        self._last_step_had_error = False

    def set_ui(self):
        """
        Initialize and set the layout and user interface components for the NellieProcessor. This includes the status label,
        buttons for running individual steps, and the button for running the entire pipeline.
        """
        main_layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout()
        status_text_layout = QVBoxLayout()
        status_text_layout.addWidget(self.status_label)
        status_text_layout.addWidget(self.output_dir_label)
        status_layout.addLayout(status_text_layout)
        status_layout.addWidget(self.open_dir_button, alignment=Qt.AlignRight)
        status_group.setLayout(status_layout)

        # Run full pipeline
        full_pipeline_group = QGroupBox("Run full pipeline")
        full_pipeline_layout = QVBoxLayout()
        full_pipeline_layout.addWidget(self.run_button, alignment=Qt.AlignCenter)
        full_pipeline_group.setLayout(full_pipeline_layout)

        # Run partial pipeline
        partial_pipeline_group = QGroupBox("Run individual steps")
        partial_pipeline_layout = QVBoxLayout()
        partial_pipeline_layout.addWidget(self.pipeline_hint_label)
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
        self._update_output_dir_label()

    def check_file_existence(self):
        """
        Check the existence of files required for each step of the pipeline (e.g., preprocessed images, segmented labels).
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

        analysis_path = self.current_im_info.pipeline_paths["features_organelles"]
        if os.path.exists(analysis_path):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
        else:
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)

        self.open_dir_button.setEnabled(os.path.exists(self.current_im_info.file_info.output_dir))
        self._update_output_dir_label()

        if os.path.exists(self.current_im_info.im_path):
            self.run_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)
        else:
            return

    def _update_output_dir_label(self):
        """
        Update the output directory label and tooltip for the open button.
        """
        output_dir = None
        if self.current_im_info is not None and self.current_im_info.file_info is not None:
            output_dir = self.current_im_info.file_info.output_dir

        if output_dir:
            self.output_dir_label.setText(f"Output directory: {output_dir}")
            self.open_dir_button.setToolTip(output_dir)
        else:
            self.output_dir_label.setText("Output directory: none")
            self.open_dir_button.setToolTip("Output directory is not available yet.")

        frangi_path = self.current_im_info.pipeline_paths["im_preprocessed"]
        if os.path.exists(frangi_path):
            self.segment_button.setEnabled(True)
        else:
            self.segment_button.setEnabled(False)
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_instance_label_path = self.current_im_info.pipeline_paths["im_instance_label"]
        im_skel_relabelled_path = self.current_im_info.pipeline_paths["im_skel_relabelled"]
        if os.path.exists(im_instance_label_path) and os.path.exists(im_skel_relabelled_path):
            self.mocap_button.setEnabled(True)
        else:
            self.mocap_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        im_marker_path = self.current_im_info.pipeline_paths["im_marker"]
        if os.path.exists(im_marker_path):
            self.track_button.setEnabled(True)
        else:
            self.track_button.setEnabled(False)
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
            return

        track_path = self.current_im_info.pipeline_paths["flow_vector_array"]
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

    # -----------------
    # Worker functions
    # -----------------

    @thread_worker(ignore_errors=True)
    def _run_preprocessing(self, im_info_list, remove_edges):
        """
        Run the preprocessing step in a separate thread. Filters the image to remove noise or unwanted edges before segmentation.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        remove_edges : bool
            Whether to remove edges.
        """
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Preprocessing file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            preprocessing = Filter(
                im_info=self.current_im_info,
                remove_edges=remove_edges,
                viewer=self.viewer,
            )
            preprocessing.run()

    def run_preprocessing(self):
        """
        Start the preprocessing step and updates the UI to reflect that preprocessing is running.
        If the full pipeline is running, it automatically proceeds to segmentation after preprocessing is finished.
        """
        self.status = "preprocessing"
        remove_edges = self.nellie.settings.remove_edges_checkbox.isChecked()
        worker = self._run_preprocessing(self.im_info_list, remove_edges)
        next_step = self.run_segmentation if self.pipeline else None
        self._start_worker(worker, next_step=next_step)

    @thread_worker(ignore_errors=True)
    def _run_segmentation(self, im_info_list):
        """
        Run the segmentation step in a separate thread. Labels and segments regions of interest in the preprocessed image.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        """
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Segmentation file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            segmenting = Label(im_info=self.current_im_info, viewer=self.viewer)
            segmenting.run()
            networking = Network(im_info=self.current_im_info, viewer=self.viewer)
            networking.run()

    def run_segmentation(self):
        """
        Start the segmentation step and updates the UI to reflect that segmentation is running.
        If the full pipeline is running, it automatically proceeds to mocap marking after segmentation is finished.
        """
        self.status = "segmentation"
        worker = self._run_segmentation(self.im_info_list)
        next_step = self.run_mocap if self.pipeline else None
        self._start_worker(worker, next_step=next_step)

    @thread_worker(ignore_errors=True)
    def _run_mocap(self, im_info_list):
        """
        Run the mocap marking step in a separate thread. Marks the motion capture points within the segmented regions.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        """
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Mocap Marking file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            mocap_marking = Markers(im_info=self.current_im_info, viewer=self.viewer)
            mocap_marking.run()

    def run_mocap(self):
        """
        Start the mocap marking step and updates the UI to reflect that mocap marking is running.
        If the full pipeline is running, it automatically proceeds to tracking after mocap marking is finished.
        """
        self.status = "mocap marking"
        worker = self._run_mocap(self.im_info_list)
        next_step = self.run_tracking if self.pipeline else None
        self._start_worker(worker, next_step=next_step)

    @thread_worker(ignore_errors=True)
    def _run_tracking(self, im_info_list):
        """
        Run the tracking step in a separate thread. Tracks the motion of the marked points over time.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        """
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Tracking file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            hu_tracking = HuMomentTracking(im_info=self.current_im_info, viewer=self.viewer)
            hu_tracking.run()

    def run_tracking(self):
        """
        Start the tracking step and updates the UI to reflect that tracking is running.
        If the full pipeline is running, it automatically proceeds to voxel reassignment or feature extraction depending on the settings.
        """
        self.status = "tracking"
        worker = self._run_tracking(self.im_info_list)
        next_step = None
        if self.pipeline:
            if self.nellie.settings.voxel_reassign.isChecked():
                next_step = self.run_reassign
            else:
                next_step = self.run_feature_export
        self._start_worker(worker, next_step=next_step)

    @thread_worker(ignore_errors=True)
    def _run_reassign(self, im_info_list):
        """
        Run the voxel reassignment step in a separate thread. Reassigns voxel labels based on the tracked motion.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        """
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Voxel Reassignment file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            vox_reassign = VoxelReassigner(im_info=self.current_im_info, viewer=self.viewer)
            vox_reassign.run()

    def run_reassign(self):
        """
        Start the voxel reassignment step and updates the UI to reflect that voxel reassignment is running.
        If the full pipeline is running, it automatically proceeds to feature extraction after voxel reassignment is finished.
        """
        self.status = "voxel reassignment"
        worker = self._run_reassign(self.im_info_list)
        next_step = self.run_feature_export if self.pipeline else None
        self._start_worker(worker, next_step=next_step)

    @thread_worker(ignore_errors=True)
    def _run_feature_export(self, im_info_list, analyze_node_level_checked, remove_intermediates_checked):
        """
        Run the feature extraction step in a separate thread. Extracts various features from the processed image data for analysis.

        Parameters
        ----------
        im_info_list : list
            List of ImInfo objects.
        analyze_node_level_checked : bool
            Whether to analyze node level.
        remove_intermediates_checked : bool
            Whether to remove intermediate files.
        """
        skip_nodes = not bool(analyze_node_level_checked)
        for im_num, im_info in enumerate(im_info_list):
            show_info(f"Nellie is running: Feature export file {im_num + 1}/{len(im_info_list)}")
            self.current_im_info = im_info
            hierarchy = Hierarchy(
                im_info=self.current_im_info,
                skip_nodes=skip_nodes,
                viewer=self.viewer,
            )
            hierarchy.run()
            if remove_intermediates_checked:
                try:
                    self.current_im_info.remove_intermediates()
                except Exception as e:
                    show_info(f"Error removing intermediates: {e}")

    def _post_feature_export(self):
        """
        Run any main-thread post-processing needed after feature export
        (e.g., updating analysis dropdowns).
        """
        analyzer = getattr(self.nellie, "analyzer", None)
        if analyzer is not None and getattr(analyzer, "initialized", False):
            try:
                analyzer.rewrite_dropdown()
            except Exception as exc:
                show_info(f"Error updating analysis dropdowns: {exc}")

    def run_feature_export(self):
        """
        Start the feature extraction step and updates the UI to reflect that feature extraction is running.
        """
        self.status = "feature export"
        analyze_node_level_checked = self.nellie.settings.analyze_node_level.isChecked()
        remove_intermediates_checked = self.nellie.settings.remove_intermediates_checkbox.isChecked()
        worker = self._run_feature_export(
            self.im_info_list,
            analyze_node_level_checked,
            remove_intermediates_checked,
        )
        # This is the last step in the pipeline; always treat as final.
        self._start_worker(worker, final=True)

    # --------------
    # Worker helpers
    # --------------

    def _handle_worker_error(self, exc: Exception):
        """
        Handle errors raised in worker threads, updating status message and UI.

        Parameters
        ----------
        exc : Exception
            The exception raised.
        """
        self._last_step_had_error = True
        self.running = False
        self._last_error_message = f"Error during {self.status}: {exc}"
        self.status_label.setText("Error, see console for details")
        self.status_timer.stop()
        show_info(self._last_error_message)
        # ensure buttons reflect current filesystem state
        if self.current_im_info is not None:
            self.check_file_existence()

    def _on_worker_finished(self, next_step=None, final=False):
        """
        Common handler for worker completion, called in the main thread.

        Parameters
        ----------
        next_step : callable, optional
            The next step to run.
        final : bool, optional
            Whether this is the final step.
        """
        self.reset_status()
        if self.current_im_info is not None:
            self.check_file_existence()

        # If there was an error in this step, stop the pipeline and do not continue.
        if self._last_step_had_error:
            self.pipeline = False
            return

        if final:
            self.turn_off_pipeline()
            self._post_feature_export()

        if next_step is not None and self.pipeline:
            next_step()
        
        # Show completion notification for individual steps or when pipeline finishes
        if not self.pipeline or final:
            show_info("Done processing!")

    def _start_worker(self, worker, *, next_step=None, final=False):
        """
        Attach common signals and start a given worker.

        Parameters
        ----------
        worker : GeneratorWorker
            The worker to start.
        next_step : callable, optional
            The next step to run.
        final : bool, optional
            Whether this is the final step.
        """
        worker.started.connect(self.turn_off_buttons)
        worker.started.connect(self.set_status)
        worker.errored.connect(self._handle_worker_error)

        def _finished():
            self._on_worker_finished(next_step=next_step, final=final)

        worker.finished.connect(_finished)
        worker.start()

    def turn_off_pipeline(self):
        """
        Turn off the pipeline flag to indicate that the full pipeline is no longer running.
        """
        self.pipeline = False

    def run_nellie(self):
        """
        Start the entire Nellie pipeline from preprocessing to feature extraction.
        """
        self.pipeline = True
        self.run_preprocessing()

    def set_status(self):
        """
        Set the status of the processor to indicate that a process is running, and starts the status update timer.
        """
        self.running = True
        self._last_step_had_error = False
        self._last_error_message = ""
        # Update every 500 ms
        self.status_timer.start(500)

    def update_status(self):
        """
        Update the status label with an ellipsis effect to indicate ongoing processing.
        """
        if self.running:
            self.status_label.setText(f"Running {self.status}{'.' * (self.num_ellipses % 4)}")
            self.num_ellipses += 1
        else:
            self.status_timer.stop()

    def reset_status(self):
        """
        Reset the status label to indicate that no process is running.
        """
        self.running = False
        if self._last_error_message:
            # Preserve the last error message so the user can see what failed.
            self.status_label.setText("Error, see console for details")
        else:
            self.status_label.setText("Awaiting your input")
        self.num_ellipses = 1
        self.status_timer.stop()

    def turn_off_buttons(self):
        """
        Disable all buttons to prevent multiple processes from running simultaneously.
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
        Open the output directory of the current image in the system file explorer.
        """
        directory = self.current_im_info.file_info.output_dir
        if os.path.exists(directory):
            try:
                if sys.platform.startswith("win"):  # Windows
                    os.startfile(directory)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":  # macOS
                    subprocess.call(["open", directory])
                else:  # Linux and other POSIX systems
                    subprocess.call(["xdg-open", directory])
            except Exception as exc:
                show_info(f"Could not open output directory: {exc}")
        else:
            show_info("Output directory does not exist.")


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    napari.run()
