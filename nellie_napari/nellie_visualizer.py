import os

from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox, QVBoxLayout
import tifffile

from nellie.utils.base_logger import logger
from nellie.tracking.all_tracks_for_label import LabelTracks


class NellieVisualizer(QWidget):
    """
    The NellieVisualizer class provides an interface for visualizing different stages of the Nellie pipeline, such as
    raw images, preprocessed images, segmentation labels, mocap markers, and reassigned labels. It also enables track
    visualization for specific labels and all frame labels using napari.
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer", nellie, parent=None):
        """
        Initializes the NellieVisualizer class, setting up buttons and layout for opening and visualizing images
        and tracks.

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

        # Raw image button
        self.raw_button = QPushButton(text="Open raw image")
        self.raw_button.clicked.connect(self.open_raw)
        self.raw_button.setEnabled(True)
        self.raw_button.setToolTip("Load and show the raw image.")

        # Preprocess open button
        self.open_preprocess_button = QPushButton(text="Open preprocessed image")
        self.open_preprocess_button.clicked.connect(self.open_preprocess_image)
        self.open_preprocess_button.setEnabled(False)
        self.open_preprocess_button.setToolTip("Load and show the preprocessed (Frangi-filtered) image.")

        # Segment open button
        self.open_segment_button = QPushButton(text="Open segmentation images")
        self.open_segment_button.clicked.connect(self.open_segment_image)
        self.open_segment_button.setEnabled(False)
        self.open_segment_button.setToolTip("Load and show the segmentation label images.")

        # Mocap image button
        self.open_mocap_button = QPushButton(text="Open mocap marker image")
        self.open_mocap_button.clicked.connect(self.open_mocap_image)
        self.open_mocap_button.setEnabled(False)
        self.open_mocap_button.setToolTip("Load and show mocap marker image.")

        # Reassign label button
        self.open_reassign_button = QPushButton(text="Open reassigned labels image")
        self.open_reassign_button.clicked.connect(self.open_reassign_image)
        self.open_reassign_button.setEnabled(False)
        self.open_reassign_button.setToolTip("Load and show reassigned branch and object labels.")

        # Tracking buttons
        self.track_button = QPushButton(text="Visualize selected label's tracks")
        self.track_button.clicked.connect(self.on_track_selected)
        self.track_button.setEnabled(False)
        self.track_button.setToolTip("Track the currently selected label in the active labels layer.")

        self.track_all_button = QPushButton(text="Visualize all frame labels' tracks")
        self.track_all_button.clicked.connect(self.track_all)
        self.track_all_button.setEnabled(False)
        self.track_all_button.setToolTip("Track all labels in the active labels layer.")

        self.set_ui()

        self.initialized = False

    def set_ui(self):
        """
        Initializes and sets the layout and UI components for the NellieVisualizer. It groups the buttons for image
        and track visualization into separate sections and arranges them within a vertical layout.
        """
        main_layout = QVBoxLayout()

        # Visualization group
        visualization_group = QGroupBox("Image visualization")
        visualization_layout = QVBoxLayout()
        visualization_layout.addWidget(self.raw_button)
        visualization_layout.addWidget(self.open_preprocess_button)
        visualization_layout.addWidget(self.open_segment_button)
        visualization_layout.addWidget(self.open_mocap_button)
        visualization_layout.addWidget(self.open_reassign_button)
        visualization_group.setLayout(visualization_layout)

        # Tracking group
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
        Post-initialization method that sets the image scale based on the image resolution, makes the scale bar visible,
        and updates button enabled states based on existing files.
        """
        self.set_scale()
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "um"
        except Exception as exc:
            logger.warning(f"Could not configure napari scale bar: {exc}")
        self.initialized = True
        self.check_file_existence()

    def check_3d(self):
        """
        Ensures that the napari viewer is in 3D display mode if the dataset contains Z-dimension data.
        """
        if not self.nellie.im_info.no_z:
            try:
                # Let napari infer ndim from data; only control ndisplay
                self.viewer.dims.ndisplay = 3
            except Exception as exc:
                logger.warning(f"Could not set viewer to 3D display: {exc}")

    def set_scale(self):
        """
        Sets the scale for image display based on the resolution of the Z, Y, and X dimensions of the image.
        """
        dim_res = getattr(self.nellie.im_info, "dim_res", None)
        if not dim_res:
            logger.warning("im_info.dim_res is not available; using scale = (1, 1, 1).")
            self.scale = (1, 1, 1)
            return

        if getattr(self.nellie.im_info, "no_z", False):
            self.scale = (dim_res.get("Y", 1.0), dim_res.get("X", 1.0))
        else:
            self.scale = (
                dim_res.get("Z", 1.0),
                dim_res.get("Y", 1.0),
                dim_res.get("X", 1.0),
            )

    def open_preprocess_image(self):
        """
        Opens and displays the preprocessed (Frangi-filtered) image in the napari viewer.
        Reuses an existing layer if it has already been created.
        """
        self.check_3d()

        if self.frangi_layer is not None and self.frangi_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.frangi_layer
            self._set_status("Activated existing preprocessed image layer.")
            return

        self.im_frangi = self._memmap_pipeline_image("im_preprocessed", "preprocessed image")
        if self.im_frangi is None:
            return

        self.frangi_layer = self.viewer.add_image(
            self.im_frangi,
            name="Pre-processed",
            colormap="turbo",
            scale=self.scale,
        )
        self.frangi_layer.interpolation = "nearest"
        self.viewer.layers.selection.active = self.frangi_layer
        self._set_status("Loaded preprocessed image.")

    def open_segment_image(self):
        """
        Opens and displays the segmentation labels (skeleton relabeled and instance labels) in the napari viewer.
        Reuses existing layers if they have already been created.
        """
        self.check_3d()

        if (
            self.im_instance_label_layer is not None
            and self.im_instance_label_layer in self.viewer.layers
            and self.im_skel_relabelled_layer is not None
            and self.im_skel_relabelled_layer in self.viewer.layers
        ):
            self.viewer.layers.selection.active = self.im_instance_label_layer
            self._set_status("Activated existing segmentation label layers.")
            return

        self.im_instance_label = self._memmap_pipeline_image("im_instance_label", "instance labels")
        self.im_skel_relabelled = self._memmap_pipeline_image("im_skel_relabelled", "skeleton relabeled labels")
        if self.im_instance_label is None or self.im_skel_relabelled is None:
            return

        self.im_skel_relabelled_layer = self.viewer.add_labels(
            self.im_skel_relabelled,
            name="Labels: Branches",
            opacity=1,
            scale=self.scale,
            visible=False,
        )
        self.im_instance_label_layer = self.viewer.add_labels(
            self.im_instance_label,
            name="Labels: Organelles",
            opacity=1,
            scale=self.scale,
        )
        self.viewer.layers.selection.active = self.im_instance_label_layer
        self._set_status("Loaded segmentation label images.")

    def on_track_selected(self):
        """
        Visualizes the tracks for the currently selected label in the napari viewer, based on the active image layer.
        """
        if not self._has_flow_vectors():
            self._set_status(
                "Flow vector array not found; tracking is not available for this dataset.",
                level="warning",
            )
            return

        layer, label_path, _ = self._get_active_label_layer_and_path()
        if layer is None or not label_path:
            self._set_status(
                "Active layer is not a recognized labels layer; select a labels layer before tracking.",
                level="warning",
            )
            return

        # Current time frame
        t = int(self.viewer.dims.current_step[0])

        # Selected label in the active labels layer
        label = getattr(layer, "selected_label", 0)
        if label == 0:
            self._set_status(
                "No label selected (label 0). Select a non-zero label to visualize its tracks.",
                level="warning",
            )
            return

        label_tracks = LabelTracks(im_info=self.nellie.im_info, label_im_path=label_path)
        try:
            label_tracks.initialize()
        except Exception as exc:
            self._set_status(f"Failed to initialize tracking for label {label}: {exc}", level="error")
            return

        # Decide whether to track across all frames or only from current frame
        track_all_frames = (
            self.nellie.settings.track_all_frames.isChecked()
            and (
                layer is self.im_branch_label_reassigned_layer
                or layer is self.im_obj_label_reassigned_layer
            )
        )

        all_tracks, all_props = self._collect_tracks_over_frames(
            label_tracks=label_tracks,
            start_frame=t,
            use_all_frames=track_all_frames,
            label_num=label,
        )

        if not all_tracks:
            self._set_status(f"No tracks found for label {label}.", level="info")
            return

        self.viewer.add_tracks(
            all_tracks,
            properties=all_props,
            name=f"Tracks: Label {label}",
            scale=self.scale,
        )
        self.viewer.layers.selection.active = layer
        self._set_status(f"Added tracks for label {label}.")

    def track_all(self):
        """
        Visualizes tracks for all labels across frames in the napari viewer, based on the active image layer.
        """
        if not self._has_flow_vectors():
            self._set_status(
                "Flow vector array not found; tracking is not available for this dataset.",
                level="warning",
            )
            return

        layer, label_path, _ = self._get_active_label_layer_and_path()
        if layer is None or not label_path:
            self._set_status(
                "Active layer is not a recognized labels layer; select a labels layer before tracking.",
                level="warning",
            )
            return

        t = int(self.viewer.dims.current_step[0])

        label_tracks = LabelTracks(im_info=self.nellie.im_info, label_im_path=label_path)
        try:
            label_tracks.initialize()
        except Exception as exc:
            self._set_status(f"Failed to initialize tracking: {exc}", level="error")
            return

        track_all_frames = (
            self.nellie.settings.track_all_frames.isChecked()
            and (
                layer is self.im_branch_label_reassigned_layer
                or layer is self.im_obj_label_reassigned_layer
            )
        )

        all_tracks, all_props = self._collect_tracks_over_frames(
            label_tracks=label_tracks,
            start_frame=t,
            use_all_frames=track_all_frames,
            label_num=None,
        )

        if not all_tracks:
            self._set_status("No tracks found for any labels.", level="info")
            return

        self.viewer.add_tracks(
            all_tracks,
            properties=all_props,
            name="Tracks: All labels",
            scale=self.scale,
        )
        self.viewer.layers.selection.active = layer
        self._set_status("Added tracks for all labels.")

    def open_mocap_image(self):
        """
        Opens and displays the mocap marker image in the napari viewer.
        Reuses an existing layer if it has already been created.
        """
        self.check_3d()

        if self.im_marker_layer is not None and self.im_marker_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.im_marker_layer
            self._set_status("Activated existing mocap marker image layer.")
            return

        self.im_marker = self._memmap_pipeline_image("im_marker", "mocap marker image")
        if self.im_marker is None:
            return

        self.im_marker_layer = self.viewer.add_image(
            self.im_marker,
            name="Mocap Markers",
            colormap="red",
            blending="additive",
            contrast_limits=[0, 1],
            scale=self.scale,
        )
        self.im_marker_layer.interpolation = "nearest"
        self.viewer.layers.selection.active = self.im_marker_layer
        self._set_status("Loaded mocap marker image.")

    def open_reassign_image(self):
        """
        Opens and displays the reassigned branch and object labels in the napari viewer.
        Reuses existing layers if they have already been created.
        """
        self.check_3d()

        if (
            self.im_branch_label_reassigned_layer is not None
            and self.im_branch_label_reassigned_layer in self.viewer.layers
            and self.im_obj_label_reassigned_layer is not None
            and self.im_obj_label_reassigned_layer in self.viewer.layers
        ):
            self.viewer.layers.selection.active = self.im_obj_label_reassigned_layer
            self._set_status("Activated existing reassigned label layers.")
            return

        self.im_branch_label_reassigned = self._memmap_pipeline_image(
            "im_branch_label_reassigned",
            "reassigned branch labels",
        )
        self.im_obj_label_reassigned = self._memmap_pipeline_image(
            "im_obj_label_reassigned",
            "reassigned object labels",
        )
        if self.im_branch_label_reassigned is None or self.im_obj_label_reassigned is None:
            return

        self.im_branch_label_reassigned_layer = self.viewer.add_labels(
            self.im_branch_label_reassigned,
            name="Reassigned px: Branches",
            scale=self.scale,
            visible=False,
        )
        self.im_obj_label_reassigned_layer = self.viewer.add_labels(
            self.im_obj_label_reassigned,
            name="Reassigned px: Organelles",
            scale=self.scale,
        )
        self.viewer.layers.selection.active = self.im_obj_label_reassigned_layer
        self._set_status("Loaded reassigned label images.")

    def open_raw(self):
        """
        Opens and displays the raw image in the napari viewer.
        Reuses an existing layer if it has already been created.
        """
        self.check_3d()

        if self.raw_layer is not None and self.raw_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.raw_layer
            self._set_status("Activated existing raw image layer.")
            return

        im_path = getattr(self.nellie.im_info, "im_path", None)
        self.im_memmap = self._memmap_tiff_path(im_path, "raw image")
        if self.im_memmap is None:
            return

        self.raw_layer = self.viewer.add_image(
            self.im_memmap,
            name="raw",
            colormap="gray",
            blending="additive",
            scale=self.scale,
        )
        self.raw_layer.interpolation = "nearest"
        self.viewer.layers.selection.active = self.raw_layer
        self._set_status("Loaded raw image.")

    def check_file_existence(self):
        """
        Checks for the existence of files related to different steps of the pipeline, enabling or disabling buttons
        accordingly.
        """
        # Disable all buttons by default; enable selectively based on file existence.
        self.raw_button.setEnabled(False)
        self.open_preprocess_button.setEnabled(False)
        self.open_segment_button.setEnabled(False)
        self.open_mocap_button.setEnabled(False)
        self.open_reassign_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.track_all_button.setEnabled(False)

        # Raw image
        im_path = getattr(self.nellie.im_info, "im_path", None)
        if im_path and os.path.exists(im_path):
            self.raw_button.setEnabled(True)

        # Preprocessed image
        frangi_path = self.nellie.im_info.pipeline_paths.get("im_preprocessed")
        if frangi_path and os.path.exists(frangi_path):
            self.open_preprocess_button.setEnabled(True)

        # Segmentation labels
        im_instance_label_path = self.nellie.im_info.pipeline_paths.get("im_instance_label")
        im_skel_relabelled_path = self.nellie.im_info.pipeline_paths.get("im_skel_relabelled")
        has_seg_labels = (
            im_instance_label_path
            and os.path.exists(im_instance_label_path)
            and im_skel_relabelled_path
            and os.path.exists(im_skel_relabelled_path)
        )
        if has_seg_labels:
            self.open_segment_button.setEnabled(True)

        # Mocap markers
        im_marker_path = self.nellie.im_info.pipeline_paths.get("im_marker")
        if im_marker_path and os.path.exists(im_marker_path):
            self.open_mocap_button.setEnabled(True)

        # Reassigned labels
        im_branch_label_path = self.nellie.im_info.pipeline_paths.get("im_branch_label_reassigned")
        im_obj_label_path = self.nellie.im_info.pipeline_paths.get("im_obj_label_reassigned")
        has_reassign_labels = (
            im_branch_label_path
            and os.path.exists(im_branch_label_path)
            and im_obj_label_path
            and os.path.exists(im_obj_label_path)
        )
        if has_reassign_labels:
            self.open_reassign_button.setEnabled(True)

        # Flow vectors (required for tracking)
        has_flow = self._has_flow_vectors()

        if has_flow and (has_seg_labels or has_reassign_labels):
            # Tracking is only enabled when flow vectors and labels are available.
            self.track_button.setEnabled(True)
            self.track_all_button.setEnabled(True)
            # Tracking settings controls are only meaningful when tracking is possible.
            self.nellie.settings.skip_vox.setEnabled(True)
            self.nellie.settings.track_all_frames.setEnabled(True)
        else:
            self.nellie.settings.skip_vox.setEnabled(False)
            self.nellie.settings.track_all_frames.setEnabled(False)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _set_status(self, message: str, level: str = "info"):
        """
        Set a message in the napari viewer status bar and log it.
        """
        if not message:
            return

        try:
            if hasattr(self.viewer, "status"):
                self.viewer.status = message
        except Exception:
            # Do not let UI feedback errors break the plugin
            pass

        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def _memmap_tiff_path(self, path: str, description: str = ""):
        """
        Safely memory-map a TIFF file from a filesystem path.
        """
        if not path:
            self._set_status(f"No path provided for {description}.", level="warning")
            return None

        if not os.path.exists(path):
            self._set_status(f"File not found for {description}: {path}", level="warning")
            return None

        try:
            return tifffile.memmap(path, mode="r")
        except Exception as exc:
            self._set_status(f"Failed to load {description} from {path}: {exc}", level="error")
            return None

    def _memmap_pipeline_image(self, key: str, description: str = ""):
        """
        Convenience wrapper around _memmap_tiff_path for images in pipeline_paths.
        """
        path = self.nellie.im_info.pipeline_paths.get(key)
        if not path:
            self._set_status(f"Missing pipeline path '{key}' for {description}.", level="warning")
            return None
        return self._memmap_tiff_path(path, description)

    def _get_active_label_layer_and_path(self):
        """
        Determine the currently active labels layer and its corresponding image path.
        Returns (layer, label_path, key) or (None, None, None) if not applicable.
        """
        layer = self.viewer.layers.selection.active

        if layer is self.im_instance_label_layer:
            key = "im_instance_label"
        elif layer is self.im_skel_relabelled_layer:
            key = "im_skel_relabelled"
        elif layer is self.im_branch_label_reassigned_layer:
            key = "im_branch_label_reassigned"
        elif layer is self.im_obj_label_reassigned_layer:
            key = "im_obj_label_reassigned"
        else:
            return None, None, None

        label_path = self.nellie.im_info.pipeline_paths.get(key)
        return layer, label_path, key

    def _has_flow_vectors(self) -> bool:
        """
        Check whether the flow vector array required for tracking exists.
        """
        flow_path = self.nellie.im_info.pipeline_paths.get("flow_vector_array")
        return bool(flow_path and os.path.exists(flow_path))

    def _collect_tracks_over_frames(self, label_tracks, start_frame: int, use_all_frames: bool, label_num=None):
        """
        Run LabelTracks over one or multiple frames, aggregating tracks and properties.

        Parameters
        ----------
        label_tracks : LabelTracks
            Initialized LabelTracks instance.
        start_frame : int
            Frame index from which to start tracking when use_all_frames is False.
        use_all_frames : bool
            If True, iterate over all frames; otherwise track only from start_frame.
        label_num : int or None
            If provided, restrict tracking to this label.

        Returns
        -------
        all_tracks : list
        all_props : dict
        """
        all_tracks = []
        all_props = {}

        skip_coords = self.nellie.settings.skip_vox.value()

        if use_all_frames:
            try:
                num_frames = int(self.nellie.im_info.shape[0])
            except Exception as exc:
                self._set_status(
                    f"Could not determine number of frames from im_info.shape: {exc}",
                    level="error",
                )
                return [], {}

            max_track_num = 0
            for frame in range(num_frames):
                run_kwargs = dict(
                    start_frame=frame,
                    end_frame=None,
                    min_track_num=max_track_num,
                    skip_coords=skip_coords,
                )
                if label_num is not None:
                    run_kwargs["label_num"] = label_num

                try:
                    tracks, track_properties = label_tracks.run(**run_kwargs)
                except Exception as exc:
                    self._set_status(
                        f"Tracking failed at frame {frame}: {exc}",
                        level="error",
                    )
                    return [], {}

                if not tracks:
                    # No more tracks; stop iterating further frames.
                    break

                all_tracks.extend(tracks)
                for prop_name, values in track_properties.items():
                    if prop_name not in all_props:
                        all_props[prop_name] = []
                    all_props[prop_name].extend(values)

                max_track_num = max(track[0] for track in tracks) + 1
        else:
            run_kwargs = dict(
                start_frame=start_frame,
                end_frame=None,
                skip_coords=skip_coords,
            )
            if label_num is not None:
                run_kwargs["label_num"] = label_num

            try:
                all_tracks, all_props = label_tracks.run(**run_kwargs)
            except Exception as exc:
                self._set_status(
                    f"Tracking failed at frame {start_frame}: {exc}",
                    level="error",
                )
                return [], {}

        return all_tracks, all_props


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    napari.run()