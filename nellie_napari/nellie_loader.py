import json
import logging
import urllib.request
from importlib.metadata import version as get_version

from qtpy.QtWidgets import QTabWidget
from qtpy.QtCore import QThread, Signal
from napari.utils.notifications import show_info, show_warning

from nellie_napari import NellieProcessor
from nellie_napari.discover_plugins import add_nellie_plugins_to_menu
from nellie_napari.nellie_home import Home
from nellie_napari.nellie_analysis import NellieAnalysis
from nellie_napari.nellie_fileselect import NellieFileSelect
from nellie_napari.nellie_settings import Settings
from nellie_napari.nellie_visualizer import NellieVisualizer

logger = logging.getLogger(__name__)


class VersionWorker(QThread):
    """
    Worker thread to check for the latest version of Nellie on PyPI.
    """
    finished_check = Signal(str, str)  # current_version, latest_version

    def run(self):
        current_version = None
        latest_version = None

        try:
            current_version = get_version("nellie")
        except Exception as e:
            logger.debug("Unable to determine current Nellie version: %s", e)

        try:
            with urllib.request.urlopen(
                "https://pypi.org/pypi/nellie/json", timeout=5
            ) as response:
                data = json.loads(response.read().decode())
                latest_version = data["info"]["version"]
        except Exception as e:
            logger.debug("Unable to fetch Nellie version info from PyPI: %s", e)

        self.finished_check.emit(current_version, latest_version)


class NellieLoader(QTabWidget):
    """
    The main loader class for managing the different stages of the Nellie pipeline within the napari viewer. This class
    provides a tabbed interface for file selection, processing, visualization, analysis, and settings management.

    Attributes
    ----------
    home : Home
        The home tab instance, providing an overview of the Nellie pipeline.
    file_select : NellieFileSelect
        The file selection tab instance, allowing users to select and validate image files.
    processor : NellieProcessor
        The image processing tab instance, where users can process images through the Nellie pipeline.
    visualizer : NellieVisualizer
        The visualization tab instance, where processed images can be visualized.
    analyzer : NellieAnalysis
        The analysis tab instance, enabling users to analyze processed image data.
    settings : Settings
        The settings tab instance, allowing users to configure various settings for the Nellie pipeline.
    home_tab, file_select_tab, processor_tab, visualizer_tab, analysis_tab, settings_tab : int
        Integer values representing the index of the respective tabs.
    im_info : ImInfo or None
        Contains metadata and information about the selected image file.
    im_info_list : list of ImInfo or None
        A list of ImInfo objects when batch processing is enabled (multiple files).

    Methods
    -------
    add_tabs()
        Adds the individual tabs to the widget.
    reset()
        Resets the state of the loader, removing and reinitializing all tabs.
    on_tab_change(index)
        Slot that is triggered when the user changes the tab.
    go_process()
        Initializes and enables the processing and visualization tabs for image processing.
    """

    def __init__(self, napari_viewer: "napari.Viewer", parent=None):
        """
        Initialize the NellieLoader class, creating instances of the individual tabs for home, file selection,
        processing, visualization, analysis, and settings.

        Parameters
        ----------
        napari_viewer : napari.Viewer
            Reference to the napari viewer instance.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)

        self.viewer = napari_viewer

        # Version information for update checking
        self.current_version = None
        self.latest_version = None

        # Start version check in background
        self.version_worker = VersionWorker()
        self.version_worker.finished_check.connect(self.on_version_checked)
        self.version_worker.start()

        # Tab widgets
        self.home = Home(self.viewer, self)
        self.file_select = NellieFileSelect(self.viewer, self)
        self.processor = NellieProcessor(self.viewer, self)
        self.visualizer = NellieVisualizer(self.viewer, self)
        self.analyzer = NellieAnalysis(self.viewer, self)
        self.settings = Settings(self.viewer, self)

        # Tab indices
        self.home_tab = None
        self.file_select_tab = None
        self.processor_tab = None
        self.visualizer_tab = None
        self.analysis_tab = None
        self.settings_tab = None

        self.add_tabs()
        self.currentChanged.connect(self.on_tab_change)

        self.im_info = None
        self.im_info_list = None

        add_nellie_plugins_to_menu(self)

    def on_version_checked(self, current_version, latest_version):
        """
        Slot called when the version check worker finishes.
        Update the version info and the Home tab status.
        """
        self.current_version = current_version
        self.latest_version = latest_version

        # Update Home tab
        if hasattr(self.home, "set_update_status"):
            self.home.set_update_status()

        # Show warning if update available
        if self.current_version and self.latest_version:
            # Compare versions robustly if packaging is available
            try:
                from packaging.version import Version

                if Version(self.latest_version) > Version(self.current_version):
                    show_warning(
                        f"A new version of Nellie ({self.latest_version}) is available. "
                        f"You are using {self.current_version}."
                    )
            except Exception:
                # Fallback: avoid breaking if packaging is unavailable; only warn on inequality
                if self.latest_version != self.current_version:
                    show_warning(
                        f"A new version of Nellie ({self.latest_version}) is available. "
                        f"You are using {self.current_version}."
                    )

    def add_tabs(self):
        """
        Add the individual tabs for Home, File validation, Process, Visualize, Analyze, and Settings.
        Initially disables the Process, Visualize, and Analyze tabs until they are needed.
        """
        # Any additional layout / configuration for the QTabWidget can go here
        ...

        self.home_tab = self.addTab(self.home, "Home")
        self.file_select_tab = self.addTab(self.file_select, "File validation")
        self.processor_tab = self.addTab(self.processor, "Process")
        self.visualizer_tab = self.addTab(self.visualizer, "Visualize")
        self.analysis_tab = self.addTab(self.analyzer, "Analyze")
        self.settings_tab = self.addTab(self.settings, "Settings")

        self.setTabEnabled(self.processor_tab, False)
        self.setTabEnabled(self.visualizer_tab, False)
        self.setTabEnabled(self.analysis_tab, False)

    def reset(self):
        """
        Reset the state of the loader, reinitializing all tabs. This method is typically called when the user
        wants to start a new session with a fresh file selection and settings.
        """
        # Remove and delete all existing tab widgets
        while self.count():
            widget = self.widget(0)
            self.removeTab(0)
            if widget is not None:
                widget.deleteLater()

        # Recreate all tab widgets with a clean state
        self.home = Home(self.viewer, self)
        self.file_select = NellieFileSelect(self.viewer, self)
        self.processor = NellieProcessor(self.viewer, self)
        self.visualizer = NellieVisualizer(self.viewer, self)
        self.analyzer = NellieAnalysis(self.viewer, self)
        self.settings = Settings(self.viewer, self)

        # Re-add tabs and reset indices
        self.add_tabs()
        self.setCurrentIndex(self.home_tab)

        # Reset pipeline state
        self.im_info = None
        self.im_info_list = None

        # Allow update check to run again in the new session if needed
        self.current_version = None
        self.latest_version = None
        if not self.version_worker.isRunning():
            self.version_worker.start()

    def on_tab_change(self, index: int):
        """
        Event handler that is triggered when the user changes the active tab. Initializes the Analyze or Visualize
        tabs if they are selected for the first time, and initializes the Settings tab when selected.

        Parameters
        ----------
        index : int
            The index of the newly selected tab.
        """
        if index == self.analysis_tab:
            # Lazy initialization of analysis tab
            if not getattr(self.analyzer, "initialized", False):
                show_info("Initializing analysis tab")
                self.analyzer.post_init()

        elif index == self.visualizer_tab:
            # Lazy initialization of visualizer tab
            if not getattr(self.visualizer, "initialized", False):
                show_info("Initializing visualizer tab")
                self.visualizer.post_init()

        elif index == self.settings_tab:
            # Run the update check lazily when the user opens Settings
            # Allow Settings to perform any initialization it needs
            if hasattr(self.settings, "post_init"):
                self.settings.post_init()

    def go_process(self):
        """
        Prepare the image(s) for processing and visualization. This method is called after a file has been selected
        and validated. It enables the Process and Visualize tabs and initializes them.
        """
        if self.file_select.batch_fileinfo_list is None:
            self.im_info = self.file_select.im_info
            self.im_info_list = None
        else:
            # For batch mode, im_info is assumed to be a list; store both the first entry and the full list.
            self.im_info = self.file_select.im_info[0]
            self.im_info_list = self.file_select.im_info

        self.setTabEnabled(self.processor_tab, True)
        self.setTabEnabled(self.visualizer_tab, True)

        # Initialize processor and visualizer; they may rely on im_info/im_info_list
        if hasattr(self.processor, "post_init"):
            self.processor.post_init()
        if hasattr(self.visualizer, "post_init"):
            self.visualizer.post_init()

        # Switch to the Process tab; currentChanged signal will trigger on_tab_change
        self.setCurrentIndex(self.processor_tab)


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    loader = NellieLoader(viewer)
    viewer.window.add_dock_widget(loader, name="Nellie", area="right")
    napari.run()