from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from qtpy.QtGui import QPixmap, QFont, QDesktopServices
from qtpy.QtCore import Qt, QUrl
import os
import datetime
from napari.utils.notifications import show_info, show_error


class Home(QWidget):
    """
    The Home screen for the Nellie application, displayed in the napari viewer.
    It provides options to start using the application, navigate to the file
    selection tab, and take screenshots.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    nellie : object
        Reference to the main Nellie object containing image processing
        pipelines and functions.
    layout : QVBoxLayout
        The vertical layout to organize the widgets on the home screen.
    start_button : QPushButton
        Button to start the application and navigate to the file selection tab.
    screenshot_button : QPushButton
        Button to take a screenshot of the current napari viewer canvas.

    Methods
    -------
    __init__(napari_viewer, nellie, parent=None)
        Initializes the home screen with a logo, title, description, and
        navigation buttons.
    set_update_status()
        Updates the label that reports the version / update status.
    screenshot(checked=False)
        Takes a screenshot of the napari viewer and saves it to a specified
        folder.
    """

    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initialize the Home screen with a logo, title, description, and
        buttons for navigation and screenshot functionality.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the main Nellie object containing image processing
            pipelines and functions.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Logo
        logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
        logo_label = QLabel(self)
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            )  # Adjust size as needed
        logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(logo_label)

        # Title
        title = QLabel("Nellie")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Automated organelle segmentation, tracking, and hierarchical "
            "feature extraction in 2D/3D live-cell microscopy."
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        self.layout.addWidget(subtitle)

        # Update status label
        self.update_text = QLabel("Checking for updates...")
        update_font = QFont()
        update_font.setPointSize(12)
        self.update_text.setFont(update_font)
        self.update_text.setAlignment(Qt.AlignCenter)
        self.update_text.setWordWrap(True)
        self.layout.addWidget(self.update_text)

        self.update_meta_text = QLabel("Last checked: pending")
        self.update_meta_text.setFont(update_font)
        self.update_meta_text.setAlignment(Qt.AlignCenter)
        self.update_meta_text.setWordWrap(True)
        self.layout.addWidget(self.update_meta_text)

        self.check_updates_button = QPushButton("Check for updates")
        self.check_updates_button.setToolTip("Re-check the Nellie version on PyPI.")
        self.check_updates_button.clicked.connect(self.check_for_updates)
        self.layout.addWidget(self.check_updates_button, alignment=Qt.AlignCenter)

        self.release_notes_link = QLabel(
            "<a href='https://github.com/aelefebv/nellie/releases'>Release notes</a>"
        )
        self.release_notes_link.setOpenExternalLinks(True)
        self.release_notes_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.release_notes_link)

        # Add a large "Select data" button
        self.start_button = QPushButton("Select data")
        start_font = QFont()
        start_font.setPointSize(18)
        start_font.setBold(True)
        self.start_button.setFont(start_font)
        self.start_button.setMinimumWidth(220)
        self.start_button.setMinimumHeight(80)
        # rounded edges
        self.start_button.setStyleSheet("border-radius: 10px;")
        self.start_button.setToolTip("Go to File validation to choose a dataset.")
        # opens the file select tab
        self.start_button.clicked.connect(
            lambda: self.nellie.setCurrentIndex(self.nellie.file_select_tab)
        )
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        start_hint = QLabel("Choose a file or folder in File validation to begin.")
        start_hint.setAlignment(Qt.AlignCenter)
        start_hint.setWordWrap(True)
        self.layout.addWidget(start_hint)

        quickstart_link = QLabel(
            "<a href='https://github.com/aelefebv/nellie#usage'>Quick start guide</a>"
        )
        quickstart_link.setOpenExternalLinks(True)
        quickstart_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(quickstart_link)

        # Link to paper
        paper_link = QLabel(
            "If you use Nellie, please cite our paper: "
            "<a href='https://www.nature.com/articles/s41592-025-02612-7'>"
            "Nature Methods (2025)</a>"
        )
        paper_link.setOpenExternalLinks(True)
        paper_link.setAlignment(Qt.AlignCenter)
        paper_link.setWordWrap(True)
        self.layout.addWidget(paper_link)

        # Link to GitHub
        github_link = QLabel(
            "<a href='https://github.com/aelefebv/nellie'>Project page and issues</a>"
        )
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # Screenshot button
        self.screenshot_button = QPushButton(text="Take screenshot\n(Ctrl/Cmd+Shift+E)")
        self.screenshot_button.setStyleSheet("border-radius: 5px;")
        self.screenshot_button.setToolTip(
            "Save a canvas-only screenshot of the current napari view\n"
            "into the configured screenshot directory for this dataset."
        )
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(False)
        # Keybinding: use napari's key string format [modifier-]key
        self.viewer.bind_key(
            "Control-Shift-E",
            lambda viewer: self.screenshot(),
            overwrite=True,
        )

        self.layout.addWidget(self.screenshot_button, alignment=Qt.AlignCenter)

        self.screenshot_folder_label = QLabel("Screenshot folder: load data to enable.")
        self.screenshot_folder_label.setAlignment(Qt.AlignCenter)
        self.screenshot_folder_label.setWordWrap(True)
        self.layout.addWidget(self.screenshot_folder_label)

        self.screenshot_open_button = QPushButton("Open screenshot folder")
        self.screenshot_open_button.setToolTip(
            "Open the configured screenshot directory for this dataset."
        )
        self.screenshot_open_button.clicked.connect(self.open_screenshot_folder)
        self.screenshot_open_button.setEnabled(False)
        self.layout.addWidget(self.screenshot_open_button, alignment=Qt.AlignCenter)

        self.update_screenshot_state()

    def showEvent(self, event):
        """
        Update UI state when the Home tab becomes visible.
        """
        super().showEvent(event)
        self.update_screenshot_state()

    def _current_im_info(self):
        """
        Return the active ImInfo if available, handling batch mode safely.
        """
        im_info = getattr(self.nellie, "im_info", None)
        if im_info is None:
            file_select = getattr(self.nellie, "file_select", None)
            if file_select is not None:
                im_info = getattr(file_select, "im_info", None)

        if isinstance(im_info, list):
            return im_info[0] if im_info else None
        return im_info

    def update_screenshot_state(self):
        """
        Enable or disable screenshot actions based on dataset availability.
        """
        im_info = self._current_im_info()
        if im_info is None:
            self.screenshot_button.setEnabled(False)
            self.screenshot_folder_label.setText("Screenshot folder: load data to enable.")
            self.screenshot_open_button.setEnabled(False)
            return

        screenshot_folder = getattr(im_info, "screenshot_dir", None)
        if screenshot_folder:
            self.screenshot_folder_label.setText(
                f"Screenshot folder: {screenshot_folder}"
            )
            self.screenshot_open_button.setEnabled(True)
        else:
            self.screenshot_folder_label.setText("Screenshot folder: unavailable.")
            self.screenshot_open_button.setEnabled(False)

        self.screenshot_button.setEnabled(True)

    def check_for_updates(self):
        """
        Trigger a manual update check using the loader's worker.
        """
        worker = getattr(self.nellie, "version_worker", None)
        if worker is None:
            show_info("Update checker is not available.")
            return
        if worker.isRunning():
            show_info("Already checking for updates.")
            return
        self.update_text.setText("Checking for updates...")
        self.update_text.setStyleSheet("")
        self.update_meta_text.setText("Last checked: checking now...")
        self.check_updates_button.setEnabled(False)
        worker.start()

    def open_screenshot_folder(self):
        """
        Open the dataset screenshot folder in the OS file browser.
        """
        im_info = self._current_im_info()
        if im_info is None:
            show_info("No dataset loaded, cannot open screenshot folder.")
            return
        screenshot_folder = getattr(im_info, "screenshot_dir", None)
        if not screenshot_folder:
            show_info("Screenshot folder is unavailable for this dataset.")
            return
        try:
            os.makedirs(screenshot_folder, exist_ok=True)
        except Exception as e:
            show_error(f"Failed to create screenshot directory:\n{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(screenshot_folder))

    def set_update_status(self):
        """
        Check if the plugin is up to date by comparing the installed version
        with the latest version on PyPI. If an update is available, it displays
        a warning to the user.
        """
        checked_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.update_meta_text.setText(f"Last checked: {checked_time}")
        self.check_updates_button.setEnabled(True)

        current = getattr(self.nellie, "current_version", None)
        latest = getattr(self.nellie, "latest_version", None)

        if current is None and latest is None:
            self.update_text.setText(
                "Update check failed.\n"
                "Unable to determine Nellie version."
            )
            self.update_text.setStyleSheet("color: red")
        elif current is None:
            self.update_text.setText(
                "Update check failed.\n"
                "Unable to determine current version."
            )
            self.update_text.setStyleSheet("color: red")
        elif latest is None:
            self.update_text.setText(
                "Update check failed.\n"
                "Please check your internet connection.\n"
                f"Current: v{current}"
            )
            self.update_text.setStyleSheet("color: red")
        elif current == latest:
            self.update_text.setText(
                "Nellie is up to date.\n"
                f"v{current}"
            )
            self.update_text.setStyleSheet("color: green")
        else:
            self.update_text.setText(
                "New version available.\n"
                f"Current: v{current}\n"
                f"Newest: v{latest}\n"
                "See release notes below."
            )
            self.update_text.setStyleSheet("color: #b36b00")

    def screenshot(self, checked: bool = False):
        """
        Take a screenshot of the napari viewer and saves it as a PNG file in
        a specified folder.

        Parameters
        ----------
        checked : bool, optional
            Parameter provided by the QPushButton clicked signal. Ignored.
        """
        del checked  # unused, but kept for signal compatibility

        im_info = self._current_im_info()
        if im_info is None:
            show_info("No dataset selected, cannot take screenshot")
            return

        # easy no prompt screenshot
        # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = im_info.screenshot_dir
        try:
            os.makedirs(screenshot_folder, exist_ok=True)
        except Exception as e:
            show_error(f"Failed to create screenshot directory:\n{e}")
            return

        im_name = f"{dt_str}-{im_info.file_info.filename_no_ext}.png"
        file_path = os.path.join(screenshot_folder, im_name)

        # Take and save screenshot using napari's built-in API
        try:
            self.viewer.screenshot(
                path=file_path,
                canvas_only=True,
                flash=False,
            )
            show_info(f"Screenshot saved to {file_path}")
        except Exception as e:
            show_error(f"Failed to save screenshot:\n{e}")


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    napari.run()
