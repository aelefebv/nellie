from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from qtpy.QtGui import QPixmap, QFont
from qtpy.QtCore import Qt
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
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Automated organelle segmentation, tracking, and hierarchical "
            "feature extraction in 2D/3D live-cell microscopy."
        )
        subtitle.setFont(QFont("Arial", 16))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        self.layout.addWidget(subtitle)

        # Update status label
        self.update_text = QLabel("Checking for updates...")
        self.update_text.setFont(QFont("Arial", 16))
        self.update_text.setAlignment(Qt.AlignCenter)
        self.update_text.setWordWrap(True)
        self.layout.addWidget(self.update_text)

        # Add a large "Start" button
        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Arial", 20))
        self.start_button.setFixedWidth(200)
        self.start_button.setFixedHeight(100)
        # rounded edges
        self.start_button.setStyleSheet("border-radius: 10px;")
        # opens the file select tab
        self.start_button.clicked.connect(
            lambda: self.nellie.setCurrentIndex(self.nellie.file_select_tab)
        )
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # Link to paper
        paper_link = QLabel(
            "<a href='https://www.nature.com/articles/s41592-025-02612-7'>"
            "Cite our paper!</a>"
        )
        paper_link.setOpenExternalLinks(True)
        paper_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(paper_link)

        # Link to GitHub
        github_link = QLabel(
            "<a href='https://github.com/aelefebv/nellie'>Visit Nellie's GitHub Page!</a>"
        )
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # Screenshot button
        self.screenshot_button = QPushButton(text="Easy screenshot:\n[Ctrl-Shift-E]")
        self.screenshot_button.setStyleSheet("border-radius: 5px;")
        self.screenshot_button.setToolTip(
            "Save a canvas-only screenshot of the current napari view\n"
            "into the configured screenshot directory for this dataset."
        )
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(True)
        # Keybinding: use napari's key string format [modifier-]key
        self.viewer.bind_key(
            'Control-Shift-E',
            lambda viewer: self.screenshot(),
            overwrite=True,
        )

        self.layout.addWidget(self.screenshot_button, alignment=Qt.AlignCenter)

    def set_update_status(self):
        """
        Check if the plugin is up to date by comparing the installed version
        with the latest version on PyPI. If an update is available, it displays
        a warning to the user.
        """
        current = getattr(self.nellie, "current_version", None)
        latest = getattr(self.nellie, "latest_version", None)

        if current is None and latest is None:
            self.update_text.setText("Failed to determine Nellie version.\n")
            self.update_text.setStyleSheet("color: red")
        elif current is None:
            self.update_text.setText("Failed to determine current version.\n")
            self.update_text.setStyleSheet("color: red")
        elif latest is None:
            self.update_text.setText(
                "Failed to check for updates.\n"
                "Please check your internet connection.\n"
                f"Current: v{current}"
            )
            self.update_text.setStyleSheet("color: red")
        elif current == latest:
            self.update_text.setText(
                "Nellie is up-to-date!\n"
                f"v{current}"
            )
            self.update_text.setStyleSheet("color: green")
        else:
            self.update_text.setText(
                "New version available!\n"
                f"Current: v{current}\n"
                f"Newest: v{latest}\n"
                "Please update to the latest version!"
            )
            self.update_text.setStyleSheet("color: red")

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

        if self.nellie.im_info is None:
            show_info("No file selected, cannot take screenshot")
            return

        # easy no prompt screenshot
        # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.nellie.im_info.screenshot_dir
        try:
            os.makedirs(screenshot_folder, exist_ok=True)
        except Exception as e:
            show_error(f"Failed to create screenshot directory:\n{e}")
            return

        im_name = f"{dt_str}-{self.nellie.im_info.file_info.filename_no_ext}.png"
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