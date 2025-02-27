from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QMessageBox
from qtpy.QtGui import QPixmap, QFont
from qtpy.QtCore import Qt
import napari
import os
import datetime
import numpy as np
from napari.utils.notifications import show_info
import matplotlib.image


class Home(QWidget):
    """
    The Home screen for the Nellie application, displayed in the napari viewer.
    It provides options to start using the application, navigate to the file selection tab, and take screenshots.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    nellie : object
        Reference to the main Nellie object containing image processing pipelines and functions.
    layout : QVBoxLayout
        The vertical layout to organize the widgets on the home screen.
    start_button : QPushButton
        Button to start the application and navigate to the file selection tab.
    screenshot_button : QPushButton
        Button to take a screenshot of the current napari viewer canvas.

    Methods
    -------
    __init__(napari_viewer, nellie, parent=None)
        Initializes the home screen with a logo, title, description, and navigation buttons.
    screenshot(event=None)
        Takes a screenshot of the napari viewer and saves it to a specified folder.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the Home screen with a logo, title, description, and buttons for navigation and screenshot functionality.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the main Nellie object containing image processing pipelines and functions.
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
        logo_label.setPixmap(
            pixmap.scaled(300, 300, Qt.KeepAspectRatio))#, Qt.SmoothTransformation))  # Adjust size as needed
        logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(logo_label)

        # Title
        title = QLabel("Nellie")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy.")
        subtitle.setFont(QFont("Arial", 16))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        self.layout.addWidget(subtitle)

        self.update_text = QLabel("Checking for updates...")
        self.update_text.setFont(QFont("Arial", 16))
        self.update_text.setStyleSheet("color: red")
        self.update_text.setAlignment(Qt.AlignCenter)
        self.update_text.setWordWrap(True)
        self.layout.addWidget(self.update_text)
        self.set_update_status()

        # Add a large "Start" button
        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Arial", 20))
        self.start_button.setFixedWidth(200)
        self.start_button.setFixedHeight(100)
        # rounded-edges
        self.start_button.setStyleSheet("border-radius: 10px;")
        # opens the file select tab
        self.start_button.clicked.connect(lambda: self.nellie.setCurrentIndex(self.nellie.file_select_tab))
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        github_link = QLabel("<a href='https://www.nature.com/articles/s41592-025-02612-7'>Cite our paper!</a>")
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # Link to GitHub
        github_link = QLabel("<a href='https://github.com/aelefebv/nellie'>Visit Nellie's GitHub Page!</a>")
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # screenshot button
        self.screenshot_button = QPushButton(text="Easy screenshot:\n[Ctrl-Shift-E]")
        self.screenshot_button.setStyleSheet("border-radius: 5px;")
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(True)
        self.viewer.bind_key('Ctrl-Shift-E', self.screenshot, overwrite=True)

        self.layout.addWidget(self.screenshot_button, alignment=Qt.AlignCenter)

    def set_update_status(self):
        """
        Checks if the plugin is up to date by comparing the installed version with the latest version on PyPI.
        If an update is available, it displays a warning to the user.
        """
        if self.nellie.current_version is None:
            self.update_text.setText("Failed to check for updates. \n")
        elif self.nellie.latest_version is None:
            self.update_text.setText("Failed to check for updates. \n"
                                     "Please check your internet connection.\n"
                                     f"Current: v{self.nellie.current_version}")
            return
        elif self.nellie.current_version == self.nellie.latest_version:
            self.update_text.setText("Nellie is up-to-date!\n"
                                     f"v{self.nellie.current_version}")
            # green
            self.update_text.setStyleSheet("color: green")
        else:
            self.update_text.setText(
                f"New version available!\n"
                f"Current: v{self.nellie.current_version}\n"
                f"Newest: v{self.nellie.latest_version}\n. "
                f"Please update to the latest version!"
            )
            self.update_text.setStyleSheet("color: red")


    def screenshot(self, event=None):
        """
        Takes a screenshot of the napari viewer and saves it as a PNG file in a specified folder.

        Parameters
        ----------
        event : optional
            An event object, if triggered by a key binding or button click (default is None).
        """
        if self.nellie.im_info is None:
            show_info("No file selected, cannot take screenshot")
            return

        # easy no prompt screenshot
        dt = datetime.datetime.now()  # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt = dt.strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.nellie.im_info.screenshot_dir
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        im_name = f'{dt}-{self.nellie.im_info.file_info.filename_no_ext}.png'
        file_path = os.path.join(screenshot_folder, im_name)

        # Take screenshot
        screenshot = self.viewer.screenshot(canvas_only=True)

        # Save the screenshot
        try:
            # save as png to file_path using imsave
            screenshot = np.ascontiguousarray(screenshot)
            matplotlib.image.imsave(file_path, screenshot, format="png")
            show_info(f"Screenshot saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to save screenshot: {str(e)}")
            raise e


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
