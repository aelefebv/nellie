from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QMessageBox
from qtpy.QtGui import QPixmap, QFont
from qtpy.QtCore import Qt
import napari
import os
import datetime
import tifffile
from napari.utils.notifications import show_info


class Home(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
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

        self.layout.addWidget(QLabel("\n"))  # Add a bit of space

        # todo link to paper
        github_link = QLabel("<a href='https://arxiv.org/abs/2403.13214'>Cite our paper!</a>")
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # Link to GitHub
        github_link = QLabel("<a href='https://github.com/aelefebv/nellie'>Visit Nellie's GitHub Page!</a>")
        github_link.setOpenExternalLinks(True)
        github_link.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(github_link)

        # screenshot button
        self.screenshot_button = QPushButton(text="Easy screenshot:\n[Ctrl/Cmd-Shift-E]")
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(True)
        self.viewer.bind_key('Ctrl-Shift-E', self.screenshot, overwrite=True)

        self.layout.addWidget(self.screenshot_button, alignment=Qt.AlignCenter)

    def screenshot(self, event=None):
        if self.nellie.im_info is None:
            show_info("No file selected, cannot take screenshot")
            return

        # easy no prompt screenshot
        dt = datetime.datetime.now()  # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt = dt.strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.nellie.im_info.screenshot_dir
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        im_name = f'{dt}-{self.nellie.im_info.basename_no_ext}.png'
        file_path = os.path.join(screenshot_folder, im_name)

        # Take screenshot
        screenshot = self.viewer.screenshot(canvas_only=True)

        # Save the screenshot
        try:
            # save as png to file_path using tifffile
            tifffile.imwrite(file_path, screenshot)
            show_info(f"Screenshot saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to save screenshot: {str(e)}")
            raise e


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
