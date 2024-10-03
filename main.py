import napari
from nellie_napari import NellieLoader
from nellie_napari.discover_plugins import add_nellie_plugins_to_menu


def main():
    viewer = napari.Viewer()
    add_nellie_plugins_to_menu(viewer)
    napari.run()


if __name__ == "__main__":
    main()

