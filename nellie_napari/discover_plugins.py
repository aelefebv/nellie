from importlib_metadata import entry_points
from qtpy.QtWidgets import QAction, QMenu


def discover_nellie_plugins():
    nellie_plugins = {}
    for entry_point in entry_points(group='nellie.plugins'):
        plugin_func = entry_point.load()
        nellie_plugins[entry_point.name] = plugin_func
    return nellie_plugins


def add_nellie_plugins_to_menu(napari_viewer):
    nellie_plugins = discover_nellie_plugins()

    # Get the main menu bar
    menu_bar = napari_viewer.window._qt_window.menuBar()

    # Find the Plugins menu
    plugins_menu = None
    for action in menu_bar.actions():
        if action.text() == "&Plugins":
            plugins_menu = action.menu()
            break

    if plugins_menu is None:
        print("Plugins menu not found")
        return

    # Find or create the Nellie submenu
    nellie_menu = None
    for action in plugins_menu.actions():
        if action.text() == "Nellie plugins":
            nellie_menu = action.menu()
            break

    if nellie_menu is None:
        nellie_menu = QMenu("Nellie plugins", napari_viewer.window._qt_window)
        plugins_menu.addMenu(nellie_menu)

    # Add plugins to Nellie submenu
    for plugin_name, plugin_func in nellie_plugins.items():
        action = QAction(plugin_name, napari_viewer.window._qt_window)
        action.triggered.connect(lambda checked, func=plugin_func: func(napari_viewer))
        nellie_menu.addAction(action)


if __name__ == '__main__':
    plugins = discover_nellie_plugins()
    print(plugins)
