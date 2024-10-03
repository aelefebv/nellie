from importlib_metadata import entry_points


def discover_nellie_plugins():
    nellie_plugins = {}
    for entry_point in entry_points(group='nellie.plugins'):
        plugin_func = entry_point.load()
        nellie_plugins[entry_point.name] = plugin_func
    return nellie_plugins


if __name__ == '__main__':
    plugins = discover_nellie_plugins()
    print(plugins)
