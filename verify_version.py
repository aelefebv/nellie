import json
import urllib.request
from importlib.metadata import version as get_version

def check_version():
    print("Checking version...")
    try:
        current = get_version("nellie")
        print(f"Current version: {current}")
    except Exception as e:
        print(f"Failed to get current version: {e}")

    print("Checking PyPI...")
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/nellie/json", timeout=5) as response:
            data = json.loads(response.read().decode())
            latest = data["info"]["version"]
            print(f"Latest version: {latest}")
    except Exception as e:
        print(f"Failed to get latest version: {e}")

if __name__ == "__main__":
    check_version()
