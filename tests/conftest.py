import importlib.util
import sys
from pathlib import Path


def pytest_configure():
    if importlib.util.find_spec("src") is None:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
