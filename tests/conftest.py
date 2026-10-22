import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data():
    from wqcadence.data import load_owenmore
    return load_owenmore(ROOT / "data" / "raw")
