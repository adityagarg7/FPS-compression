import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLES = os.environ.get("GOPRO_SAMPLES", "/home/user/gopro/gpmf-parser/samples")


def sample(name: str) -> str:
    p = os.path.join(SAMPLES, name)
    if not os.path.exists(p):
        pytest.skip(f"sample {name} not available (set GOPRO_SAMPLES)")
    return p


@pytest.fixture(scope="session")
def hero8():
    return sample("hero8.mp4")


@pytest.fixture(scope="session")
def all_samples():
    if not os.path.isdir(SAMPLES):
        pytest.skip("samples dir missing")
    return sorted(os.path.join(SAMPLES, f) for f in os.listdir(SAMPLES) if f.endswith(".mp4"))


@pytest.fixture(scope="session")
def fake50(tmp_path_factory, hero8):
    from tests.make_fixture import make_50fps_fixture
    out = tmp_path_factory.mktemp("fixtures") / "fake50.mp4"
    return make_50fps_fixture(hero8, str(out))
