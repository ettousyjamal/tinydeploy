from pathlib import Path

import pytest

from tinydeploy.demo import prepare_ota, prepare_selection


@pytest.fixture
def selection_config(tmp_path: Path) -> Path:
    """Three-candidate example configuration."""
    return prepare_selection(tmp_path / "selection")


@pytest.fixture
def ota_inputs(tmp_path: Path):
    """Old/new images and the 500-device fleet configuration."""
    return prepare_ota(tmp_path / "ota-inputs")
