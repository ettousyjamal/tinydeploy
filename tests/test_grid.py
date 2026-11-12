"""Algorithm 1: bit-width x pruning grid with support checks and logged commands."""

import pytest

from tinydeploy import TinyDeployError, run
from tinydeploy._util import read_json, write_json
from tinydeploy.demo import prepare_grid
from tinydeploy.grid import candidate_id


@pytest.fixture
def grid_config(tmp_path):
    return prepare_grid(tmp_path / "grid")


def test_paper_grid_is_eight_configurations(tmp_path, grid_config):
    result = run(grid_config, tmp_path / "run", allow_synthetic=True)
    evaluated = [c["id"] for c in result["candidates"]]
    skipped = [s["id"] for s in result["skipped"]]
    assert len(evaluated) + len(skipped) == 8
    assert evaluated == ["int8-p00", "int8-p30", "int8-p50", "int8-p70"]
    assert skipped == ["int4-p00", "int4-p30", "int4-p50", "int4-p70"]
    assert all("not supported" in s["reason"] for s in result["skipped"])


def test_grid_filters_and_selects_maximum_accuracy(tmp_path, grid_config):
    result = run(grid_config, tmp_path / "run", allow_synthetic=True)
    reasons = {c["id"]: c["rejection_reasons"] for c in result["candidates"]}
    assert reasons == {"int8-p00": ["sram", "latency"], "int8-p30": [], "int8-p50": [],
                       "int8-p70": ["score"]}
    assert result["selected"] == "int8-p30"
    sizes = [c["model_size_bytes"] for c in result["candidates"]]
    assert sizes == sorted(sizes, reverse=True)  # more pruning, smaller model


def test_every_grid_step_is_logged(tmp_path, grid_config):
    run(grid_config, tmp_path / "run", allow_synthetic=True)
    logs = tmp_path / "run" / "logs"
    for cid in ("int8-p00", "int8-p30", "int8-p50", "int8-p70"):
        for step in ("compress", "profile"):
            log = read_json(logs / f"{cid}-{step}.json")
            assert log["returncode"] == 0 and log["argv"]


def test_supporting_int4_evaluates_all_eight(tmp_path, grid_config):
    config = read_json(grid_config)
    config["target"]["supported_bit_widths"] = [8, 4]
    write_json(grid_config, config)
    result = run(grid_config, tmp_path / "run", allow_synthetic=True)
    assert len(result["candidates"]) == 8 and result["skipped"] == []


def test_grid_requires_command_templates(tmp_path, grid_config):
    config = read_json(grid_config)
    del config["grid"]["profile"]
    write_json(grid_config, config)
    with pytest.raises(TinyDeployError, match="profile"):
        run(grid_config, tmp_path / "run", allow_synthetic=True)


def test_candidate_ids():
    assert candidate_id(8, 0.0) == "int8-p00"
    assert candidate_id(4, 0.7) == "int4-p70"
