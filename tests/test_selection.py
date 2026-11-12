import sys

import pytest

from tinydeploy import TinyDeployError, run
from tinydeploy._util import read_json, write_json
from tinydeploy.build import c_header
from tinydeploy.selection import energy_uj, pareto


def test_budget_filter_pareto_and_selection(tmp_path, selection_config):
    result = run(selection_config, tmp_path / "run", allow_synthetic=True)
    rows = {row["id"]: row for row in result["candidates"]}
    assert result["selected"] == "large"
    assert result["feasible_ids"] == ["large", "compact"]
    assert set(result["pareto_ids"]) == {"large", "compact"}
    assert rows["over_budget"]["rejection_reasons"] == ["sram", "latency"]
    assert rows["large"]["estimated_energy_uj"] == pytest.approx(330)  # 20 ms x 5 mA x 3.3 V
    header = (tmp_path / "run" / "model_data.h").read_text()
    assert "tinydeploy_model_size" in header
    assert (tmp_path / "run" / "selected_model.bin").read_bytes() == (selection_config.parent / "large.bin").read_bytes()


def test_run_record_lists_every_candidate(tmp_path, selection_config):
    run(selection_config, tmp_path / "run", allow_synthetic=True)
    record = read_json(tmp_path / "run" / "results.json")
    assert [c["id"] for c in record["candidates"]] == ["large", "compact", "over_budget"]
    assert record["contains_synthetic"] is True
    assert sorted(p.name for p in (tmp_path / "run" / "profiles").iterdir()) == [
        "compact.json", "large.json", "over_budget.json"]


def test_synthetic_profiles_need_explicit_opt_in(tmp_path, selection_config):
    with pytest.raises(TinyDeployError, match="synthetic"):
        run(selection_config, tmp_path / "run")


def test_profile_is_bound_to_model_hash(tmp_path, selection_config):
    (selection_config.parent / "large.bin").write_bytes(b"tampered")
    with pytest.raises(TinyDeployError, match="hash mismatch"):
        run(selection_config, tmp_path / "run", allow_synthetic=True)


def test_no_feasible_candidate(tmp_path, selection_config):
    config = read_json(selection_config)
    config["minimum_score"] = 0.99
    write_json(selection_config, config)
    result = run(selection_config, tmp_path / "run", allow_synthetic=True)
    assert result["selected"] is None
    assert not (tmp_path / "run" / "selected_model.bin").exists()


def test_existing_output_is_never_overwritten(tmp_path, selection_config):
    run(selection_config, tmp_path / "run", allow_synthetic=True)
    with pytest.raises(FileExistsError):
        run(selection_config, tmp_path / "run", allow_synthetic=True)


def test_external_worker_and_build_are_logged(tmp_path, selection_config):
    config = read_json(selection_config)
    config["candidates"][0]["command"] = [sys.executable, "-c", "print('worker invoked')"]
    config["build"] = {"command": [sys.executable, "-c",
                                   "import sys; from pathlib import Path; "
                                   "assert Path(sys.argv[1]).exists(); "
                                   "Path('firmware.bin').write_bytes(b'firmware'); print('built')",
                                   "{model_header}"],
                       "binary": "firmware.bin"}
    write_json(selection_config, config)
    result = run(selection_config, tmp_path / "run", allow_synthetic=True)
    assert result["firmware"]["bytes"] == len(b"firmware")
    assert "worker invoked" in read_json(tmp_path / "run" / "logs" / "large-worker.json")["stdout"]
    build_log = read_json(tmp_path / "run" / "logs" / "build.json")
    assert build_log["returncode"] == 0 and "built" in build_log["stdout"]


def test_failed_build_keeps_its_log(tmp_path, selection_config):
    config = read_json(selection_config)
    config["build"] = {"command": [sys.executable, "-c", "raise SystemExit(3)"], "binary": "missing.bin"}
    write_json(selection_config, config)
    with pytest.raises(TinyDeployError, match="exit code 3"):
        run(selection_config, tmp_path / "run", allow_synthetic=True)
    assert read_json(tmp_path / "run" / "logs" / "build.json")["returncode"] == 3
    assert not (tmp_path / "run" / "firmware.bin").exists()


def test_duplicate_candidate_ids_are_rejected(tmp_path, selection_config):
    config = read_json(selection_config)
    config["candidates"].append(dict(config["candidates"][0]))
    write_json(selection_config, config)
    with pytest.raises(TinyDeployError, match="duplicate"):
        run(selection_config, tmp_path / "run", allow_synthetic=True)


def test_energy_units_and_pareto_dominance():
    assert energy_uj(90, 10, 3.3) == pytest.approx(2970)  # paper: KWS on STM32L476
    rows = [{"id": "a", "score": 0.9, "model_size_bytes": 10, "peak_sram_bytes": 10, "latency_ms": 10},
            {"id": "b", "score": 0.8, "model_size_bytes": 20, "peak_sram_bytes": 20, "latency_ms": 20},
            {"id": "c", "score": 0.8, "model_size_bytes": 5, "peak_sram_bytes": 10, "latency_ms": 10}]
    assert pareto(rows) == ["a", "c"]


def test_c_header_is_valid_byte_array():
    header = c_header(bytes(range(20)))
    assert "0x00, 0x01" in header and "0x13" in header
    assert header.count("0x") == 20
