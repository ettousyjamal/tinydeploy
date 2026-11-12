import json
import subprocess
import sys

from tinydeploy import __version__
from tinydeploy._util import read_json
from tinydeploy.cli import main


def tinydeploy(*args):
    return subprocess.run([sys.executable, "-m", "tinydeploy", *map(str, args)],
                          capture_output=True, text=True)


def test_version():
    result = tinydeploy("--version")
    assert result.returncode == 0 and __version__ in result.stdout


def test_run_command_exit_codes(tmp_path, selection_config):
    assert tinydeploy("run", selection_config, "--output", tmp_path / "a").returncode == 1  # synthetic
    ok = tinydeploy("run", selection_config, "--output", tmp_path / "b", "--allow-synthetic")
    assert ok.returncode == 0 and "selected=large" in ok.stdout
    config = read_json(selection_config)
    config["minimum_score"] = 0.99
    selection_config.write_text(json.dumps(config))
    none = tinydeploy("run", selection_config, "--output", tmp_path / "c", "--allow-synthetic")
    assert none.returncode == 2


def test_delta_roundtrip_via_cli(tmp_path, ota_inputs):
    old, new, _ = ota_inputs
    assert main(["delta", str(old), str(new), "--output", str(tmp_path / "d.json")]) == 0
    assert main(["apply-delta", str(old), str(tmp_path / "d.json"), "--output", str(tmp_path / "n.bin")]) == 0
    assert (tmp_path / "n.bin").read_bytes() == new.read_bytes()


def test_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["apply-delta", str(tmp_path / "missing"), str(tmp_path / "x"), "--output", "y"]) == 1
    assert "tinydeploy: error" in capsys.readouterr().err


def test_demo_command(tmp_path):
    result = tinydeploy("demo", tmp_path / "demo")
    assert result.returncode == 0, result.stderr
    summary = read_json(tmp_path / "demo" / "demo_summary.json")
    assert summary["selection"]["selected"] == "large"
    assert summary["grid"]["selected"] == "int8-p30"
