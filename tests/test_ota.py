import json

import pytest

from tinydeploy import TinyDeployError, make_delta, simulate
from tinydeploy._util import digest, read_json
from tinydeploy.ota import Device


def test_commit_then_rollback_keeps_last_good_image():
    device = Device(b"old", 1024, 64, 16, 32)
    delta = make_delta(b"old", b"new")
    assert device.update(delta, delta["new_sha256"], lambda img: img == b"new", delta=True) == "committed"
    assert device.image == b"new"
    assert device.update(b"bad", digest(b"bad"), lambda _: False) == "rolled_back"
    assert device.image == b"new"


def test_integrity_failure_installs_nothing():
    device = Device(b"old", 1024, 64, 16, 32)
    assert device.update(b"new", "not-the-hash", lambda _: True) == "integrity_rejected"
    assert device.image == b"old"


def test_slot_size_accounts_for_reserved_regions():
    device = Device(b"old", 100, 20, 10, 10)  # (100 - 40) // 2 = 30-byte slots
    assert device.slot_bytes == 30
    with pytest.raises(TinyDeployError, match="A/B flash"):
        device.update(b"x" * 31, digest(b"x" * 31), lambda _: True)


def test_simulation_is_reproducible_and_logged(tmp_path, ota_inputs):
    old, new, config_path = ota_inputs
    config = dict(read_json(config_path), devices=10, rounds=3)
    first = simulate(old, new, config, tmp_path / "a")
    second = simulate(old, new, config, tmp_path / "b")
    assert first == second
    assert (tmp_path / "a" / "events.jsonl").read_bytes() == (tmp_path / "b" / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in (tmp_path / "a" / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2 * 10 * 3
    assert first["modes"]["delta"]["transmitted_bytes"] < first["modes"]["full"]["transmitted_bytes"]


def test_forced_faults_always_roll_back(tmp_path, ota_inputs):
    old, new, config_path = ota_inputs
    config = dict(read_json(config_path), devices=2, rounds=1, packet_loss=0,
                  post_verify_fault_probability=1)
    result = simulate(old, new, config, tmp_path / "ota")
    for mode in result["modes"].values():
        assert mode["rollback_rate"] == 1 and mode["success_rate"] == 0


def test_total_packet_loss_fails_transfers(tmp_path, ota_inputs):
    old, new, config_path = ota_inputs
    config = dict(read_json(config_path), devices=1, rounds=1, packet_loss=1)
    result = simulate(old, new, config, tmp_path / "ota")
    assert result["modes"]["full"]["transfer_failed"] == 1


def test_invalid_configuration_is_rejected(tmp_path, ota_inputs):
    old, new, config_path = ota_inputs
    with pytest.raises(TinyDeployError):
        simulate(old, new, dict(read_json(config_path), packet_loss=1.5), tmp_path / "ota")
