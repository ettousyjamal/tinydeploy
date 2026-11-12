import base64
import random

import pytest

from tinydeploy import TinyDeployError, apply_delta, make_delta
from tinydeploy.delta import wire


@pytest.mark.parametrize("old_size,new_size", [(0, 0), (0, 100), (100, 0), (50, 80), (80, 50), (200, 200)])
def test_roundtrip_random_images(old_size, new_size):
    rng = random.Random(old_size * 1000 + new_size)
    old, new = rng.randbytes(old_size), rng.randbytes(new_size)
    assert apply_delta(old, make_delta(old, new)) == new


def test_small_change_gives_small_delta():
    old = random.Random(1).randbytes(8192)
    new = old[:900] + b"changed" + old[907:]
    delta = make_delta(old, new)
    assert apply_delta(old, delta) == new
    changed = sum(len(base64.b64decode(data)) for _, data in delta["runs"])
    assert changed <= 7  # only bytes that actually differ are sent
    assert len(wire(delta)) < 400


def test_wrong_base_is_rejected():
    with pytest.raises(TinyDeployError, match="base image hash"):
        apply_delta(b"wrong", make_delta(b"old", b"new"))


def test_tampered_run_is_rejected():
    delta = make_delta(b"old", b"new")
    delta["runs"][0][1] = "YmFk"  # base64 of b"bad"
    with pytest.raises(TinyDeployError, match="hash mismatch"):
        apply_delta(b"old", delta)


def test_overlapping_runs_are_rejected():
    delta = make_delta(b"aaaa", b"bbbb")
    delta["runs"].append([1, "Yg=="])
    with pytest.raises(TinyDeployError, match="overlapping"):
        apply_delta(b"aaaa", delta)
