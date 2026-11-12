"""Regression test: the example run gives the same deterministic outcome.

The example (``tinydeploy demo``) is seeded, so its selection and OTA results
are fixed. If this test fails, a change altered the selection rule, the delta
codec or the fleet simulation.
"""

import pytest

from tinydeploy._util import read_json
from tinydeploy.demo import EXPECTED_EXAMPLE, prepare_ota, prepare_selection
from tinydeploy.ota import simulate
from tinydeploy.selection import run


def test_example_selection_is_stable(tmp_path):
    expected = EXPECTED_EXAMPLE["selection"]
    result = run(prepare_selection(tmp_path / "in"), tmp_path / "out", allow_synthetic=True)
    rows = {row["id"]: row for row in result["candidates"]}
    assert len(result["candidates"]) == expected["candidates"]
    assert len(result["feasible_ids"]) == expected["feasible"]
    assert len(result["pareto_ids"]) == expected["pareto"]
    assert result["selected"] == expected["selected"]
    assert rows["over_budget"]["rejection_reasons"] == expected["over_budget_rejections"]


def test_example_ota_is_stable(tmp_path):
    expected = EXPECTED_EXAMPLE["ota"]
    old, new, config = prepare_ota(tmp_path / "in")
    result = simulate(old, new, read_json(config), tmp_path / "out")
    for totals in result["modes"].values():
        assert totals["events"] == expected["events_per_mode"]
        assert totals["committed"] == expected["committed"]
        assert totals["rolled_back"] == expected["rolled_back"]
    assert result["modes"]["full"]["transmitted_bytes"] / 1e6 == pytest.approx(expected["full_mb_on_air"], abs=0.005)
    assert result["modes"]["delta"]["transmitted_bytes"] / 1e6 == pytest.approx(expected["delta_mb_on_air"], abs=0.005)
    assert 100 * result["delta_energy_saving_fraction"] == pytest.approx(expected["energy_saving_pct"], abs=0.05)
