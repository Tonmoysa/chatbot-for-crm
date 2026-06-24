"""Phase 9 / Phase A — YAML-driven conversation scenario tests."""

from __future__ import annotations

import pytest

from tests.helpers.yaml_scenario_runner import load_yaml_scenarios, run_yaml_scenario

_SCENARIOS = load_yaml_scenarios()
_PARAMS = []
for _scenario in _SCENARIOS:
    _marks = [pytest.mark.django_db]
    if _scenario.get("xfail"):
        _marks.append(
            pytest.mark.xfail(reason=str(_scenario["xfail"]), strict=False)
        )
    _PARAMS.append(pytest.param(_scenario, id=_scenario["id"], marks=_marks))


@pytest.mark.parametrize("scenario", _PARAMS)
def test_yaml_scenario(scenario):
    run_yaml_scenario(scenario)
