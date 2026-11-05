from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pytest_regressions.dataframe_regression import DataFrameRegressionFixture

from activitysim.estimation.test.test_larch_estimation import _regression_check


@pytest.fixture
def parameter_regression(tmp_path, request):
    expected = pd.DataFrame(
        {"value": [1.0, 2.0], "initvalue": [0.0, 0.0]},
        index=pd.Index(["coef_a", "coef_b"], name="param_name"),
    )
    expected.to_csv(tmp_path / "parameters.csv")
    regression = DataFrameRegressionFixture(tmp_path, tmp_path, request)
    return regression, expected


def test_parameter_regression_accepts_reordered_rows(parameter_regression):
    regression, expected = parameter_regression
    _regression_check(regression, expected.iloc[::-1], basename="parameters")


@pytest.mark.parametrize(
    "change", ["missing", "extra", "renamed", "duplicate", "value"]
)
def test_parameter_regression_rejects_changed_parameters(parameter_regression, change):
    regression, expected = parameter_regression
    actual = expected.copy()
    if change == "missing":
        actual = actual.iloc[:1]
    elif change == "extra":
        actual.loc["coef_c"] = [3.0, 0.0]
    elif change == "renamed":
        actual = actual.rename(index={"coef_b": "coef_c"})
    elif change == "duplicate":
        actual.index = ["coef_a", "coef_a"]
    else:
        actual.loc["coef_b", "value"] = 20.0
    with pytest.raises(AssertionError):
        _regression_check(regression, actual, basename="parameters")


def test_parameter_baselines_have_unique_sorted_names():
    baseline_dir = Path(__file__).with_name("test_larch_estimation")
    for filename in [
        "test_simple_simulate_auto_ownership_BHHH_.csv",
        "test_location_model_workplace_location_SLSQP_None_.csv",
        "test_workplace_location.csv",
        "test_tour_and_subtour_mode_choice.csv",
    ]:
        path = baseline_dir / filename
        baseline = pd.read_csv(path)
        if "param_name" not in baseline:
            continue
        names = baseline["param_name"]
        assert names.is_unique, path.name
        assert names.tolist() == sorted(names), path.name
