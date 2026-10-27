from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from activitysim.core.calibration import component
from activitysim.core.calibration.component import _evaluate_and_update
from activitysim.core.calibration.expressions import _compute_delta
from activitysim.core.calibration.multiprocess import (
    _install_restored_subprocess_state,
)
from activitysim.core.calibration.orchestrator import (
    _components_ran_for_convergence,
)
from activitysim.core.calibration.reporting import (
    _append_iteration_records,
    _coefficient_trajectory,
    _read_component_iteration_records,
)
from activitysim.core.calibration.settings import (
    CalibrationComponentSettings,
    CalibrationConfig,
)
from activitysim.core.random import Random


@pytest.mark.parametrize(
    ("method", "model_value", "target_value", "damping", "expected"),
    [
        # log_ratio: delta = log(target / model) * damping = log(2) * 0.5
        ("log_ratio", 0.25, 0.5, 0.5, math.log(2) * 0.5),
        (
            # odds_ratio: delta = log((t*(1-m)) / (m*(1-t))) * damping
            "odds_ratio",
            0.4,
            0.6,
            0.5,
            math.log((0.6 * (1 - 0.4)) / (0.4 * (1 - 0.6))) * 0.5,
        ),
    ],
)
def test_compute_delta(method, model_value, target_value, damping, expected):
    delta = _compute_delta(
        method=method,
        model_value=model_value,
        target_value=target_value,
        damping=damping,
        component_name="test_component",
        description="test target",
        default_increment=2.0,
    )

    assert delta == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model_value", "target_value", "expected"),
    [
        (
            0.0,
            0.5,
            2.0,
        ),  # model is zero but target is not: fallback must nudge coefficient upward
        (
            1.0,
            0.5,
            -2.0,
        ),  # model is one but target is not: fallback must nudge coefficient downward
        (0.0, 0.0, 0.0),  # both at boundary zero: no change needed
    ],
)
def test_odds_ratio_boundary_fallback_has_correct_direction(
    model_value, target_value, expected
):
    delta = _compute_delta(
        method="odds_ratio",
        model_value=model_value,
        target_value=target_value,
        damping=1.0,
        component_name="test_component",
        description="test target",
        default_increment=2.0,
    )

    assert delta == expected


def test_evaluate_and_update_only_changes_eligible_coefficients():
    calibration_spec = pd.DataFrame(
        [
            {
                "description": "unconverged",
                "coefficient": "coef_update",
                "model_value": 0.25,
                "target_value": 0.5,
                "hold_fast": False,
                "min": -10.0,
                "max": 10.0,
                "damping": 0.5,
                "method": "log_ratio",
                "tolerance": 0.01,
            },
            {
                "description": "within tolerance",
                "coefficient": "coef_converged",
                "model_value": 0.49,
                "target_value": 0.5,
                "hold_fast": False,
                "min": -10.0,
                "max": 10.0,
                "damping": 1.0,
                "method": "log_ratio",
                "tolerance": 0.02,
            },
            {
                "description": "held fixed",
                "coefficient": "coef_held",
                "model_value": 0.25,
                "target_value": 0.5,
                "hold_fast": True,
                "min": -10.0,
                "max": 10.0,
                "damping": 1.0,
                "method": "log_ratio",
                "tolerance": 0.01,
            },
        ]
    )
    coefficients = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=["coef_update", "coef_converged", "coef_held"],
    )

    records, _, updated, component_converged = _evaluate_and_update(
        component_name="test_component",
        calibration_spec_df=calibration_spec,
        coefficients_df=coefficients,
        eval_context={},
        global_iter=1,
        component_iter=1,
    )

    assert updated.loc["coef_update", "value"] == pytest.approx(
        1.0 + math.log(2) * 0.5
    )  # unconverged: full log-ratio delta applied
    assert (
        updated.loc["coef_converged", "value"] == 2.0
    )  # within tolerance: value must not change
    assert (
        updated.loc["coef_held", "value"] == 3.0
    )  # hold_fast=True: value must not change
    assert records[1]["coef_delta"] == 0.0  # converged row records zero change
    assert records[2]["coef_delta"] == 0.0  # held row records zero change
    assert (
        component_converged is False
    )  # one unconverged row means the whole component is not done


@pytest.mark.parametrize(
    (
        "first_model_idx",
        "last_calib_model_idx",
        "global_iter",
        "start_global_iter",
        "expected",
    ),
    [
        (
            None,
            10,
            1,
            1,
            True,
        ),  # no pre-calibration model ran: always counts toward convergence
        (
            5,
            10,
            1,
            1,
            True,
        ),  # model ran before last calibration model in pipeline: counts
        (
            11,
            10,
            1,
            1,
            False,
        ),  # model ran after last calibration model: skipped on first global iter
        (
            11,
            10,
            2,
            1,
            True,
        ),  # same position but later global iter: skip only applies to first pass
    ],
)
def test_components_ran_for_convergence(
    first_model_idx,
    last_calib_model_idx,
    global_iter,
    start_global_iter,
    expected,
):
    assert (
        _components_ran_for_convergence(
            first_model_idx=first_model_idx,
            last_calib_model_idx=last_calib_model_idx,
            global_iter=global_iter,
            start_global_iter=start_global_iter,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("location", "unknown_setting"),
    [
        ((), "cleanup_pipeline_after_run"),  # unknown at top-level CalibrationConfig
        (("run",), "restart_after"),  # unknown inside CalibrationRunSettings
        (
            ("model_settings", "test_component"),
            "survey_file",
        ),  # unknown inside CalibrationComponentSettings
        (
            ("model_settings", "test_component", "reports"),
            "unexpected_report",
        ),  # unknown inside CalibrationReportSettings
    ],
)
def test_calibration_settings_reject_unknown_fields(location, unknown_setting):
    settings = {
        "enable": True,
        "run": {
            "calibrate_models": ["test_component"],
            "global_iterations": 1,
        },
        "model_settings": {
            "test_component": {
                "calibration_spec": "test_calibration.csv",
                "reports": {"generic": True},
            }
        },
    }
    invalid_settings = copy.deepcopy(settings)
    container = invalid_settings
    for key in location:
        container = container[key]
    container[unknown_setting] = True

    with pytest.raises(ValidationError) as error:
        CalibrationConfig.model_validate(invalid_settings)

    assert error.value.errors()[0]["loc"] == (*location, unknown_setting)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_iterations", 0),
        ("submodel_max_iterations", 0),
    ],
)
def test_calibration_iteration_limits_must_be_positive(field, value):
    settings = {
        "enable": True,
        "run": {
            "calibrate_models": ["test_component"],
            "global_iterations": value if field == "global_iterations" else 1,
        },
        "model_settings": {
            "test_component": {
                "calibration_spec": "test_calibration.csv",
                "submodel_max_iterations": (
                    value if field == "submodel_max_iterations" else 1
                ),
            }
        },
    }

    with pytest.raises(ValidationError) as error:
        CalibrationConfig.model_validate(settings)

    assert error.value.errors()[0]["type"] == "greater_than_equal"


def test_calibration_models_must_be_unique():
    settings = {
        "enable": True,
        "run": {
            "calibrate_models": ["test_component", "test_component"],
            "global_iterations": 1,
        },
        "model_settings": {
            "test_component": {"calibration_spec": "test_calibration.csv"}
        },
    }

    with pytest.raises(ValidationError, match="duplicate model name"):
        CalibrationConfig.model_validate(settings)


class _RestoreCheckpoint:
    def __init__(self, state):
        self.state = state
        self.is_open = True
        self.added = []
        self.last_checkpoint = {}

    def initialize(self):
        self.is_open = False
        self.last_checkpoint = {}

    def store_is_open(self):
        return self.is_open

    def close_store(self):
        self.is_open = False

    def open_store(self, overwrite=False):
        assert overwrite is False
        self.is_open = True

    def add(self, checkpoint_name):
        self.added.append(checkpoint_name)
        self.last_checkpoint = {"checkpoint_name": checkpoint_name}
        for table_name in self.state.existing_table_status:
            self.state.existing_table_status[table_name] = False


class _RestoreState:
    def __init__(self):
        households = pd.DataFrame(index=pd.Index([1], name="household_id"))
        tours = pd.DataFrame(index=pd.Index([10], name="tour_id"))
        self.context = {
            "households": households,
            "tours": tours,
            "rng_channels": ["households", "tours"],
        }
        self.existing_table_status = {"households": False, "tours": False}
        self._rng = Random()
        self._rng.add_channel("households", households)
        self._rng.add_channel("tours", tours)
        self.checkpoint = _RestoreCheckpoint(self)

    def __contains__(self, key):
        return key in self.context

    def registered_tables(self):
        return [name for name in self.existing_table_status if name in self.context]

    def get_injectable(self, name, default=None):
        return self.context.get(name, default)

    def add_injectable(self, name, value):
        self.context[name] = value

    def rng(self):
        return self._rng

    def drop(self, name):
        del self.context[name]

    def init_state(self):
        # Match workflow.State: reset bookkeeping and RNG, retain context.
        self.checkpoint.initialize()
        self._rng = Random()
        self.existing_table_status = {}

    def add_table(self, name, table):
        self.context[name] = table
        self.existing_table_status[name] = True

    def is_table(self, name):
        return name in self.existing_table_status

    def get_dataframe(self, name):
        return self.context[name].copy()


def test_subprocess_restore_is_exact_and_durable():
    state = _RestoreState()
    restored_households = pd.DataFrame(
        {"value": [2]}, index=pd.Index([1], name="household_id")
    )

    _install_restored_subprocess_state(
        state,
        tables={"households": restored_households},
        checkpoint_name="model_a",
    )

    assert "tours" not in state.context
    assert "tours" not in state.rng().channels
    assert "tour_id" not in state.rng().index_to_channel
    assert state.get_injectable("rng_channels") == ["households"]
    pd.testing.assert_frame_equal(
        state.get_dataframe("households"), restored_households
    )
    assert state.checkpoint.last_checkpoint == {"checkpoint_name": "model_a"}
    assert state.checkpoint.added == ["model_a"]


class _State:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def get_output_file_path(self, file_name: str) -> Path:
        return self.output_dir / file_name


def _record(attempt: int, previous: float, next_value: float) -> dict:
    return {
        "global_iter": 1,
        "attempt": attempt,
        "component_iter": 1,
        "description": "test target",
        "component": "model_a",
        "coefficient": "coef_a",
        "target_value": 0.5,
        "model_value": 0.25,
        "difference": 0.25,
        "pct_difference": 50.0,
        "hold_fast": False,
        "prev_coefficient": previous,
        "coef_delta": next_value - previous,
        "next_coefficient": next_value,
        "converged": False,
        "at_min": False,
        "at_max": False,
    }


def test_recovery_attempts_preserve_complete_coefficient_trajectory(tmp_path):
    state = _State(tmp_path)
    # attempt 1: coefficient moves 1.0 → 1.5
    _append_iteration_records(state, "model_a", [_record(1, 1.0, 1.5)])
    # attempt 2: picks up exactly where attempt 1 ended, 1.5 → 1.75
    _append_iteration_records(state, "model_a", [_record(2, 1.5, 1.75)])

    stored = pd.read_csv(tmp_path / "calibration" / "calibration_iteration_records.csv")
    assert list(stored["attempt"]) == [1, 2]
    # the chain must be unbroken: attempt 2 prev_coefficient must equal attempt 1 next_coefficient
    assert stored.loc[1, "prev_coefficient"] == stored.loc[0, "next_coefficient"]

    records = _read_component_iteration_records(state, "model_a")
    trajectory, labels = _coefficient_trajectory(records, ["coef_a"])

    # labels: one "Start" entry (initial value) plus one label per recorded iteration
    assert labels == ["Start", "G1-A1-C1", "G1-A2-C1"]
    assert list(trajectory["coef_a"]) == [1.0, 1.5, 1.75]


@pytest.mark.parametrize(
    "model_values, expected_converged",
    [([0.25, 0.5], True), ([0.25, 0.75], False), ([0.5], True)],
)
def test_component_reports_final_simulation(
    monkeypatch, tmp_path, model_values, expected_converged
):
    """Report final results without losing the last update or updating again."""
    coefficients = pd.DataFrame({"value": [0.0]}, index=["coef"])
    spec = pd.DataFrame(
        [
            dict(
                description="target",
                coefficient="coef",
                model_value="modeled",
                target_value=0.5,
                hold_fast=False,
                min=-10.0,
                max=10.0,
                damping=1.0,
                method="log_ratio",
                tolerance=0.01,
            )
        ]
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(multiprocess=False),
        filesystem=SimpleNamespace(
            read_model_settings=Mock(return_value={}),
            read_model_coefficients=Mock(return_value=coefficients),
        ),
        checkpoint=SimpleNamespace(add=Mock()),
        get_output_file_path=lambda name: tmp_path / name,
    )
    settings = CalibrationComponentSettings(
        calibration_spec="calibration.csv", submodel_max_iterations=1
    )
    bespoke = Mock()
    generic = Mock()
    persist = Mock()
    runs = []
    values = iter(model_values)

    def run_model(**kwargs):
        runs.append(kwargs["run_model_name"])
        state.modeled = next(values)

    monkeypatch.setattr(component, "_run_component_model", run_model)
    monkeypatch.setattr(component, "_read_calibration_spec", lambda *args: spec)
    monkeypatch.setattr(
        component, "_extract_utility_coefficient_names", lambda *args: {"coef"}
    )
    monkeypatch.setattr(
        component, "_load_helper_symbols", lambda *args: ({}, bespoke, None)
    )
    monkeypatch.setattr(
        component,
        "_build_expression_context",
        lambda *args: {"modeled": state.modeled},
    )
    monkeypatch.setattr(component, "_persist_coefficients_to_config", persist)
    monkeypatch.setattr(component, "_write_generic_report", generic)

    result = component._calibrate_component(
        state, "test_component", settings, "prior_step", global_iter=2, attempt=3
    )

    assert result.converged is expected_converged
    assert result.component_iterations == 1
    assert len(runs) == len(model_values)
    if len(model_values) == 2:
        assert runs[-1] == "test_component.c_final;g_i2;a_i3"
    persist.assert_called_once()
    persisted = persist.call_args.args[2]
    expected_coefficient = math.log(2) if len(model_values) == 2 else 0.0
    assert persisted.loc["coef", "value"] == pytest.approx(expected_coefficient)

    history = pd.read_csv(tmp_path / "calibration/calibration_iteration_records.csv")
    local_history = pd.read_csv(
        tmp_path / "calibration/test_component/calibration_iteration_records.csv"
    )
    pd.testing.assert_frame_equal(history, local_history)
    assert history["model_value"].tolist() == model_values
    assert history["component_iter"].tolist() == list(range(1, len(model_values) + 1))
    final = history.iloc[-1]
    assert bool(final["converged"]) is expected_converged
    assert final["prev_coefficient"] == pytest.approx(expected_coefficient)
    assert final["next_coefficient"] == pytest.approx(expected_coefficient)
    assert final["coef_delta"] == 0.0
    if len(model_values) == 2:
        assert history.iloc[0]["coef_delta"] == pytest.approx(math.log(2))

    summary = pd.read_csv(tmp_path / "calibration/calibration_iteration_summary.csv")
    assert len(summary) == len(model_values)
    assert summary.iloc[-1]["num_unconverged"] == int(not expected_converged)
    assert summary.iloc[-1]["max_difference"] == abs(0.5 - model_values[-1])
    assert summary.iloc[-1]["max_change"] == 0.0
    assert generic.call_count == bespoke.call_count == len(model_values)
    assert generic.call_args.args[2][0]["model_value"] == model_values[-1]
    assert bespoke.call_args.args[0] == dict(
        modeled=model_values[-1],
        calibration_global_iteration=2,
        calibration_attempt=3,
        calibration_component_iteration=len(model_values),
    )
    state.checkpoint.add.assert_called_once_with("test_component")


def test_evaluation_only_does_not_clip_or_compute_adjustments(monkeypatch):
    coefficients = pd.DataFrame({"value": [5.0]}, index=["coef"])
    spec = pd.DataFrame(
        [
            dict(
                description="target",
                coefficient="coef",
                model_value=0.0,
                target_value=0.5,
                hold_fast=False,
                min=-1.0,
                max=1.0,
                damping=1.0,
                method="log_ratio",
                tolerance=0.01,
            )
        ]
    )
    compute_delta = Mock(side_effect=AssertionError("must not compute an update"))
    monkeypatch.setattr(component, "_compute_delta", compute_delta)
    records, summary, updated, converged = _evaluate_and_update(
        "test_component", spec, coefficients, {}, 1, 2, update_coefficients=False
    )
    pd.testing.assert_frame_equal(updated, coefficients)
    assert not converged
    assert records[0]["coef_delta"] == summary["max_change"] == 0.0
    assert records[0]["at_max"]
    compute_delta.assert_not_called()


@pytest.fixture
def calibration_spec_csv(tmp_path):
    """Write real CSV inputs so tests exercise optional-column parsing."""

    def read(rows):
        path = tmp_path / "calibration.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        state = SimpleNamespace(
            filesystem=SimpleNamespace(get_config_file_path=lambda name: path)
        )
        return component._read_calibration_spec(state, path.name)

    return read


@pytest.mark.parametrize(
    "method, model_value, target_value, direction",
    [
        ("log_ratio", 0.0, 0.5, 1),
        ("log_ratio", 0.5, 0.0, -1),
        ("odds_ratio", 0.0, 0.5, 1),
        ("odds_ratio", 1.0, 0.5, -1),
    ],
)
@pytest.mark.parametrize("include_increment", [False, True])
def test_csv_default_increment_reaches_coefficient_update(
    calibration_spec_csv,
    method,
    model_value,
    target_value,
    direction,
    include_increment,
):
    rows = []
    for i, increment in enumerate([0.25, None, 0.0, "  "]):
        row = dict(
            description=f"target {i}",
            coefficient=f"coef_{i}",
            model_value=model_value,
            target_value=target_value,
            hold_fast=False,
            min=-10.0,
            max=10.0,
            damping=1.0,
            method=method,
            tolerance=0.01,
        )
        if include_increment:
            row["default_increment"] = increment
        rows.append(row)
    spec = calibration_spec_csv(rows)
    magnitudes = [0.25, 2.0, 0.0, 2.0] if include_increment else [2.0] * 4
    assert spec["default_increment"].tolist() == magnitudes
    coefficients = pd.DataFrame(
        {"value": [1.0] * 4}, index=[f"coef_{i}" for i in range(4)]
    )

    records, _, updated, converged = _evaluate_and_update(
        "test_component", spec, coefficients, {}, 1, 1
    )

    assert not converged
    assert updated["value"].tolist() == pytest.approx(
        [1.0 + direction * magnitude for magnitude in magnitudes]
    )
    assert [row["coef_delta"] for row in records] == pytest.approx(magnitudes)


@pytest.mark.parametrize("increment", ["invalid", "inf", "-inf", -0.25])
def test_csv_rejects_invalid_default_increment(calibration_spec_csv, increment):
    row = dict(
        description="target",
        coefficient="coef",
        model_value=0.0,
        target_value=0.5,
        hold_fast=False,
        min=-10.0,
        max=10.0,
        damping=1.0,
        method="log_ratio",
        tolerance=0.01,
        default_increment=increment,
    )
    with pytest.raises(
        ValueError,
        match="calibration.csv default_increment must be finite, numeric, and nonnegative",
    ):
        calibration_spec_csv([row])


@pytest.mark.parametrize("initial_value", [-5.0, 5.0])
@pytest.mark.parametrize(
    "hold_fast, model_value, expected_converged",
    [(True, 0.25, False), (True, 0.5, True), (False, 0.5, True)],
)
def test_bounds_preserve_held_and_converged_coefficients(
    monkeypatch, caplog, initial_value, hold_fast, model_value, expected_converged
):
    spec = pd.DataFrame(
        [
            dict(
                description="target",
                coefficient="coef",
                model_value=model_value,
                target_value=0.5,
                hold_fast=hold_fast,
                min=-1.0,
                max=1.0,
                damping=1.0,
                method="log_ratio",
                tolerance=0.01,
            )
        ]
    )
    coefficients = pd.DataFrame({"value": [initial_value]}, index=["coef"])
    compute_delta = Mock(side_effect=AssertionError("fixed values need no update"))
    monkeypatch.setattr(component, "_compute_delta", compute_delta)

    component._warn_if_initial_values_outside_bounds(
        "test_component", spec, coefficients
    )
    assert "test_component coefficient coef starts" in caplog.text
    assert (
        "below min bound" if initial_value < 0 else "above max bound"
    ) in caplog.text
    records, summary, updated, converged = _evaluate_and_update(
        "test_component", spec, coefficients, {}, 1, 1
    )

    pd.testing.assert_frame_equal(updated, coefficients)
    compute_delta.assert_not_called()
    assert converged is expected_converged
    assert records[0]["converged"] is expected_converged
    assert records[0]["prev_coefficient"] == initial_value
    assert records[0]["next_coefficient"] == initial_value
    assert records[0]["coef_delta"] == summary["max_change"] == 0.0
    assert records[0]["at_min"] == (initial_value < 0)
    assert records[0]["at_max"] == (initial_value > 0)
    assert summary["num_unconverged"] == int(not expected_converged)


@pytest.mark.parametrize(
    "initial_value, model_value, target_value, expected_value",
    [(0.9, 0.25, 0.5, 1.0), (-0.9, 0.5, 0.25, -1.0)],
)
def test_bounds_still_clip_eligible_updates(
    initial_value, model_value, target_value, expected_value
):
    spec = pd.DataFrame(
        [
            dict(
                description="target",
                coefficient="coef",
                model_value=model_value,
                target_value=target_value,
                hold_fast=False,
                min=-1.0,
                max=1.0,
                damping=1.0,
                method="log_ratio",
                tolerance=0.01,
            )
        ]
    )
    coefficients = pd.DataFrame({"value": [initial_value]}, index=["coef"])

    records, summary, updated, converged = _evaluate_and_update(
        "test_component", spec, coefficients, {}, 1, 1
    )

    assert updated.loc["coef", "value"] == expected_value
    assert not converged
    assert records[0]["coef_delta"] == pytest.approx(0.1)
    assert summary["max_change"] == pytest.approx(0.1)
    assert records[0]["at_min"] == (expected_value == -1.0)
    assert records[0]["at_max"] == (expected_value == 1.0)


@pytest.mark.parametrize("templated", [False, True])
def test_utility_coefficient_names_follow_referenced_template_rows(tmp_path, templated):
    # Only alternative columns count. Labels, descriptions, expressions, and
    # commented rows must not make an otherwise unused template row reachable.
    (tmp_path / "spec.csv").write_text(
        "Label,Description,Expression,A,B\n"
        "coef_unused,coef_unused,coef_unused,coef_generic + 2 * coef_shared,0\n"
        "#comment,ignored,1,coef_unused,coef_unused\n"
    )
    (tmp_path / "sample.csv").write_text(
        "description,expression,utility\nignored,1,-coef_extra\n"
    )
    (tmp_path / "template.csv").write_text(
        "coefficient_name,work,school\n"
        "coef_generic,coef_work,coef_school\n"
        "coef_shared,,\n"
        "coef_extra,coef_extra_work,coef_extra_school\n"
        "coef_unused,coef_unused_work,coef_unused_school\n"
    )
    state = SimpleNamespace(
        filesystem=SimpleNamespace(get_config_file_path=lambda name: tmp_path / name)
    )
    settings = {"SPEC": "spec.csv", "SAMPLE_SPEC": "sample.csv"}
    if templated:
        settings["COEFFICIENT_TEMPLATE"] = "template.csv"
    names = component._extract_utility_coefficient_names(state, settings)
    expected = (
        {
            "coef_work",
            "coef_school",
            "coef_shared",
            "coef_extra_work",
            "coef_extra_school",
        }
        if templated
        else {"coef_generic", "coef_shared", "coef_extra"}
    )
    assert names == expected
    component._validate_calibration_coefficients_against_utility_spec(
        "test_component", pd.DataFrame({"coefficient": sorted(expected)}), names
    )
    with pytest.raises(ValueError, match="coef_unused_work"):
        component._validate_calibration_coefficients_against_utility_spec(
            "test_component", pd.DataFrame({"coefficient": ["coef_unused_work"]}), names
        )
