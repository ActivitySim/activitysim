from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from activitysim.core import mp_tasks
from activitysim.core.calibration import component, multiprocess, orchestrator
from activitysim.core.calibration.settings import (
    CalibrationComponentSettings,
    CalibrationConfig,
)
from activitysim.core.workflow.checkpoint import HdfStore, ParquetStore


@pytest.fixture
def subprocess_pipelines(tmp_path, monkeypatch):
    """Two workers, with distinct table versions at the requested checkpoint."""
    monkeypatch.setattr(mp_tasks, "debug", Mock())
    install = Mock()
    monkeypatch.setattr(multiprocess, "_install_restored_subprocess_state", install)

    def prepare(checkpoint_format, slice_info, missing=None):
        store_class = HdfStore if checkpoint_format == "hdf" else ParquetStore
        for worker in range(2):
            if missing == "worker" and worker == 1:
                continue
            store = store_class(tmp_path / f"households_{worker}_pipeline.h5")
            version = f"saved_{worker}"
            household = worker + 1
            person = household * 10
            tour = household * 100
            tables = {
                "households": pd.DataFrame(
                    index=pd.Index([household], name="household_id")
                ),
                "persons": pd.DataFrame(
                    {"household_id": [household]},
                    index=pd.Index([person], name="person_id"),
                ),
                # Core slice inference follows table order. Index-only tables
                # precede the downstream tables that introduce other indexes.
                "person_windows": pd.DataFrame(
                    {"available": [True]}, index=pd.Index([person], name="person_id")
                ),
                "tours": pd.DataFrame(
                    {"person_id": [person]}, index=pd.Index([tour], name="tour_id")
                ),
                "trips": pd.DataFrame(
                    {"tour_id": [tour]}, index=pd.Index([tour * 10], name="trip_id")
                ),
                "joint_tour_participants": pd.DataFrame(
                    {"tour_id": [tour]}, index=pd.Index([tour], name="participant_id")
                ),
                "land_use": pd.DataFrame(
                    {"value": [1]}, index=pd.Index([1], name="zone_id")
                ),
                "weights": pd.DataFrame(
                    {"value": [household]},
                    index=pd.Index([household], name="weight_id"),
                ),
            }
            row = {"checkpoint_name": "model_a", **{name: version for name in tables}}
            if worker == 1 and missing == "mapping":
                row["tours"] = ""
            for name, table in tables.items():
                if not (worker == 1 and missing == "table" and name == "tours"):
                    store.put(name, table, checkpoint_name=version)
                store.put(name, table.iloc[:0], checkpoint_name="later")
            later = {"checkpoint_name": "later", **{name: "later" for name in tables}}
            rows = [later] if worker == 1 and missing == "checkpoint" else [row, later]
            store.put("checkpoints", pd.DataFrame(rows))
            store.close()
        state = SimpleNamespace(
            settings=SimpleNamespace(
                models=["model_a"],
                num_processes=2,
                checkpoint_format=checkpoint_format,
                multiprocess_steps=[
                    SimpleNamespace(
                        name="households",
                        begin="model_a",
                        num_processes=2,
                        slice=SimpleNamespace(model_dump=lambda: slice_info),
                    )
                ],
            ),
            filesystem=SimpleNamespace(pipeline_file_name="pipeline.h5"),
            get_output_file_path=lambda name, prefix: tmp_path / f"{prefix}_{name}",
        )
        return state, install

    return prepare


@pytest.mark.parametrize("checkpoint_format", ["hdf", "parquet"])
@pytest.mark.parametrize("exclude", [None, ["person_windows"], True, "*"])
def test_recovery_coalesces_dependent_and_explicit_tables(
    subprocess_pipelines, checkpoint_format, exclude
):
    slice_info = {
        "tables": ["households", "persons"],
        "exclude": exclude,
        "coalesce": ["weights"],
    }
    state, install = subprocess_pipelines(checkpoint_format, slice_info)
    assert multiprocess._restore_from_subprocess_pipelines(state, "model_a")
    tables = install.call_args.args[1]
    assert tables["households"].index.tolist() == [1, 2]
    assert tables["persons"].index.tolist() == [10, 20]
    assert tables["weights"]["value"].tolist() == [1, 2]
    inferred = exclude not in (True, "*")
    for name, indexes in {
        "tours": [100, 200],
        "trips": [1000, 2000],
        "joint_tour_participants": [100, 200],
    }.items():
        assert tables[name].index.tolist() == (indexes if inferred else indexes[:1])
    assert len(tables["person_windows"]) == (2 if exclude is None else 1)
    assert len(tables["land_use"]) == 1


@pytest.mark.parametrize("checkpoint_format", ["hdf", "parquet"])
@pytest.mark.parametrize("missing", ["worker", "checkpoint", "mapping", "table"])
def test_recovery_does_not_install_partial_snapshots(
    subprocess_pipelines, checkpoint_format, missing
):
    state, install = subprocess_pipelines(
        checkpoint_format, {"tables": ["households", "persons"]}, missing
    )
    assert multiprocess._restore_from_subprocess_pipelines(state, "model_a") is False
    install.assert_not_called()


@pytest.mark.parametrize("multiprocess_enabled", [False, True])
@pytest.mark.parametrize(
    "resume_after", ["model_a", "between_1", "between_2", "output"]
)
@pytest.mark.parametrize("global_iterations", [1, 2])
def test_resume_executes_pending_models_before_calibration(
    monkeypatch, multiprocess_enabled, resume_after, global_iterations
):
    models = ["initialize", "model_a", "between_1", "between_2", "model_b", "output"]
    settings = CalibrationConfig.model_validate(
        {
            "enable": True,
            "run": {
                "calibrate_models": ["model_a", "model_b"],
                "global_iterations": global_iterations,
                "complete_steps": True,
            },
            "model_settings": {
                name: {"calibration_spec": "unused.csv"}
                for name in ["model_a", "model_b"]
            },
        }
    )
    completed = {
        name: {"converged": False}
        for name in ["model_a", "model_b"]
        if models.index(name) <= models.index(resume_after)
    }
    monkeypatch.setattr(
        orchestrator, "read_calibration_settings", lambda state: settings
    )
    monkeypatch.setattr(
        orchestrator,
        "_read_progress",
        lambda state: {"in_progress_iteration": 1, "completed_components": completed},
    )
    for name in [
        "_ensure_calibration_output_dir",
        "_mark_global_iteration_in_progress",
        "_write_progress",
        "_write_completed_progress",
        "_write_final_coefficients_snapshot",
        "_write_component_plots",
    ]:
        monkeypatch.setattr(orchestrator, name, Mock())
    monkeypatch.setattr(orchestrator, "_prep_model_data", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        orchestrator, "_initialize_mp_shared_resources", lambda state: {}
    )
    events = []
    ordinary_runs = []

    def run_models(state, models, resume_after, **kwargs):
        ordinary_runs.append((models, resume_after))
        events.extend(models)

    monkeypatch.setattr(
        orchestrator,
        "_run_in_configured_mode",
        run_models,
    )

    def calibrate(**kwargs):
        events.append(kwargs["component_name"])
        return SimpleNamespace(converged=False, component_iterations=1)

    monkeypatch.setattr(orchestrator, "_calibrate_component", calibrate)
    checkpoint = SimpleNamespace(
        checkpoints=[{"checkpoint_name": resume_after}],
        last_checkpoint={"checkpoint_name": resume_after},
        add=Mock(),
        close_store=Mock(),
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(
            resume_after=resume_after,
            duplicate_step_execution="allow",
            multiprocess=multiprocess_enabled,
        ),
        filesystem=SimpleNamespace(pipeline_file_name="pipeline.h5"),
        checkpoint=checkpoint,
    )
    orchestrator.run_calibration_loop(state, models)
    expected = models[models.index(resume_after) + 1 :]
    if global_iterations == 2:
        expected += models[1:]
    assert events == expected
    if resume_after in ["model_a", "between_1"]:
        assert ordinary_runs[0] == (
            models[models.index(resume_after) + 1 : models.index("model_b")],
            resume_after,
        )


@pytest.mark.parametrize("sharrow", [False, True])
@pytest.mark.parametrize(
    "include_model, iterate", [(True, True), (True, False), (False, True)]
)
def test_calibration_allocates_park_and_ride_buffers(
    monkeypatch, sharrow, include_model, iterate
):
    from activitysim.abm.models import park_and_ride_lot_choice
    from activitysim.abm.models.util import park_and_ride_capacity

    monkeypatch.setattr(mp_tasks, "info", Mock())
    for name in [
        "allocate_shared_skim_buffers",
        "allocate_shared_shadow_pricing_buffers",
        "allocate_shared_shadow_pricing_buffers_choice",
    ]:
        monkeypatch.setattr(mp_tasks, name, lambda state: {})
    monkeypatch.setattr(mp_tasks, "run_sub_task", Mock())
    model_settings = SimpleNamespace(ITERATE_WITH_TOUR_MODE_CHOICE=iterate)
    monkeypatch.setattr(
        park_and_ride_lot_choice.ParkAndRideLotChoiceSettings,
        "read_settings_file",
        Mock(return_value=model_settings),
    )
    pnr = {
        "shared_pnr_choice": [0],
        "shared_pnr_choice_idx": [0],
        "shared_pnr_choice_start": [0],
        "pnr_mp_tally": [0, 0],
    }
    monkeypatch.setattr(
        park_and_ride_capacity,
        "create_park_and_ride_capacity_data_buffers",
        lambda state: pnr,
    )
    monkeypatch.setattr(
        park_and_ride_capacity.ParkAndRideCapacity, "scale_pnr_capacity", Mock()
    )
    injectables = {}
    state = SimpleNamespace(
        settings=SimpleNamespace(
            sharrow=sharrow,
            models=["park_and_ride_lot_choice"] if include_model else ["model_a"],
        ),
        filesystem=SimpleNamespace(),
        get_injectable=lambda name, default=None: injectables.get(name, default),
        add_injectable=lambda name, value: injectables.update({name: value}),
        get_dataframe=lambda name: pd.DataFrame(index=pd.Index([1], name="zone_id")),
    )
    buffers = multiprocess._initialize_mp_shared_resources(state)
    assert injectables["data_buffers"] is buffers
    if include_model and iterate:
        consumer = park_and_ride_capacity.ParkAndRideCapacity(state, model_settings)
        assert consumer.shared_pnr_choice is pnr["shared_pnr_choice"]
    else:
        assert "shared_pnr_choice" not in buffers


def test_missing_coefficients_are_on_disk_before_first_simulation(
    monkeypatch, tmp_path
):
    path = tmp_path / "coefficients.csv"
    pd.DataFrame(
        {"value": [3.0]}, index=pd.Index(["existing"], name="coefficient_name")
    ).to_csv(path)
    settings = {"COEFFICIENTS": path.name}
    state = SimpleNamespace(
        settings=SimpleNamespace(multiprocess=False),
        checkpoint=SimpleNamespace(add=Mock()),
        filesystem=SimpleNamespace(
            read_model_settings=lambda *args, **kwargs: settings,
            read_model_coefficients=lambda **kwargs: pd.read_csv(
                path, index_col="coefficient_name"
            ),
            get_config_file_path=lambda name: tmp_path / name,
        ),
        get_output_file_path=lambda name: tmp_path / name,
    )
    spec = pd.DataFrame(
        [
            dict(
                description="new",
                coefficient="new",
                model_value=1.0,
                target_value=1.0,
                hold_fast=False,
                min=-10,
                max=10,
                damping=1,
                method="log_ratio",
                tolerance=0.01,
            )
        ]
    )
    monkeypatch.setattr(component, "_read_calibration_spec", lambda *args: spec)
    monkeypatch.setattr(
        component, "_extract_utility_coefficient_names", lambda *args: {"new"}
    )
    monkeypatch.setattr(
        component, "_load_helper_symbols", lambda *args: ({}, None, None)
    )
    monkeypatch.setattr(component, "_build_expression_context", lambda *args: {})
    runs = []

    def simulate(**kwargs):
        coefficients = pd.read_csv(path, index_col="coefficient_name")
        assert coefficients.loc["new", "value"] == 0.0
        assert coefficients.loc["existing", "value"] == 3.0
        runs.append(kwargs["run_model_name"])

    monkeypatch.setattr(component, "_run_component_model", simulate)
    component._calibrate_component(
        state,
        "model_a",
        CalibrationComponentSettings(
            calibration_spec="unused.csv", reports={"generic": False}
        ),
        "initialize",
        1,
        1,
    )
    assert len(runs) == 1


@pytest.mark.parametrize(
    "flags",
    [
        [1, None, 0],
        [1.0, None, 0.0],
        [" TRUE ", "false", "yes", "N", "t", "y", "1", "0"],
    ],
)
def test_csv_hold_flags_preserve_held_coefficients(tmp_path, flags):
    path = tmp_path / "calibration.csv"
    rows = [
        dict(
            description=str(i),
            coefficient=f"coef_{i}",
            model_value=0.25,
            target_value=0.5,
            hold_fast=flag,
            min=-10,
            max=10,
            damping=1,
            method="log_ratio",
            tolerance=0.01,
        )
        for i, flag in enumerate(flags)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    state = SimpleNamespace(
        filesystem=SimpleNamespace(get_config_file_path=lambda name: path)
    )
    spec = component._read_calibration_spec(state, path.name)
    expected = (
        [True, False, False]
        if len(flags) == 3
        else [True, False, True, False, True, True, True, False]
    )
    assert spec.hold_fast.tolist() == expected
    coefficients = pd.DataFrame({"value": [1.0] * len(flags)}, index=spec.coefficient)
    _, _, updated, _ = component._evaluate_and_update(
        "model_a", spec, coefficients, {}, 1, 1
    )
    for held, value in zip(expected, updated.value, strict=True):
        assert (value == 1.0) == held
