# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from activitysim.core import workflow

from .coefficients import _resolve_model_settings_file
from .settings import CalibrationConfig

plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.use("Agg")

CALIBRATION_OUTPUT_DIR = "calibration"
CALIBRATION_ITERATION_FILE = "calibration/calibration_iteration_records.csv"
CALIBRATION_SUMMARY_FILE = "calibration/calibration_iteration_summary.csv"
CALIBRATION_FINAL_COEFFICIENTS_FILE = "calibration/final_calibrated_coefficients.csv"
MAX_COEFFS_IN_GRAPH = 15


def _append_iteration_records(
    state: workflow.State, component_name: str, records: list[dict[str, Any]]
) -> None:
    """Append per-coefficient calibration iteration records."""
    if not records:
        return
    df = pd.DataFrame(records)

    # Save a global iteration history file
    global_path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    _append_csv(
        df,
        global_path,
        unique_on=[
            "global_iter",
            "attempt",
            "component_iter",
            "component",
            "coefficient",
        ],
    )

    # Also write component-local iteration history
    component_path = (
        _component_output_dir(state, component_name)
        / Path(CALIBRATION_ITERATION_FILE).name
    )
    _append_csv(
        df,
        component_path,
        unique_on=[
            "global_iter",
            "attempt",
            "component_iter",
            "component",
            "coefficient",
        ],
    )


def _append_summary_records(
    state: workflow.State, records: list[dict[str, Any]]
) -> None:
    """Append per-iteration summary records."""
    if not records:
        return
    path = state.get_output_file_path(CALIBRATION_SUMMARY_FILE)
    df = pd.DataFrame(records)
    _append_csv(
        df,
        path,
        unique_on=["global_iter", "attempt", "component_iter", "component"],
    )


def _append_csv(
    df: pd.DataFrame, path: Path, unique_on: list[str] | None = None
) -> None:
    """Append a dataframe to a CSV file, replacing rows with matching keys."""
    os.makedirs(path.parent, exist_ok=True)
    if unique_on and path.exists():
        existing = pd.read_csv(path)
        if "attempt" in unique_on and "attempt" not in existing.columns:
            # Histories created before recovery attempts were introduced belong
            # to the first attempt of their logical global iteration.
            existing["attempt"] = 1
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=unique_on, keep="last"
        )
        df.to_csv(path, index=False)
        return

    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def _component_output_dir(state: workflow.State, component_name: str) -> Path:
    """Return output/calibration/<component_name> and ensure it exists."""
    component_dir = state.get_output_file_path(f"calibration/{component_name}")
    os.makedirs(component_dir, exist_ok=True)
    return component_dir


def _write_component_plots(state: workflow.State, component_name: str) -> None:
    """Write/update all standard plots for one calibrated component."""
    recs = _read_component_iteration_records(state, component_name)
    if recs is None or recs.empty:
        return

    # Segment coefficients into manageable sets for plotting
    coefs = sorted(recs.index.get_level_values("coefficient").unique())
    n_sets = math.ceil(len(coefs) / MAX_COEFFS_IN_GRAPH)
    for coef_set in range(n_sets):
        set_coefs = coefs[
            coef_set
            * MAX_COEFFS_IN_GRAPH : min(
                len(coefs), (coef_set + 1) * MAX_COEFFS_IN_GRAPH
            )
        ]
        _plot_coefficient_progress(state, component_name, recs, set_coefs, coef_set)
        last_records = _component_last_records(recs, set_coefs)
        _plot_component_values(state, component_name, last_records, coef_set)
        _plot_component_pct_change(state, component_name, last_records, coef_set)


def _read_component_iteration_records(
    state: workflow.State, component_name: str
) -> pd.DataFrame | None:
    """Read all iteration records for a single component."""
    path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    if not path.exists():
        return None

    iteration_records = pd.read_csv(path)
    if "attempt" not in iteration_records.columns:
        iteration_records["attempt"] = 1
    iteration_records = iteration_records.set_index(
        ["global_iter", "attempt", "component_iter", "coefficient"]
    ).sort_index()
    return iteration_records.loc[iteration_records.component == component_name]


def _plot_coefficient_progress(
    state: workflow.State,
    component_name: str,
    recs: pd.DataFrame,
    set_coefs: list[str],
    coef_set: int,
) -> None:
    """Plot coefficient value progression for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    trajectory, step_labels = _coefficient_trajectory(recs, set_coefs)
    ax = trajectory.plot(figsize=(10, 5))
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels(step_labels, rotation=45, ha="right")
    ax.xaxis.set_label_text("Calibration update (global-attempt-component)")
    ax.yaxis.set_label_text("Coefficient value")
    ax.legend(title="Coefficient label", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    ax.figure.savefig(
        component_dir / f"coefficient_progress_set_{coef_set}.png",
        bbox_inches="tight",
    )
    plt.close(ax.figure)


def _coefficient_trajectory(
    recs: pd.DataFrame,
    set_coefs: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the complete ordered coefficient path and compact step labels."""
    filtered = recs[
        recs.index.get_level_values("coefficient").isin(set_coefs)
    ].reset_index()
    history = filtered.pivot(
        index=["global_iter", "attempt", "component_iter"],
        columns="coefficient",
        values="next_coefficient",
    ).sort_index()
    initial_values = (
        filtered.sort_values(["global_iter", "attempt", "component_iter"])
        .groupby("coefficient", sort=False)
        .first()["prev_coefficient"]
        .reindex(history.columns)
    )
    trajectory = pd.concat(
        [
            pd.DataFrame([initial_values], index=["Start"]),
            history.reset_index(drop=True),
        ]
    )
    step_labels = ["Start"] + [
        f"G{global_iter}-A{attempt}-C{component_iter}"
        for global_iter, attempt, component_iter in history.index
    ]
    return trajectory, step_labels


def _component_last_records(recs: pd.DataFrame, set_coefs: list[str]) -> pd.DataFrame:
    """Select target/model values for the latest iteration and coefficient subset."""
    filtered = recs[recs.index.get_level_values("coefficient").isin(set_coefs)]
    last_global = filtered.index.get_level_values("global_iter")[-1]
    last_attempt = filtered.loc[last_global].index.get_level_values("attempt")[-1]
    last_comp = filtered.loc[(last_global, last_attempt)].index.get_level_values(
        "component_iter"
    )[-1]
    return filtered.xs(
        (last_global, last_attempt, last_comp),
        level=("global_iter", "attempt", "component_iter"),
    )[["target_value", "model_value"]]


def _plot_component_values(
    state: workflow.State,
    component_name: str,
    last_records: pd.DataFrame,
    coef_set: int,
) -> None:
    """Plot final target/model component values for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    ax = last_records.plot.bar(figsize=(10, 5))
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_label_text("Component value")
    plt.tight_layout()
    ax.figure.savefig(component_dir / f"final_components_set_{coef_set}.png")
    plt.close(ax.figure)


def _plot_component_pct_change(
    state: workflow.State,
    component_name: str,
    last_records: pd.DataFrame,
    coef_set: int,
) -> None:
    """Plot final percent difference for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    fig, ax = plt.subplots(figsize=(10, 5))
    pct_diff = last_records.diff(axis=1).model_value / last_records.target_value
    ax = pct_diff.plot.bar(ax=ax)
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_label_text("Coefficient")
    ax.yaxis.set_label_text("% Difference between Model and Target")
    plt.tight_layout()
    ax.figure.savefig(component_dir / f"final_pct_change_set_{coef_set}.png")
    plt.close(ax.figure)


def _write_generic_report(
    state: workflow.State,
    component_name: str,
    row_records: list[dict[str, Any]],
) -> None:
    """Write a simple dashboard-friendly generic report for the current component iteration."""
    if not row_records:
        return

    df = pd.DataFrame(row_records)
    report = (
        df[
            [
                "global_iter",
                "attempt",
                "component_iter",
                "component",
                "description",
                "difference",
                "pct_difference",
                "converged",
            ]
        ]
        .copy()
        .sort_values(["global_iter", "attempt", "component_iter", "description"])
    )

    path = _component_output_dir(state, component_name) / "generic_report.csv"
    _append_csv(
        report,
        path,
        unique_on=[
            "global_iter",
            "attempt",
            "component_iter",
            "component",
            "description",
        ],
    )


def _write_final_coefficients_snapshot(
    state: workflow.State,
    calibration_settings: CalibrationConfig,
) -> None:
    """Write a combined final coefficients file snapshot for calibrated components."""
    frames = []
    for component_name in calibration_settings.run.calibrate_models:
        component_settings = calibration_settings.model_settings[component_name]
        model_settings_file = _resolve_model_settings_file(
            component_name, component_settings
        )
        model_settings = state.filesystem.read_model_settings(
            model_settings_file, mandatory=True
        )
        coeff_df = state.filesystem.read_model_coefficients(
            model_settings=model_settings
        ).copy()
        coeff_df = coeff_df.reset_index().rename(columns={"index": "coefficient_name"})
        coeff_df.insert(0, "component", component_name)
        frames.append(coeff_df)

    if not frames:
        return

    final_df = pd.concat(frames, ignore_index=True)
    path = state.get_output_file_path(CALIBRATION_FINAL_COEFFICIENTS_FILE)
    os.makedirs(path.parent, exist_ok=True)
    final_df.to_csv(path, index=False)


def _ensure_calibration_output_dir(state: workflow.State) -> None:
    """Ensure output/calibration exists."""
    path = state.get_output_file_path(CALIBRATION_OUTPUT_DIR)
    os.makedirs(path, exist_ok=True)
