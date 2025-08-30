# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.mode import run_tour_mode_choice_simulate
from activitysim.abm.models.util.logsums import setup_skims
from activitysim.abm.models.park_and_ride_lot_choice import run_park_and_ride_lot_choice
from activitysim.core import config, estimation, expressions, los, tracing, workflow
from activitysim.core.configuration.logit import TourModeComponentSettings
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)


@workflow.step
def atwork_subtour_mode_choice(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: TourModeComponentSettings | None = None,
    model_settings_file_name: str = "tour_mode_choice.yaml",
    trace_label: str = "atwork_subtour_mode_choice",
) -> None:
    """
    At-work subtour mode choice simulate
    """

    trace_hh_id = state.settings.trace_hh_id

    if model_settings is None:
        model_settings = TourModeComponentSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    logsum_column_name = model_settings.MODE_CHOICE_LOGSUM_COLUMN_NAME
    mode_column_name = "tour_mode"

    subtours = tours[tours.tour_category == "atwork"]

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    subtours_merged = pd.merge(
        subtours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
    )

    logger.info("Running %s with %d subtours" % (trace_label, subtours_merged.shape[0]))

    tracing.print_summary(
        "%s tour_type" % trace_label, subtours_merged.tour_type, value_counts=True
    )

    constants = {}
    constants.update(model_settings.CONSTANTS)

    if "pnr_zone_id" in subtours_merged.columns:
        if model_settings.run_atwork_pnr_lot_choice:
            subtours_merged["pnr_zone_id"] = run_park_and_ride_lot_choice(
                state,
                choosers=subtours_merged.copy(),
                land_use=state.get_dataframe("land_use"),
                network_los=network_los,
                model_settings=None,
                choosers_dest_col_name="destination",
                choosers_origin_col_name="workplace_zone_id",
                estimator=None,
                pnr_capacity_cls=None,
                trace_label=tracing.extend_trace_label(trace_label, "pnr_lot_choice"),
            )
        else:
            subtours_merged["pnr_zone_id"].fillna(-1, inplace=True)

    # setup skim keys
    skims = setup_skims(
        network_los,
        subtours_merged,
        add_periods=False,
        include_pnr_skims=model_settings.run_atwork_pnr_lot_choice,
        orig_col_name="workplace_zone_id",
        dest_col_name="destination",
        trace_label=trace_label,
    )

    if network_los.zone_system == los.THREE_ZONE:
        # TVPB constants can appear in expressions
        constants.update(
            network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
        )

    estimator = estimation.manager.begin_estimation(state, "atwork_subtour_mode_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # FIXME run_tour_mode_choice_simulate writes choosers post-annotation

    choices_df = run_tour_mode_choice_simulate(
        state,
        subtours_merged,
        tour_purpose="atwork",
        model_settings=model_settings,
        mode_column_name=mode_column_name,
        logsum_column_name=logsum_column_name,
        network_los=network_los,
        skims=skims,
        constants=constants,
        estimator=estimator,
        trace_label=trace_label,
        trace_choice_name="tour_mode_choice",
    )

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:
        tvpb_mode_path_types = model_settings.tvpb_mode_path_types
        for mode, path_types in tvpb_mode_path_types.items():
            for direction, skim in zip(
                ["od", "do"], [skims["tvpb_logsum_odt"], skims["tvpb_logsum_dot"]]
            ):
                path_type = path_types[direction]
                skim_cache = skim.cache[path_type]

                print(f"mode {mode} direction {direction} path_type {path_type}")

                for c in skim_cache:
                    dest_col = f"{direction}_{c}"

                    if dest_col not in choices_df:
                        choices_df[dest_col] = (
                            np.nan
                            if pd.api.types.is_numeric_dtype(skim_cache[c])
                            else ""
                        )

                    choices_df[dest_col].where(
                        choices_df.tour_mode != mode, skim_cache[c], inplace=True
                    )

    if estimator:
        estimator.write_choices(choices_df[mode_column_name])
        choices_df[mode_column_name] = estimator.get_survey_values(
            choices_df[mode_column_name], "tours", mode_column_name
        )
        estimator.write_override_choices(choices_df[mode_column_name])
        estimator.end_estimation()

    tracing.print_summary(
        "%s choices" % trace_label, choices_df[mode_column_name], value_counts=True
    )

    assign_in_place(
        tours, choices_df, state.settings.downcast_int, state.settings.downcast_float
    )
    state.add_table("tours", tours)

    if trace_hh_id:
        state.tracing.trace_df(
            tours[tours.tour_category == "atwork"],
            label=tracing.extend_trace_label(trace_label, mode_column_name),
            slicer="tour_id",
            index_label="tour_id",
        )

    expressions.annotate_tables(
        state,
        locals_dict=constants,
        skims=skims,
        model_settings=model_settings,
        trace_label=trace_label,
    )
