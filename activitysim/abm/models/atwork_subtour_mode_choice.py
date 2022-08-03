# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    inject,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.util import assign_in_place

from .util import estimation
from .util.mode import run_tour_mode_choice_simulate

logger = logging.getLogger(__name__)


@inject.step()
def atwork_subtour_mode_choice(
    tours, persons_merged, network_los, chunk_size, trace_hh_id
):
    """
    At-work subtour mode choice simulate
    """

    trace_label = "atwork_subtour_mode_choice"

    model_settings_file_name = "tour_mode_choice.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
    mode_column_name = "tour_mode"

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == "atwork"]

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    subtours_merged = pd.merge(
        subtours,
        persons_merged.to_frame(),
        left_on="person_id",
        right_index=True,
        how="left",
    )

    logger.info("Running %s with %d subtours" % (trace_label, subtours_merged.shape[0]))

    tracing.print_summary(
        "%s tour_type" % trace_label, subtours_merged.tour_type, value_counts=True
    )

    constants = {}
    constants.update(config.get_model_constants(model_settings))

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    orig_col_name = "workplace_zone_id"
    dest_col_name = "destination"
    out_time_col_name = "start"
    in_time_col_name = "end"
    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )
    odr_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="in_period"
    )
    dor_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="out_period"
    )
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,
        "dor_skims": dor_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        "orig_col_name": orig_col_name,
        "dest_col_name": dest_col_name,
        "out_time_col_name": out_time_col_name,
        "in_time_col_name": in_time_col_name,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?
        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col_name,
            dest_key=dest_col_name,
            tod_key="out_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col_name,
            dest_key=orig_col_name,
            tod_key="in_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        constants.update(
            network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
        )

    estimator = estimation.manager.begin_estimation("atwork_subtour_mode_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # FIXME run_tour_mode_choice_simulate writes choosers post-annotation

    choices_df = run_tour_mode_choice_simulate(
        subtours_merged,
        tour_purpose="atwork",
        model_settings=model_settings,
        mode_column_name=mode_column_name,
        logsum_column_name=logsum_column_name,
        network_los=network_los,
        skims=skims,
        constants=constants,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="tour_mode_choice",
    )

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get("tvpb_mode_path_types")
        for mode, path_types in tvpb_mode_path_types.items():

            for direction, skim in zip(
                ["od", "do"], [tvpb_logsum_odt, tvpb_logsum_dot]
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

    assign_in_place(tours, choices_df)
    pipeline.replace_table("tours", tours)

    # - annotate tours table
    if model_settings.get("annotate_tours"):
        tours = inject.get_table("tours").to_frame()
        expressions.assign_columns(
            df=tours,
            model_settings=model_settings.get("annotate_tours"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
        )
        pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(
            tours[tours.tour_category == "atwork"],
            label=tracing.extend_trace_label(trace_label, mode_column_name),
            slicer="tour_id",
            index_label="tour_id",
        )
