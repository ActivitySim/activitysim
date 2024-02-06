# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util import annotate, tour_destination
from activitysim.core import estimation, los, tracing, workflow
from activitysim.core.configuration.logit import TourLocationComponentSettings
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)


@workflow.step
def non_mandatory_tour_destination(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: TourLocationComponentSettings | None = None,
    model_settings_file_name: str = "non_mandatory_tour_destination.yaml",
    trace_label: str = "non_mandatory_tour_destination",
) -> None:
    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    if model_settings is None:
        model_settings = TourLocationComponentSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trace_hh_id = state.settings.trace_hh_id

    logsum_column_name = model_settings.DEST_CHOICE_LOGSUM_COLUMN_NAME
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.DEST_CHOICE_SAMPLE_TABLE_NAME
    want_sample_table = (
        state.settings.want_dest_choice_sample_tables and sample_table_name is not None
    )

    # choosers are tours - in a sense tours are choosing their destination
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    # separating out pure escort school tours
    # they already have their destination set
    if state.is_table("school_escort_tours"):
        nm_tour_index = non_mandatory_tours.index
        pure_school_escort_tours = non_mandatory_tours[
            (non_mandatory_tours["school_esc_outbound"] == "pure_escort")
            | (non_mandatory_tours["school_esc_inbound"] == "pure_escort")
        ]
        non_mandatory_tours = non_mandatory_tours[
            ~non_mandatory_tours.index.isin(pure_school_escort_tours.index)
        ]

    if non_mandatory_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    estimator = estimation.manager.begin_estimation(
        state, "non_mandatory_tour_destination"
    )
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag="SPEC")
        estimator.set_alt_id(model_settings.ALT_DEST_COL_NAME)
        estimator.write_table(
            state.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(state.get_dataframe("land_use"), "landuse", append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        state,
        non_mandatory_tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        trace_label,
    )

    if estimator:
        estimator.write_choices(choices_df.choice)
        choices_df.choice = estimator.get_survey_values(
            choices_df.choice, "tours", "destination"
        )
        estimator.write_override_choices(choices_df.choice)
        estimator.end_estimation()

    non_mandatory_tours["destination"] = choices_df.choice

    # merging back in school escort tours and preserving index
    if state.is_table("school_escort_tours"):
        non_mandatory_tours = pd.concat(
            [pure_school_escort_tours, non_mandatory_tours]
        ).set_index(nm_tour_index)

    assign_in_place(
        tours,
        non_mandatory_tours[["destination"]],
        state.settings.downcast_int,
        state.settings.downcast_float,
    )

    if want_logsums:
        non_mandatory_tours[logsum_column_name] = choices_df["logsum"]
        assign_in_place(
            tours,
            non_mandatory_tours[[logsum_column_name]],
            state.settings.downcast_int,
            state.settings.downcast_float,
        )

    assert all(
        ~tours["destination"].isna()
    ), f"Tours are missing destination: {tours[tours['destination'].isna()]}"

    state.add_table("tours", tours)

    if model_settings.annotate_tours:
        annotate.annotate_tours(state, model_settings, trace_label)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        state.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        state.tracing.trace_df(
            tours[tours.tour_category == "non_mandatory"],
            label="non_mandatory_tour_destination",
            slicer="person_id",
            index_label="tour",
            columns=None,
            warn_if_empty=True,
        )
