# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, inject, los, tracing, workflow
from activitysim.core.util import assign_in_place

from .util import estimation, tour_destination

logger = logging.getLogger(__name__)


@workflow.step
def joint_tour_destination(
    whale: workflow.Whale,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    households_merged,
    network_los: los.Network_LOS,
    chunk_size,
):
    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = "joint_tour_destination"
    model_settings_file_name = "joint_tour_destination.yaml"
    model_settings = whale.filesystem.read_model_settings(model_settings_file_name)
    trace_hh_id = whale.settings.trace_hh_id

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        whale.settings.want_dest_choice_sample_tables and sample_table_name is not None
    )

    # choosers are tours - in a sense tours are choosing their destination
    joint_tours = tours[tours.tour_category == "joint"]

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results("joint_tour_destination")
        return

    estimator = estimation.manager.begin_estimation(whale, "joint_tour_destination")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag="SPEC")
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(
            whale.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(whale.get_dataframe("land_use"), "landuse", append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        whale,
        tours,
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

    # add column as we want joint_tours table for tracing.
    joint_tours["destination"] = choices_df.choice
    assign_in_place(tours, joint_tours[["destination"]])
    whale.add_table("tours", tours)

    if want_logsums:
        joint_tours[logsum_column_name] = choices_df["logsum"]
        assign_in_place(tours, joint_tours[[logsum_column_name]])

    tracing.print_summary("destination", joint_tours.destination, describe=True)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        whale.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(joint_tours, label="joint_tour_destination.joint_tours")
