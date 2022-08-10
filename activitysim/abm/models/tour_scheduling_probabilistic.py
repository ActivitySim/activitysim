# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util import estimation
from activitysim.core import chunk, config, inject, logit, pipeline, tracing
from activitysim.core.util import reindex

from .util import probabilistic_scheduling as ps

logger = logging.getLogger(__name__)


def run_tour_scheduling_probabilistic(
    tours_df,
    scheduling_probs,
    probs_join_cols,
    depart_alt_base,
    chunk_size,
    trace_label,
    trace_hh_id,
):
    """Make probabilistic tour scheduling choices in chunks

    Parameters
    ----------
    tours_df : pandas.DataFrame
        table of tours
    scheduling_probs : pandas.DataFrame
        Probability lookup table for tour depature and return times
    probs_join_cols : str or list of strs
        Columns to use for merging probability lookup table with tours table
    depart_alt_base : int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am
    chunk_size : int
        size of chooser chunks, set in main settings.yaml
    trace_label : str
        label to append to tracing logs and table names
    trace_hh_id : int
        households to trace

    Returns
    -------
    pandas.Series
        series of chosen alternative indices for each chooser
    """
    result_list = []
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        tours_df, chunk_size, trace_label, trace_label
    ):
        choices = ps.make_scheduling_choices(
            chooser_chunk,
            "departure",
            scheduling_probs,
            probs_join_cols,
            depart_alt_base,
            first_trip_in_leg=False,
            report_failed_trips=True,
            trace_label=chunk_trace_label,
            trace_hh_id=trace_hh_id,
            trace_choice_col_name="depart_return",
            clip_earliest_latest=False,
        )
        result_list.append(choices)

    choices = pd.concat(result_list)
    return choices


@inject.step()
def tour_scheduling_probabilistic(tours, chunk_size, trace_hh_id):
    """Makes tour departure and arrival choices by sampling from a probability lookup table

    This model samples tour scheduling choices from an exogenously defined probability
    distribution rather than simulating choices from a discrete choice model. This is particularly
    useful when estimating from sparse survey data with small numbers of observations
    across tour scheduling alternatives.

    Parameters
    ----------
    tours :  orca.DataFrameWrapper
        lazy-loaded table of tours
    chunk_size :  int
        size of chooser chunks, defined in main settings.yaml
    trace_hh_id : int
        households to trace, defined in main settings.yaml

    """

    trace_label = "tour_scheduling_probabilistic"
    model_settings_file_name = "tour_scheduling_probabilistic.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)
    depart_alt_base = model_settings.get("depart_alt_base", 0)
    scheduling_probs_filepath = config.config_file_path(model_settings["PROBS_SPEC"])
    scheduling_probs = pd.read_csv(scheduling_probs_filepath)
    probs_join_cols = model_settings["PROBS_JOIN_COLS"]
    tours_df = tours.to_frame()

    # trip_scheduling is a probabilistic model ane we don't support estimation,
    # but we do need to override choices in estimation mode
    estimator = estimation.manager.begin_estimation("tour_scheduling_probabilistic")
    if estimator:
        estimator.write_spec(model_settings, tag="PROBS_SPEC")
        estimator.write_model_settings(model_settings, model_settings_file_name)
        chooser_cols_for_estimation = ["purpose_id"]
        estimator.write_choosers(tours_df[chooser_cols_for_estimation])

    choices = run_tour_scheduling_probabilistic(
        tours_df,
        scheduling_probs,
        probs_join_cols,
        depart_alt_base,
        chunk_size,
        trace_label,
        trace_hh_id,
    )

    # convert alt index choices to depart/return times
    probs_cols = pd.Series(
        [c for c in scheduling_probs.columns if c not in probs_join_cols]
    )
    dep_ret_choices = probs_cols.loc[choices]
    dep_ret_choices.index = choices.index
    choices.update(dep_ret_choices)
    departures = choices.str.split("_").str[0].astype(int)
    returns = choices.str.split("_").str[1].astype(int)

    if estimator:
        estimator.write_choices(choices)
        choices_df = estimator.get_survey_values(
            choices, "tours", ["start", "end"]
        )  # override choices
        choices = choices_df["start"].astype(str) + "_" + choices_df["end"].astype(str)
        estimator.write_override_choices(choices)
        estimator.end_estimation()
        assert not choices.isnull().any()

    # these column names are required for downstream models (e.g. tour mode choice)
    # normally generated by time_windows.py and used as alts for vectorize_tour_scheduling
    tours_df["start"] = departures
    tours_df["end"] = returns
    tours_df["duration"] = tours_df["end"] - tours_df["start"]

    assert not tours_df["start"].isnull().any()
    assert not tours_df["end"].isnull().any()
    assert not tours_df["duration"].isnull().any()

    pipeline.replace_table("tours", tours_df)
