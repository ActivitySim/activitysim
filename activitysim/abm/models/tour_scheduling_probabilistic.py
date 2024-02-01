# ActivitySim
# See full license in LICENSE.txt
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util import probabilistic_scheduling as ps
from activitysim.core import chunk, estimation, workflow
from activitysim.core.configuration.base import PydanticReadable

logger = logging.getLogger(__name__)


def run_tour_scheduling_probabilistic(
    state: workflow.State,
    tours_df: pd.DataFrame,
    scheduling_probs: pd.DataFrame,
    probs_join_cols: str | list[str],
    depart_alt_base: int,
    trace_label: str,
):
    """Make probabilistic tour scheduling choices in chunks

    Parameters
    ----------
    state: workflow.State
    tours_df : pandas.DataFrame
        table of tours
    scheduling_probs : pandas.DataFrame
        Probability lookup table for tour depature and return times
    probs_join_cols : str or list of strs
        Columns to use for merging probability lookup table with tours table
    depart_alt_base : int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am
    trace_label : str
        label to append to tracing logs and table names

    Returns
    -------
    pandas.Series
        series of chosen alternative indices for each chooser
    """
    result_list = []
    for (
        i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(state, tours_df, trace_label, trace_label):
        choices = ps.make_scheduling_choices(
            state,
            chooser_chunk,
            "departure",
            scheduling_probs,
            probs_join_cols,
            depart_alt_base,
            first_trip_in_leg=False,
            report_failed_trips=True,
            trace_label=chunk_trace_label,
            trace_choice_col_name="depart_return",
            clip_earliest_latest=False,
            chunk_sizer=chunk_sizer,
        )
        result_list.append(choices)

    choices = pd.concat(result_list)
    return choices


class TourSchedulingProbabilisticSettings(PydanticReadable):
    """
    Settings for the `tour_scheduling_probabilistic` component.
    """

    depart_alt_base: int = 0

    PROBS_SPEC: str = "tour_scheduling_probs.csv"
    """Filename for the tour scheduling probabilistic specification (csv) file."""

    PROBS_JOIN_COLS: list[str] | None = None
    """List of columns"""


@workflow.step
def tour_scheduling_probabilistic(
    state: workflow.State,
    tours: pd.DataFrame,
    model_settings: TourSchedulingProbabilisticSettings | None = None,
    model_settings_file_name: str = "tour_scheduling_probabilistic.yaml",
    trace_label: str = "tour_scheduling_probabilistic",
) -> None:
    """Makes tour departure and arrival choices by sampling from a probability lookup table

    This model samples tour scheduling choices from an exogenously defined probability
    distribution rather than simulating choices from a discrete choice model. This is particularly
    useful when estimating from sparse survey data with small numbers of observations
    across tour scheduling alternatives.

    Parameters
    ----------
    tours : DataFrame
        lazy-loaded table of tours
    chunk_size :  int
        size of chooser chunks, defined in main settings.yaml
    trace_hh_id : int
        households to trace, defined in main settings.yaml

    """

    if model_settings is None:
        model_settings = TourSchedulingProbabilisticSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    depart_alt_base = model_settings.depart_alt_base
    scheduling_probs_filepath = state.filesystem.get_config_file_path(
        model_settings.PROBS_SPEC
    )
    scheduling_probs = pd.read_csv(scheduling_probs_filepath)
    probs_join_cols = model_settings.PROBS_JOIN_COLS
    tours_df = tours

    # trip_scheduling is a probabilistic model ane we don't support estimation,
    # but we do need to override choices in estimation mode
    estimator = estimation.manager.begin_estimation(
        state, "tour_scheduling_probabilistic"
    )
    if estimator:
        estimator.write_spec(model_settings, tag="PROBS_SPEC")
        estimator.write_model_settings(model_settings, model_settings_file_name)
        chooser_cols_for_estimation = ["purpose_id"]
        estimator.write_choosers(tours_df[chooser_cols_for_estimation])

    choices = run_tour_scheduling_probabilistic(
        state,
        tours_df,
        scheduling_probs,
        probs_join_cols,
        depart_alt_base,
        trace_label,
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

    state.add_table("tours", tours_df)
