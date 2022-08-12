# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import chunk, config, inject, logit, pipeline, simulate, tracing

logger = logging.getLogger(__name__)


def _clip_probs(choosers_df, probs, depart_alt_base):
    """
    zero out probs before chooser.earliest or after chooser.latest

    Parameters
    ----------
    choosers_df: pd.DataFrame
    probs: pd.DataFrame
        one row per chooser, one column per time period, with float prob of picking that time period

    depart_alt_base: int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am

    Returns
    -------
    probs: pd.DataFrame
        clipped version of probs

    """

    # there should be one row in probs per trip
    assert choosers_df.shape[0] == probs.shape[0]

    # probs should sum to 1 across rows before clipping
    probs = probs.div(probs.sum(axis=1), axis=0)

    num_rows, num_cols = probs.shape
    ix_map = (
        np.tile(np.arange(0, num_cols), num_rows).reshape(num_rows, num_cols)
        + depart_alt_base
    )
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...

    clip_mask = (
        (ix_map >= choosers_df.earliest.values.reshape(num_rows, 1))
        & (ix_map <= choosers_df.latest.values.reshape(num_rows, 1))
    ) * 1
    #  [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0]
    #  [0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]...

    probs = probs * clip_mask

    return probs


def _report_bad_choices(bad_row_map, df, filename, trace_label, trace_choosers=None):
    """

    Parameters
    ----------
    bad_row_map
    df : pandas.DataFrame
        utils or probs dataframe
    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        because we  can't deduce hh_id from the interaction_dataset which is indexed on index
        values from alternatives df

    """

    df = df[bad_row_map]
    if trace_choosers is None:
        hh_ids = tracing.hh_id_for_chooser(df.index, df)
    else:
        hh_ids = tracing.hh_id_for_chooser(df.index, trace_choosers)
    df["household_id"] = hh_ids

    filename = "%s.%s" % (trace_label, filename)

    logger.info("dumping %s" % filename)
    tracing.write_csv(df, file_name=filename, transpose=False)

    # log the indexes of the first MAX_PRINT offending rows
    MAX_PRINT = 0
    for idx in df.index[:MAX_PRINT].values:

        row_msg = "%s : failed %s = %s (hh_id = %s)" % (
            trace_label,
            df.index.name,
            idx,
            df.household_id.loc[idx],
        )

        logger.warning(row_msg)


def _preprocess_departure_probs(
    choosers_df,
    choosers,
    probs_spec,
    probs_join_cols,
    clip_earliest_latest,
    depart_alt_base,
    first_trip_in_leg,
):

    # zero out probs outside earliest-latest window if one exists
    probs_cols = [c for c in probs_spec.columns if c not in probs_join_cols]
    if clip_earliest_latest:
        chooser_probs = _clip_probs(choosers_df, choosers[probs_cols], depart_alt_base)
    else:
        chooser_probs = choosers.loc[:, probs_cols]

    if first_trip_in_leg:
        # probs should sum to 1 unless all zero
        chooser_probs = chooser_probs.div(chooser_probs.sum(axis=1), axis=0).fillna(0)

    return chooser_probs


def _preprocess_stop_duration_probs(choosers):

    # convert wide to long. duration probs are stored in long format so that
    # inbound/outbound duration probs, which both have a 0 alternative, can be
    # stored in a single config file. open to better suggestions here.
    choosers["periods_left"] = choosers["latest"] - choosers["earliest"]
    choosers = choosers.query("periods_left_min <= periods_left <= periods_left_max")
    chooser_probs = choosers.reset_index().pivot(
        index="trip_id", columns="duration_offset", values="prob"
    )

    # re-order columns if using negative offsets bc pivot (above) will sort
    # columns in ascending order by default. but we need the zero-indexed
    # col to be the 0 offset alternative for both outbound (positive) and
    # inbound (negative) alts in order for the post-processing step to work.
    if all(chooser_probs.columns < 1):
        chooser_probs = chooser_probs.sort_index(axis=1, ascending=False)

    return chooser_probs


def _preprocess_scheduling_probs(
    scheduling_mode,
    choosers_df,
    choosers,
    probs_spec,
    probs_join_cols,
    clip_earliest_latest,
    depart_alt_base,
    first_trip_in_leg,
):
    """
    Preprocesses the choosers tables depending on the trip scheduling mode
    selected.
    """

    if scheduling_mode == "departure":
        chooser_probs = _preprocess_departure_probs(
            choosers_df,
            choosers,
            probs_spec,
            probs_join_cols,
            clip_earliest_latest,
            depart_alt_base,
            first_trip_in_leg,
        )
    elif scheduling_mode == "stop_duration":
        chooser_probs = _preprocess_stop_duration_probs(choosers)
    else:
        logger.error(
            "Invalid scheduling mode specified: {0}.".format(scheduling_mode),
            "Please select one of ['departure', 'stop_duration'] and try again.",
        )

    # probs should sum to 1 with residual probs resulting in choice of 'fail'
    chooser_probs["fail"] = 1 - chooser_probs.sum(axis=1).clip(0, 1)

    return chooser_probs


def _postprocess_scheduling_choices(
    scheduling_mode, depart_alt_base, choices, choice_cols, choosers
):

    """
    This method just converts the choice column indexes returned by the
    logit.make_choices() method into actual departure time values that are
    more useful for downstream models.
    """
    # convert alt choice index to depart time (setting failed choices to -1)
    failed = choices == choice_cols.get_loc("fail")

    # For the stop duration-based probabilities, the alternatives are offsets that
    # get applied to trip-specific departure and arrival times, so depart_alt_base
    # is a column/series rather than a scalar.
    if scheduling_mode == "stop_duration":

        # for outbound trips, offsets get added to the departure time constraint
        if choosers.outbound.all():
            depart_alt_base = choosers["earliest"]

        # for inbound trips, offsets get subtracted from tour end constraint
        elif not choosers.outbound.any():
            depart_alt_base = choosers["latest"]
            choices *= -1

        else:
            logger.error(
                "Outbound trips are being scheduled at the same "
                "time as inbound trips. That should never happen."
            )

    choices = (choices + depart_alt_base).where(~failed, -1)

    return choices, failed


def make_scheduling_choices(
    choosers_df,
    scheduling_mode,
    probs_spec,
    probs_join_cols,
    depart_alt_base,
    first_trip_in_leg,
    report_failed_trips,
    trace_hh_id,
    trace_label,
    trace_choice_col_name="depart",
    clip_earliest_latest=True,
):
    """
    We join each trip with the appropriate row in probs_spec by joining on probs_join_cols,
    which should exist in both trips, probs_spec dataframe.

    Parameters
    ----------
    choosers: pd.DataFrame
    scheduling_mode: str
        Either 'departure' or 'stop_duration' depending on whether the probability
        lookup table is keyed on depature period or stop duration.
    trips: pd.DataFrame
    probs_spec: pd.DataFrame
        Dataframe of probs for choice of depart times and join columns to match them with trips.
        Depart columns names are irrelevant. Instead, they are position dependent,
        time period choice is their index + depart_alt_base
    depart_alt_base: int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am
    report_failed_trips : bool
    trace_hh_id
    trace_label

    Returns
    -------
    choices: pd.Series
        time periods depart choices, one per trip (except for trips with zero probs)
    """

    choosers = pd.merge(
        choosers_df.reset_index(), probs_spec, on=probs_join_cols, how="left"
    ).set_index(choosers_df.index.name)
    chunk.log_df(trace_label, "choosers", choosers)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(choosers, "%s.choosers" % trace_label)

    # different pre-processing is required based on the scheduling mode
    chooser_probs = _preprocess_scheduling_probs(
        scheduling_mode,
        choosers_df,
        choosers,
        probs_spec,
        probs_join_cols,
        clip_earliest_latest,
        depart_alt_base,
        first_trip_in_leg,
    )

    chunk.log_df(trace_label, "chooser_probs", chooser_probs)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(chooser_probs, "%s.chooser_probs" % trace_label)

    raw_choices, rands = logit.make_choices(
        chooser_probs, trace_label=trace_label, trace_choosers=choosers
    )

    chunk.log_df(trace_label, "choices", raw_choices)
    chunk.log_df(trace_label, "rands", rands)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(
            raw_choices,
            "%s.choices" % trace_label,
            columns=[None, trace_choice_col_name],
        )
        tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])

    # different post-processing is required based on the scheduling mode
    choices, failed = _postprocess_scheduling_choices(
        scheduling_mode,
        depart_alt_base,
        raw_choices,
        chooser_probs.columns,
        choosers_df,
    )

    chunk.log_df(trace_label, "failed", failed)

    # report failed trips while we have the best diagnostic info
    if report_failed_trips and failed.any():
        _report_bad_choices(
            bad_row_map=failed,
            df=choosers,
            filename="failed_choosers",
            trace_label=trace_label,
            trace_choosers=None,
        )

    # trace before removing failures
    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, trace_choice_col_name]
        )
        tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])

    # remove any failed choices
    if failed.any():
        choices = choices[~failed]

    if all([check_col in choosers_df.columns for check_col in ["earliest", "latest"]]):
        assert (choices >= choosers_df.earliest[~failed]).all()
        assert (choices <= choosers_df.latest[~failed]).all()

    return choices
