# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import chunk
from activitysim.core import logit
from activitysim.core import inject

from activitysim.core.mem import force_garbage_collect

logger = logging.getLogger(__name__)


def _clip_probs(choosers_df, probs, model_settings):
    """
    zero out probs before chooser.earliest or after chooser.latest

    Parameters
    ----------
    trips: pd.DataFrame
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

    depart_alt_base = model_settings.get(DEPART_ALT_BASE)

    # there should be one row in probs per trip
    assert choosers_df.shape[0] == probs.shape[0]

    # probs should sum to 1 across rows before clipping
    probs = probs.div(probs.sum(axis=1), axis=0)

    num_rows, num_cols = probs.shape
    ix_map = np.tile(np.arange(0, num_cols), num_rows).reshape(num_rows, num_cols) + depart_alt_base
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...

    clip_mask = ((ix_map >= choosers_df.earliest.values.reshape(num_rows, 1)) &
                 (ix_map <= choosers_df.latest.values.reshape(num_rows, 1))) * 1
    #  [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0]
    #  [0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]...

    probs = probs*clip_mask

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
    df['household_id'] = hh_ids

    filename = "%s.%s" % (trace_label, filename)

    logger.info("dumping %s" % filename)
    tracing.write_csv(df, file_name=filename, transpose=False)

    # log the indexes of the first MAX_PRINT offending rows
    MAX_PRINT = 0
    for idx in df.index[:MAX_PRINT].values:

        row_msg = "%s : failed %s = %s (hh_id = %s)" % \
                  (trace_label, df.index.name, idx, df.household_id.loc[idx])

        logger.warning(row_msg)


def calc_row_size(choosers, spec, probs_join_cols, trace_label, chooser_type):

    sizer = chunk.RowSizeEstimator(trace_label)

    # NOTE we chunk chunk_id
    # scale row_size by average number of chooser rows per chunk_id
    num_choosers = choosers['chunk_id'].max() + 1
    rows_per_chunk_id = len(choosers) / num_choosers

    if chooser_type == 'trip':
        # only non-initial trips require scheduling, segment handing first such trip in tour will use most space
        outbound_chooser = (choosers.trip_num == 2) & choosers.outbound & (choosers.primary_purpose != 'atwork')
        inbound_chooser = (choosers.trip_num == choosers.trip_count-1) & ~choosers.outbound & (choosers.primary_purpose != 'atwork')

        # furthermore, inbound and outbound are scheduled independently
        if outbound_chooser.sum() > inbound_chooser.sum():
            is_chooser = outbound_chooser
            logger.debug(f"{trace_label} {is_chooser.sum()} outbound_choosers of {len(choosers)} require scheduling")
        else:
            is_chooser = inbound_chooser
            logger.debug(f"{trace_label} {is_chooser.sum()} inbound_choosers of {len(choosers)} require scheduling")

        chooser_fraction = is_chooser.sum()/len(choosers)
    else:
        chooser_fraction = 1
    logger.debug(f"{trace_label} chooser_fraction {chooser_fraction *100}%")

    chooser_row_size = len(choosers.columns) + len(spec.columns) - len(probs_join_cols)
    sizer.add_elements(chooser_fraction * chooser_row_size, 'choosers')

    # might be clipped to fewer but this is worst case
    chooser_probs_row_size = len(spec.columns) - len(probs_join_cols)
    sizer.add_elements(chooser_fraction * chooser_probs_row_size, 'chooser_probs')

    sizer.add_elements(chooser_fraction, 'choices')
    sizer.add_elements(chooser_fraction, 'rands')
    sizer.add_elements(chooser_fraction, 'failed')

    row_size = sizer.get_hwm()
    row_size = row_size * rows_per_chunk_id

    return row_size


def probabilistic_scheduling(
        choosers_df, probs_spec, probs_join_cols, model_settings,
        trace_label, trace_hh_id, trace_choice_col_name,
        report_failed_trips=True,
        clip_earliest_latest=False, first_trip_in_leg=False):

    if model_settings.get('DEPART_ALT_BASE'):
        depart_alt_base = depart_alt_base
    else:
        depart_alt_base = 0

    choosers = pd.merge(choosers_df.reset_index(), probs_spec, on=probs_join_cols,
                        how='left').set_index(choosers_df.index.name)
    chunk.log_df(trace_label, "choosers", choosers)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(choosers, '%s.choosers' % trace_label)

    # choosers should now match choosers_df row for row
    assert choosers.index.is_unique
    assert len(choosers.index) == len(choosers_df.index)

    # zero out probs outside earliest-latest window if one exists
    probs_cols = [c for c in probs_spec.columns if c not in probs_join_cols]
    if clip_earliest_latest:
        chooser_probs = _clip_probs(choosers_df, choosers[probs_cols], model_settings)
    else:
        chooser_probs = choosers.loc[:, probs_cols]
    chunk.log_df(trace_label, "chooser_probs", chooser_probs)

    if first_trip_in_leg:
        # probs should sum to 1 unless all zero
        chooser_probs = chooser_probs.div(chooser_probs.sum(axis=1), axis=0).fillna(0)

    # probs should sum to 1 with residual probs resulting in choice of 'fail'
    chooser_probs['fail'] = 1 - chooser_probs.sum(axis=1).clip(0, 1)
    chunk.log_df(trace_label, "chooser_probs", chooser_probs)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(chooser_probs, '%s.chooser_probs' % trace_label)
    
    choices, rands = logit.make_choices(chooser_probs, trace_label=trace_label, trace_choosers=choosers)

    chunk.log_df(trace_label, "choices", choices)
    chunk.log_df(trace_label, "rands", rands)

    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, trace_choice_col_name])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    # convert alt choice index to depart time (setting failed choices to -1)
    failed = (choices == chooser_probs.columns.get_loc('fail'))
    choices = (choices + depart_alt_base).where(~failed, -1)

    chunk.log_df(trace_label, "failed", failed)

    # report failed trips while we have the best diagnostic info
    if report_failed_trips and failed.any():
        report_bad_choices(
            bad_row_map=failed,
            df=choosers,
            filename='failed_choosers',
            trace_label=trace_label,
            trace_choosers=None)

    # trace before removing failures
    if trace_hh_id and tracing.has_trace_targets(choosers_df):
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, trace_choice_col_name])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    # remove any failed choices
    if failed.any():
        choices = choices[~failed]

    return choices
