# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402


from builtins import range

import logging

from math import ceil
import numpy as np
import pandas as pd

from activitysim.core.mem import force_garbage_collect

from . import logit
from . import tracing
from . import chunk
from .simulate import set_skim_wrapper_targets


from .interaction_simulate import eval_interaction_utilities
from . import pipeline

logger = logging.getLogger(__name__)

DUMP = False


def make_sample_choices(
        choosers, probs,
        alternatives,
        sample_size, alternative_count, alt_col_name,
        allow_zero_probs,
        trace_label):
    """

    Parameters
    ----------
    choosers
    probs : pandas DataFrame
        one row per chooser and one column per alternative
    alternatives
        dataframe with index containing alt ids
    sample_size : int
        number of samples/choices to make
    alternative_count
    alt_col_name : str
    trace_label

    Returns
    -------

    """

    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (len(choosers), alternative_count)

    assert isinstance(alternatives, pd.DataFrame)
    assert len(alternatives) == alternative_count

    if allow_zero_probs:
        zero_probs = (probs.sum(axis=1) == 0)
        if zero_probs.all():
            return pd.DataFrame(columns=[alt_col_name, 'rand', 'prob', choosers.index.name])
        if zero_probs.any():
            # remove from sample
            probs = probs[~zero_probs]
            choosers = choosers[~zero_probs]

    cum_probs_arr = probs.values.cumsum(axis=1)

    # alt probs in convenient layout to return prob of chose alternative
    # (same layout as cum_probs_arr)
    alt_probs_array = probs.values.flatten()

    # get sample_size rands for each chooser
    # transform as we iterate over alternatives
    # reshape so rands[i] is in broadcastable (2-D) shape for cum_probs_arr
    # i.e rands[i] is a 2-D array of one alt choice rand for each chooser
    rands = pipeline.get_rn_generator().random_for_df(probs, n=sample_size)
    rands = rands.T.reshape(sample_size, -1, 1)

    # the alternative value chosen
    choices_array = np.empty([sample_size, len(choosers)]).astype(int)

    # the probability of the chosen alternative
    choice_probs_array = np.empty([sample_size, len(choosers)])

    alts = np.tile(alternatives.index.values, len(choosers))

    # FIXME - do this all at once rather than iterate?
    for i in range(sample_size):

        # FIXME - do this in numpy, not pandas?

        # rands for this alt in broadcastable shape
        r = rands[i]

        # position of first occurrence of positive value
        positions = np.argmax(cum_probs_arr > r, axis=1)

        # FIXME - leave positions as numpy array, not pandas series?
        # positions is series with the chosen alternative represented as a column index in probs
        # which is an integer between zero and num alternatives in the alternative sample
        positions = pd.Series(positions, index=probs.index)

        # need to get from an integer offset into the alternative sample to the alternative index
        # that is, we want the index value of the row that is offset by <position> rows into the
        # tranche of this choosers alternatives created by cross join of alternatives and choosers

        # offsets is the offset into model_design df of first row of chooser alternatives
        offsets = np.arange(len(positions)) * alternative_count

        # choices and choice_probs have one element per chooser and is in same order as choosers
        choices_array[i] = np.take(alts, positions + offsets)
        choice_probs_array[i] = np.take(alt_probs_array, positions + offsets)

    # explode to one row per chooser.index, alt_TAZ
    choices_df = pd.DataFrame(
        {alt_col_name: choices_array.flatten(order='F'),
         'rand': rands.flatten(order='F'),
         'prob': choice_probs_array.flatten(order='F'),
         choosers.index.name: np.repeat(np.asanyarray(choosers.index), sample_size)
         })

    return choices_df


def _interaction_sample(
        choosers, alternatives,
        spec, sample_size, alt_col_name, allow_zero_probs,
        skims=None, locals_d=None,
        trace_label=None):
    """
    Run a MNL simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    Parameters are same as for public function interaction_sa,ple

    spec : dataframe
        one row per spec expression and one col with utility coefficient

    interaction_df : dataframe
        cross join (cartesian product) of choosers with alternatives
        combines columns of choosers and alternatives
        len(df) == len(choosers) * len(alternatives)
        index values (non-unique) are index values from alternatives df

    interaction_utilities : dataframe
        the utility of each alternative is sum of the partial utilities determined by the
        various spec expressions and their corresponding coefficients
        yielding a dataframe  with len(interaction_df) rows and one utility column
        having the same index as interaction_df (non-unique values from alternatives df)

    utilities : dataframe
        dot product of model_design.dot(spec)
        yields utility value for element in the cross product of choosers and alternatives
        this is then reshaped as a dataframe with one row per chooser and one column per alternative

    probs : dataframe
        utilities exponentiated and converted to probabilities
        same shape as utilities, one row per chooser and one column per alternative

    positions : series
        choices among alternatives with the chosen alternative represented
        as the integer index of the selected alternative column in probs

    choices : series
        series with the alternative chosen for each chooser
        the index is same as choosers
        and the series value is the alternative df index of chosen alternative

    Returns
    -------
    choices_df : pandas.DataFrame

        A DataFrame where index should match the index of the choosers DataFrame
        and columns alt_col_name, prob, rand, pick_count

        prob: float
            the probability of the chosen alternative
        rand: float
            the rand that did the choosing
        pick_count : int
            number of duplicate picks for chooser, alt
    """

    have_trace_targets = tracing.has_trace_targets(choosers)

    assert len(choosers.index) > 0

    if have_trace_targets:
        tracing.trace_df(choosers, tracing.extend_trace_label(trace_label, 'choosers'))
        tracing.trace_df(alternatives, tracing.extend_trace_label(trace_label, 'alternatives'),
                         slicer='NONE', transpose=False)

    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    if skims is not None:
        alternatives[alternatives.index.name] = alternatives.index

    # - cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    alternative_count = alternatives.shape[0]
    interaction_df = \
        logit.interaction_dataset(choosers, alternatives, sample_size=alternative_count)

    chunk.log_df(trace_label, 'interaction_df', interaction_df)

    assert alternative_count == len(interaction_df.index) / len(choosers.index)

    if skims is not None:
        set_skim_wrapper_targets(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of interaction_df match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in interaction_df
    if have_trace_targets:
        trace_rows, trace_ids \
            = tracing.interaction_trace_rows(interaction_df, choosers, alternative_count)

        tracing.trace_df(interaction_df[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_df'),
                         slicer='NONE', transpose=False)
    else:
        trace_rows = trace_ids = None

    # interaction_utilities is a df with one utility column and one row per interaction_df row
    interaction_utilities, trace_eval_results \
        = eval_interaction_utilities(spec, interaction_df, locals_d, trace_label, trace_rows)
    chunk.log_df(trace_label, 'interaction_utilities', interaction_utilities)

    del interaction_df
    chunk.log_df(trace_label, 'interaction_df', None)

    if have_trace_targets:
        tracing.trace_interaction_eval_results(trace_eval_results, trace_ids,
                                               tracing.extend_trace_label(trace_label, 'eval'))

        tracing.trace_df(interaction_utilities[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_utilities'),
                         slicer='NONE', transpose=False)

    tracing.dump_df(DUMP, interaction_utilities, trace_label, 'interaction_utilities')

    # reshape utilities (one utility column and one row per row in interaction_utilities)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.values.reshape(len(choosers), alternative_count),
        index=choosers.index)
    chunk.log_df(trace_label, 'utilities', utilities)

    del interaction_utilities
    chunk.log_df(trace_label, 'interaction_utilities', None)

    if have_trace_targets:
        tracing.trace_df(utilities, tracing.extend_trace_label(trace_label, 'utilities'),
                         column_labels=['alternative', 'utility'])

    tracing.dump_df(DUMP, utilities, trace_label, 'utilities')

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, allow_zero_probs=allow_zero_probs,
                                 trace_label=trace_label, trace_choosers=choosers)
    chunk.log_df(trace_label, 'probs', probs)

    del utilities
    chunk.log_df(trace_label, 'utilities', None)

    if have_trace_targets:
        tracing.trace_df(probs, tracing.extend_trace_label(trace_label, 'probs'),
                         column_labels=['alternative', 'probability'])

    choices_df = make_sample_choices(
        choosers, probs, alternatives,
        sample_size, alternative_count, alt_col_name,
        allow_zero_probs=allow_zero_probs,
        trace_label=trace_label)

    chunk.log_df(trace_label, 'choices_df', choices_df)

    del probs
    chunk.log_df(trace_label, 'probs', None)

    # make_sample_choices should return choosers index as choices_df column
    assert choosers.index.name in choices_df.columns

    # pick_count and pick_dup
    # pick_count is number of duplicate picks
    # pick_dup flag is True for all but first of duplicates
    pick_group = choices_df.groupby([choosers.index.name, alt_col_name])

    # number each item in each group from 0 to the length of that group - 1.
    choices_df['pick_count'] = pick_group.cumcount(ascending=True)
    # flag duplicate rows after first
    choices_df['pick_dup'] = choices_df['pick_count'] > 0
    # add reverse cumcount to get total pick_count (conveniently faster than groupby.count + merge)
    choices_df['pick_count'] += pick_group.cumcount(ascending=False) + 1

    # drop the duplicates
    choices_df = choices_df[~choices_df['pick_dup']]
    del choices_df['pick_dup']
    chunk.log_df(trace_label, 'choices_df', choices_df)

    # set index after groupby so we can trace on it
    choices_df.set_index(choosers.index.name, inplace=True)

    tracing.dump_df(DUMP, choices_df, trace_label, 'choices_df')

    if have_trace_targets:
        tracing.trace_df(choices_df,
                         tracing.extend_trace_label(trace_label, 'sampled_alternatives'),
                         transpose=False,
                         column_labels=['sample_alt', 'alternative'])

    # don't need this after tracing
    del choices_df['rand']
    chunk.log_df(trace_label, 'choices_df', choices_df)

    # - NARROW
    choices_df['prob'] = choices_df['prob'].astype(np.float32)
    assert (choices_df['pick_count'].max() < 4294967295) or (choices_df.empty)
    choices_df['pick_count'] = choices_df['pick_count'].astype(np.uint32)

    return choices_df


def calc_rows_per_chunk(chunk_size, choosers, alternatives, trace_label):

    num_choosers = choosers.shape[0]

    # if not chunking, then return num_choosers
    # if chunk_size == 0:
    #     return num_choosers, 0

    # all columns from choosers
    chooser_row_size = choosers.shape[1]

    # interaction_df has one column per alternative plus a skim column and a join column
    alt_row_size = alternatives.shape[1] + 2

    # interaction_utilities
    alt_row_size += 1

    # interaction_df includes all alternatives and is only afterwards sampled
    row_size = (chooser_row_size + alt_row_size) * alternatives.shape[0]

    # utilities and probs have one row per chooser and one column per alternative row
    row_size += 2 * alternatives.shape[0]

    # logger.debug("%s #chunk_calc choosers %s" % (trace_label, choosers.shape))
    # logger.debug("%s #chunk_calc alternatives %s" % (trace_label, alternatives.shape))
    # logger.debug("%s #chunk_calc alt_row_size %s" % (trace_label, alt_row_size))

    return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def interaction_sample(
        choosers, alternatives, spec, sample_size,
        alt_col_name, allow_zero_probs=False,
        skims=None, locals_d=None, chunk_size=0,
        trace_label=None):

    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    optionally (if chunk_size > 0) iterates over choosers in chunk_size chunks

    Parameters
    ----------
    choosers : pandas.DataFrame
        DataFrame of choosers
    alternatives : pandas.DataFrame
        DataFrame of alternatives - will be merged with choosers and sampled
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.
    alt_col_name: str
        name to give the sampled_alternative column
    skims : Skims object
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    chunk_size : int
        if chunk_size > 0 iterates over choosers in chunk_size chunks
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.

    Returns
    -------
    choices_df : pandas.DataFrame

        A DataFrame where index should match the index of the choosers DataFrame
        (except with sample_size rows for each choser row, one row for each alt sample)
        and columns alt_col_name, prob, rand, pick_count

        <alt_col_name>:
            alt identifier from alternatives[<alt_col_name>
        prob: float
            the probability of the chosen alternative
        pick_count : int
            number of duplicate picks for chooser, alt
    """

    trace_label = tracing.extend_trace_label(trace_label, 'interaction_sample')

    # we return alternatives ordered in (index, alt_col_name)
    # if choosers index is not ordered, it is probably a mistake, since the alts wont line up
    assert alt_col_name is not None
    assert choosers.index.is_monotonic

    assert sample_size > 0
    sample_size = min(sample_size, len(alternatives.index))

    rows_per_chunk, effective_chunk_size = \
        calc_rows_per_chunk(chunk_size, choosers, alternatives, trace_label)

    result_list = []
    for i, num_chunks, chooser_chunk in chunk.chunked_choosers(choosers, rows_per_chunk):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(chooser_chunk)))

        chunk_trace_label = tracing.extend_trace_label(trace_label, 'chunk_%s' % i) \
            if num_chunks > 1 else trace_label

        chunk.log_open(chunk_trace_label, chunk_size, effective_chunk_size)

        choices = _interaction_sample(chooser_chunk, alternatives,
                                      spec, sample_size, alt_col_name, allow_zero_probs,
                                      skims, locals_d,
                                      chunk_trace_label)

        chunk.log_close(chunk_trace_label)

        if choices.shape[0] > 0:
            # might not be any if allow_zero_probs
            result_list.append(choices)

        force_garbage_collect()

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert allow_zero_probs or (len(choosers.index) == len(np.unique(choices.index.values)))

    # keep alts in canonical order so choices based on their probs are stable across runs
    choices = choices.sort_values(by=alt_col_name).sort_index(kind='mergesort')

    return choices
