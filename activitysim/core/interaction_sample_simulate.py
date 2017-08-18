# ActivitySim
# See full license in LICENSE.txt.

import os
import psutil
import gc

import logging

import numpy as np
import pandas as pd

from . import logit
from . import tracing
from .simulate import add_skims
from .simulate import chunked_choosers_and_alts
from .simulate import num_chunk_rows_for_chunk_size

from .interaction_simulate import eval_interaction_utilities

logger = logging.getLogger(__name__)

DUMP = False


def _interaction_sample_simulate(
        choosers, alternatives, spec, choice_column,
        skims, locals_d,
        trace_label=None, trace_choice_name=None):
    """
    Run a MNL simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    Parameters are same as for public function interaction_simulate

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
        same shape as utilities, one row per chooser and one column for alternative

    positions : series
        choices among alternatives with the chosen alternative represented
        as the integer index of the selected alternative column in probs

    choices : series
        series with the alternative chosen for each chooser
        the index is same as choosers
        and the series value is the alternative df index of chosen alternative

    Returns
    -------
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    assert len(choosers.index) == len(np.unique(alternatives.index.values))

    trace_label = tracing.extend_trace_label(trace_label, 'interaction_simulate')

    have_trace_targets = trace_label and tracing.has_trace_targets(choosers)

    if have_trace_targets:
        tracing.trace_df(choosers, tracing.extend_trace_label(trace_label, 'choosers'))
        tracing.trace_df(alternatives, tracing.extend_trace_label(trace_label, 'alternatives'),
                         slicer='NONE', transpose=False)

    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    if skims:
        alternatives[alternatives.index.name] = alternatives.index

    # in vanilla interaction_simulate interaction_df is cross join of choosers and alternatives
    # interaction_df = logit.interaction_dataset(choosers, alternatives, sample_size)
    # here, alternatives is sparsely repeated once for each (non-dup) sample
    # we expect alternatives to have same index of choosers (but with duplicate index values)
    # so we just need to left join alternatives with choosers
    assert alternatives.index.name == choosers.index.name

    interaction_df = pd.merge(
        alternatives, choosers,
        left_index=True, right_index=True,
        suffixes=('', '_r'))

    tracing.dump_df(DUMP, interaction_df, trace_label, 'interaction_df')

    if skims:
        add_skims(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of choosers match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in alternative
    if have_trace_targets:
        trace_rows, trace_ids = tracing.interaction_trace_rows(interaction_df, choosers)

        tracing.trace_df(interaction_df,
                         tracing.extend_trace_label(trace_label, 'interaction_df'),
                         transpose=False)
    else:
        trace_rows = trace_ids = None

    interaction_utilities, trace_eval_results \
        = eval_interaction_utilities(spec, interaction_df, locals_d, trace_label, trace_rows)

    tracing.dump_df(DUMP, interaction_utilities, trace_label, 'interaction_utilities')

    if have_trace_targets:
        tracing.trace_interaction_eval_results(trace_eval_results, trace_ids,
                                               tracing.extend_trace_label(trace_label, 'eval'))

        tracing.trace_df(interaction_utilities,
                         tracing.extend_trace_label(trace_label, 'interaction_utilities'),
                         transpose=False)

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    # interaction_utilities is sparse because duplicate sampled alternatives were dropped
    # so we need to pad with dummy utilities so low that they are never chosen

    # number of samples per chooser
    sample_counts = interaction_utilities.groupby(interaction_utilities.index).size().values

    # max number of alternatvies for any chooser
    max_sample_count = sample_counts.max()

    # offset of the last row of each chooser in sparse interaction_utilities
    row_offsets = np.insert(sample_counts.cumsum(), 0, 0)

    # repeat the row offsets once for each dummy utility to insert
    # (we want to insert dummy utilities at the END of the list of alternative utilities)
    inserts = np.repeat(row_offsets[1:], max_sample_count - sample_counts)

    # insert the zero-prob utilities to pad each alternative set to same size
    padded_utilities = np.insert(interaction_utilities.utility.values, inserts, -999)

    # reshape to array with one row per chooser, on column per alternative
    padded_utilities = padded_utilities.reshape(-1, max_sample_count)

    # convert to a dataframe with one row per chooser and one column per alternative
    utilities_df = pd.DataFrame(
        padded_utilities,
        index=choosers.index)

    # print "\nsample_counts\n", sample_counts
    # print "\nmax_sample_count\n", max_sample_count
    # print "\nlast_row_offsets\n", last_row_offsets
    # print "\ninserts\n", inserts
    # print "\nsparse utilities\n", interaction_utilities.utility.values
    # print "\npadded_utilities\n", padded_utilities
    # print "\nreshaped padded_utilities\n", padded_utilities

    tracing.dump_df(DUMP, utilities_df, trace_label, 'utilities_df')

    if have_trace_targets:
        tracing.trace_df(utilities_df, tracing.extend_trace_label(trace_label, 'utilities'),
                         column_labels=['alternative', 'utility'])

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities_df, trace_label=trace_label, trace_choosers=choosers)

    if have_trace_targets:
        tracing.trace_df(probs, tracing.extend_trace_label(trace_label, 'probs'),
                         column_labels=['alternative', 'probability'])

    tracing.dump_df(DUMP, probs, trace_label, 'probs')

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)

    # shouldn't have chosen any of the dummy pad utilities
    assert positions.max() < max_sample_count

    # need to get from an integer offset into the alternative sample to the alternative index
    # that is, we want the index value of the row that is offset by <position> rows into the
    # tranche of this choosers alternatives created by cross join of alternatives and choosers

    # first_row_offsets is the offset into interaction_df df of first row of chooser alternatives
    first_row_offsets = row_offsets[:-1]

    # resulting pandas Int64Index has one element per chooser row and is in same order as choosers
    choices = interaction_df[choice_column].take(positions + first_row_offsets)

    # create a series with index from choosers and the index of the chosen alternative
    choices = pd.Series(choices, index=choosers.index)

    tracing.dump_df(DUMP, choices, trace_label, 'choices')

    if have_trace_targets:
        tracing.trace_df(choices, tracing.extend_trace_label(trace_label, 'choices'),
                         columns=[None, trace_choice_name])
        tracing.trace_df(rands, tracing.extend_trace_label(trace_label, 'rands'),
                         columns=[None, 'rand'])

    return choices


def interaction_sample_simulate(
        choosers, alternatives, spec, choice_column=None,
        skims=None, locals_d=None, chunk_size=0,
        trace_label=None, trace_choice_name=None):

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
        DataFrame of alternatives - will be merged with choosers
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
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
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    choices : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    rows_per_chunk = num_chunk_rows_for_chunk_size(chunk_size, choosers, alternatives)

    logger.info("interaction_simulate chunk_size %s num_choosers %s"
                % (chunk_size, len(choosers.index)))

    result_list = []
    for i, num_chunks, chooser_chunk, alternative_chunk \
            in chunked_choosers_and_alts(choosers, alternatives, rows_per_chunk):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(chooser_chunk)))

        choices = _interaction_sample_simulate(
            chooser_chunk, alternative_chunk, spec, choice_column,
            skims, locals_d,
            tracing.extend_trace_label(trace_label, 'chunk_%s' % i), trace_choice_name)

        result_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices
