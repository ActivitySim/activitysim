# ActivitySim
# See full license in LICENSE.txt.

import os
import psutil
import gc

import logging

import numpy as np
import pandas as pd

from activitysim.core.util import quick_loc_series

from . import logit
from . import tracing
from .simulate import add_skims
from .interaction_simulate import eval_interaction_utilities

logger = logging.getLogger(__name__)

DUMP = False


def _interaction_sample(
        choosers, alternatives, spec, sample_size, alt_col_name,
        skims=None, locals_d=None,
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

    trace_label = tracing.extend_trace_label(trace_label, 'interaction_simulate')
    have_trace_targets = trace_label and tracing.has_trace_targets(choosers)

    if have_trace_targets:
        tracing.trace_df(choosers, tracing.extend_trace_label(trace_label, 'choosers'))
        tracing.trace_df(alternatives, tracing.extend_trace_label(trace_label, 'alternatives'),
                         slicer='NONE', transpose=False)

    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    alternative_count = len(alternatives)
    logger.debug("_interaction_sample alternative_count %s" % alternative_count)

    if alternative_count > len(alternatives):
        logger.debug("clipping alternative_count %s to len(alternatives) %s" %
                     (alternative_count, len(alternatives)))
        alternative_count = min(alternative_count, len(alternatives))

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    if skims:
        alternatives[alternatives.index.name] = alternatives.index

    # cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    interaction_df = logit.interaction_dataset(choosers, alternatives, alternative_count)

    if skims:
        add_skims(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of interaction_df match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in interaction_df
    if have_trace_targets:
        trace_rows, trace_ids = tracing.interaction_trace_rows(interaction_df, choosers)

        tracing.trace_df(interaction_df[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_df'),
                         slicer='NONE', transpose=False)
    else:
        trace_rows = trace_ids = None

    interaction_utilities, trace_eval_results \
        = eval_interaction_utilities(spec, interaction_df, locals_d, trace_label, trace_rows)

    if have_trace_targets:
        tracing.trace_interaction_eval_results(trace_eval_results, trace_ids,
                                               tracing.extend_trace_label(trace_label, 'eval'))

        tracing.trace_df(interaction_utilities[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_utilities'),
                         slicer='NONE', transpose=False)

    # FIXME - do this in numpy, not pandas?
    # reshape utilities (one utility column and one row per row in interaction_utilities)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.as_matrix().reshape(len(choosers), alternative_count),
        index=choosers.index)

    if have_trace_targets:
        tracing.trace_df(utilities, tracing.extend_trace_label(trace_label, 'utilities'),
                         column_labels=['alternative', 'utility'])

    tracing.dump_df(DUMP, utilities, trace_label, 'utilities')

    # FIXME - do this in numpy, not pandas?
    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)

    if have_trace_targets:
        tracing.trace_df(probs, tracing.extend_trace_label(trace_label, 'probs'),
                         column_labels=['alternative', 'probability'])

    tracing.dump_df(DUMP, probs, trace_label, 'probs')

    if have_trace_targets:
        # FIXME - this is wasteful of memory
        choices_df = pd.DataFrame()
        rands_df = pd.DataFrame()

    choices_array = np.empty([sample_size, len(choosers)]).astype(int)

    # FIXME - do this all at once rather than iterate?
    for i in range(sample_size):

        # FIXME - do this in numpy, not pandas?
        # make choices
        # positions is series with the chosen alternative represented as a column index in probs
        # which is an integer between zero and num alternatives in the alternative sample
        positions, rands = logit.make_choices(probs, trace_label=trace_label,
                                              trace_choosers=choosers)

        # need to get from an integer offset into the alternative sample to the alternative index
        # that is, we want the index value of the row that is offset by <position> rows into the
        # tranche of this choosers alternatives created by cross join of alternatives and choosers

        # offsets is the offset into model_design df of first row of chooser alternatives
        offsets = np.arange(len(positions)) * alternative_count
        # resulting pandas Int64Index has one element per chooser and is in same order as choosers
        choices = interaction_utilities.index.take(positions + offsets)

        choices_array[i] = choices

        if have_trace_targets:
            # FIXME - this is wasteful of memory
            choices_df[i] = pd.Series(choices, index=choosers.index)
            rands_df[i] = rands

    if have_trace_targets:
        # FIXME - this is wasteful of memory
        tracing.trace_df(choices_df,
                         tracing.extend_trace_label(trace_label, 'sampled_alternatives'),
                         column_labels=['sample_alt', 'alternative'])

        tracing.trace_df(rands_df, tracing.extend_trace_label(trace_label, 'sample_rands'),
                         column_labels=['sample_alt', 'rand'])
        del choices_df
        del rands_df

    # explode to one row per chooser.index, alt_TAZ
    if alt_col_name is None:
        alt_col_name = 'alt_%s' % alternatives.index.name
    choices_df = pd.DataFrame(
        {alt_col_name: choices_array.flatten(order='F'),
         choosers.index.name: np.repeat(np.asanyarray(choosers.index), sample_size)
         })

    # dataframe with one column of pick_counts for each chooser,alt with duplicate rows conflated
    # (couldn't think of any good way to get pick_counts without grouping)
    pick_counts = choices_df\
        .groupby([choosers.index.name, alt_col_name])\
        .size().to_frame('pick_count')

    # annotate with pick_count
    choices_df = \
        pd.merge(choices_df, pick_counts,
                 left_on=[choosers.index.name, alt_col_name],
                 right_index=True,
                 how="left").set_index(choosers.index.name)

    tracing.dump_df(DUMP, choices_df, trace_label, 'choices_df')

    return choices_df


def chunked_choosers(choosers, chunk_size):
    # generator to iterate over chooses in chunk_size chunks
    chunk_size = int(chunk_size)
    num_choosers = len(choosers.index)

    i = offset = 0
    while offset < num_choosers:
        yield i, choosers[offset: offset+chunk_size]
        offset += chunk_size
        i += 1


def interaction_sample(
        choosers, alternatives, spec, sample_size,
        alt_col_name=None,
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
        DataFrame of alternatives - will be merged with choosers and sampled
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
    sample_size : int
        desired number of alternatives per chooser
    alt_col_name: str or None
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
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.
    chunk_size : int
        if chunk_size > 0 iterates over choosers in chunk_size chunks
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    # FIXME - chunk size should take number of chooser and alternative columns into account
    # FIXME - that is, chunk size should represent memory footprint (rows X columns) not just rows

    chunk_size = int(chunk_size)

    if (chunk_size == 0) or (chunk_size >= len(choosers.index)):
        choices = _interaction_sample(choosers, alternatives, spec, sample_size, alt_col_name,
                                      skims, locals_d,
                                      trace_label, trace_choice_name)
        return choices

    logger.info("interaction_simulate chunk_size %s num_choosers %s" %
                (chunk_size, len(choosers.index)))

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, chooser_chunk in chunked_choosers(choosers, chunk_size):

        logger.info("Running chunk %s of size %d" % (i, len(chooser_chunk)))

        choices = _interaction_sample(chooser_chunk, alternatives, spec, sample_size, alt_col_name,
                                      skims, locals_d,
                                      tracing.extend_trace_label(trace_label, 'chunk_%s' % i),
                                      trace_choice_name)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    assert len(choices.index == len(choosers.index))

    return choices
