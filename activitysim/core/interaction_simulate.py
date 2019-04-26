# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import zip

import logging

import numpy as np
import pandas as pd
from collections import OrderedDict

from . import logit
from . import tracing
from . import config
from .simulate import set_skim_wrapper_targets
from . import chunk
from . import mem

from . import assign

from activitysim.core.mem import force_garbage_collect


logger = logging.getLogger(__name__)

DUMP = False


def eval_interaction_utilities(spec, df, locals_d, trace_label, trace_rows):
    """
    Compute the utilities for a single-alternative spec evaluated in the context of df

    We could compute the utilities for interaction datasets just as we do for simple_simulate
    specs with multiple alternative columns byt calling eval_variables and then computing the
    utilities by matrix-multiplication of eval results with the utility coefficients in the
    spec alternative columns.

    But interaction simulate computes the utilities of each alternative in the context of a
    separate row in interaction dataset df, and so there is only one alternative in spec.
    This turns out to be quite a bit faster (in this special case) than the pandas dot function.

    For efficiency, we combine eval_variables and multiplication of coefficients into a single step,
    so we don't have to create a separate column for each partial utility. Instead, we simply
    multiply the eval result by a single alternative coefficient and sum the partial utilities.


    spec : dataframe
        one row per spec expression and one col with utility coefficient

    df : dataframe
        cross join (cartesian product) of choosers with alternatives
        combines columns of choosers and alternatives
        len(df) == len(choosers) * len(alternatives)
        index values (non-unique) are index values from alternatives df

    interaction_utilities : dataframe
        the utility of each alternative is sum of the partial utilities determined by the
        various spec expressions and their corresponding coefficients
        yielding a dataframe  with len(interaction_df) rows and one utility column
        having the same index as interaction_df (non-unique values from alternatives df)

    Returns
    -------
    utilities : pandas.DataFrame
        Will have the index of `df` and a single column of utilities

    """
    trace_label = tracing.extend_trace_label(trace_label, "eval_interaction_utilities")
    logger.info("Running eval_interaction_utilities on %s rows" % df.shape[0])

    assert(len(spec.columns) == 1)

    # avoid altering caller's passed-in locals_d parameter (they may be looping)
    locals_d = locals_d.copy() if locals_d is not None else {}
    locals_d.update(locals())

    def to_series(x):
        if np.isscalar(x):
            return pd.Series([x] * len(df), index=df.index)
        return x

    if trace_rows is not None and trace_rows.any():
        # # convert to numpy array so we can slice ndarrays as well as series
        # trace_rows = np.asanyarray(trace_rows)
        assert type(trace_rows) == np.ndarray
        trace_eval_results = OrderedDict()
    else:
        trace_eval_results = None

    check_for_variability = config.setting('check_for_variability')

    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously

    utilities = pd.DataFrame({'utility': 0.0}, index=df.index)
    no_variability = has_missing_vals = 0

    for expr, coefficient in zip(spec.index, spec.iloc[:, 0]):
        try:

            # - allow temps of form _od_DIST@od_skim['DIST']
            if expr.startswith('_'):
                target = expr[:expr.index('@')]
                rhs = expr[expr.index('@') + 1:]
                v = to_series(eval(rhs, globals(), locals_d))

                # update locals to allows us to ref previously assigned targets
                locals_d[target] = v

                if trace_eval_results is not None:
                    trace_eval_results[expr] = v[trace_rows]

                # mem.trace_memory_info("eval_interaction_utilities TEMP: %s" % expr)
                continue

            if expr.startswith('@'):
                v = to_series(eval(expr[1:], globals(), locals_d))
            else:
                v = df.eval(expr)

            if check_for_variability and v.std() == 0:
                logger.info("%s: no variability (%s) in: %s" % (trace_label, v.iloc[0], expr))
                no_variability += 1

            # FIXME - how likely is this to happen? Not sure it is really a problem?
            if check_for_variability and np.count_nonzero(v.isnull().values) > 0:
                logger.info("%s: missing values in: %s" % (trace_label, expr))
                has_missing_vals += 1

            utilities.utility += (v * coefficient).astype('float')

            if trace_eval_results is not None:

                # expressions should have been uniquified when spec was read
                # (though we could do it here if need be...)
                # expr = assign.uniquify_key(trace_eval_results, expr, template="{} # ({})")
                assert expr not in trace_eval_results

                trace_eval_results[expr] = v[trace_rows]
                k = 'partial utility (coefficient = %s) for %s' % (coefficient, expr)
                trace_eval_results[k] = v[trace_rows] * coefficient

        except Exception as err:
            logger.exception("Variable evaluation failed for: %s" % str(expr))
            raise err

        # mem.trace_memory_info("eval_interaction_utilities: %s" % expr)

    if no_variability > 0:
        logger.warning("%s: %s columns have no variability" % (trace_label, no_variability))

    if has_missing_vals > 0:
        logger.warning("%s: %s columns have missing values" % (trace_label, has_missing_vals))

    if trace_eval_results is not None:
        trace_eval_results['total utility'] = utilities.utility[trace_rows]

        trace_eval_results = pd.DataFrame.from_dict(trace_eval_results)
        trace_eval_results.index = df[trace_rows].index

        # add df columns to trace_results
        trace_eval_results = pd.concat([df[trace_rows], trace_eval_results], axis=1)

    return utilities, trace_eval_results


def _interaction_simulate(
        choosers, alternatives, spec,
        skims=None, locals_d=None, sample_size=None,
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
    have_trace_targets = tracing.has_trace_targets(choosers)

    if have_trace_targets:
        tracing.trace_df(choosers, tracing.extend_trace_label(trace_label, 'choosers'))
        tracing.trace_df(alternatives, tracing.extend_trace_label(trace_label, 'alternatives'),
                         slicer='NONE', transpose=False)

    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    sample_size = sample_size or len(alternatives)

    if sample_size > len(alternatives):
        logger.debug("clipping sample size %s to len(alternatives) %s" %
                     (sample_size, len(alternatives)))
        sample_size = min(sample_size, len(alternatives))

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    if skims is not None:
        alternatives[alternatives.index.name] = alternatives.index

    # cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    interaction_df = logit.interaction_dataset(choosers, alternatives, sample_size)
    chunk.log_df(trace_label, 'interaction_df', interaction_df)

    if skims is not None:
        set_skim_wrapper_targets(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of model_design match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in model_design
    if have_trace_targets:
        trace_rows, trace_ids \
            = tracing.interaction_trace_rows(interaction_df, choosers, sample_size)

        tracing.trace_df(interaction_df[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_df'),
                         slicer='NONE', transpose=False)
    else:
        trace_rows = trace_ids = None

    interaction_utilities, trace_eval_results \
        = eval_interaction_utilities(spec, interaction_df, locals_d, trace_label, trace_rows)
    chunk.log_df(trace_label, 'interaction_utilities', interaction_utilities)

    if have_trace_targets:
        tracing.trace_interaction_eval_results(trace_eval_results, trace_ids,
                                               tracing.extend_trace_label(trace_label, 'eval'))

        tracing.trace_df(interaction_utilities[trace_rows],
                         tracing.extend_trace_label(trace_label, 'interaction_utilities'),
                         slicer='NONE', transpose=False)

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.values.reshape(len(choosers), sample_size),
        index=choosers.index)
    chunk.log_df(trace_label, 'utilities', utilities)

    if have_trace_targets:
        tracing.trace_df(utilities, tracing.extend_trace_label(trace_label, 'utilities'),
                         column_labels=['alternative', 'utility'])

    tracing.dump_df(DUMP, utilities, trace_label, 'utilities')

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)
    chunk.log_df(trace_label, 'probs', probs)

    if have_trace_targets:
        tracing.trace_df(probs, tracing.extend_trace_label(trace_label, 'probs'),
                         column_labels=['alternative', 'probability'])

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = \
        logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)
    chunk.log_df(trace_label, 'positions', positions)
    chunk.log_df(trace_label, 'rands', rands)

    # need to get from an integer offset into the alternative sample to the alternative index
    # that is, we want the index value of the row that is offset by <position> rows into the
    # tranche of this choosers alternatives created by cross join of alternatives and choosers
    # offsets is the offset into model_design df of first row of chooser alternatives
    offsets = np.arange(len(positions)) * sample_size
    # resulting Int64Index has one element per chooser row and is in same order as choosers
    choices = interaction_utilities.index.take(positions + offsets)
    # create a series with index from choosers and the index of the chosen alternative
    choices = pd.Series(choices, index=choosers.index)
    chunk.log_df(trace_label, 'choices', choices)

    if have_trace_targets:
        tracing.trace_df(choices, tracing.extend_trace_label(trace_label, 'choices'),
                         columns=[None, trace_choice_name])
        tracing.trace_df(rands, tracing.extend_trace_label(trace_label, 'rands'),
                         columns=[None, 'rand'])

    return choices


def calc_rows_per_chunk(chunk_size, choosers, alternatives, sample_size, skims, trace_label=None):

    num_choosers = len(choosers.index)

    # if not chunking, then return num_choosers
    # if chunk_size == 0:
    #     return num_choosers, 0

    chooser_row_size = len(choosers.columns)

    # alternative columns plus join column
    alt_row_size = alternatives.shape[1] + 1

    if skims is not None:
        alt_row_size += 1

    sample_size = sample_size or alternatives.shape[0]
    row_size = (chooser_row_size + alt_row_size) * sample_size

    # logger.debug("%s #chunk_calc choosers %s" % (trace_label, choosers.shape))
    # logger.debug("%s #chunk_calc alternatives %s" % (trace_label, alternatives.shape))
    # logger.debug("%s #chunk_calc chooser_row_size %s" % (trace_label, chooser_row_size))
    # logger.debug("%s #chunk_calc sample_size %s" % (trace_label, sample_size))
    # logger.debug("%s #chunk_calc alt_row_size %s" % (trace_label, alt_row_size))

    return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def interaction_simulate(
        choosers, alternatives, spec,
        skims=None, locals_d=None, sample_size=None, chunk_size=0,
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
        DataFrame of alternatives - will be merged with choosers, currently
        without sampling
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
    choices : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    trace_label = tracing.extend_trace_label(trace_label, 'interaction_simulate')

    assert len(choosers) > 0

    rows_per_chunk, effective_chunk_size = \
        calc_rows_per_chunk(chunk_size, choosers, alternatives=alternatives,
                            sample_size=sample_size, skims=skims,
                            trace_label=trace_label)

    result_list = []
    for i, num_chunks, chooser_chunk in chunk.chunked_choosers(choosers, rows_per_chunk):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(chooser_chunk)))

        chunk_trace_label = tracing.extend_trace_label(trace_label, 'chunk_%s' % i) \
            if num_chunks > 1 else trace_label

        chunk.log_open(chunk_trace_label, chunk_size, effective_chunk_size)

        choices = _interaction_simulate(chooser_chunk, alternatives, spec,
                                        skims, locals_d, sample_size,
                                        chunk_trace_label,
                                        trace_choice_name)

        chunk.log_close(chunk_trace_label)

        result_list.append(choices)

        force_garbage_collect()

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices
