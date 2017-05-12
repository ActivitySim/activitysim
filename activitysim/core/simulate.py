# ActivitySim
# See full license in LICENSE.txt.

import os
import psutil
import gc

import logging


import numpy as np
import pandas as pd
from zbox import toolz as tz

from .skim import SkimDictWrapper, SkimStackWrapper
from . import logit
from . import tracing
from . import pipeline


logger = logging.getLogger(__name__)


def memory_info():
    gc.collect()
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    mb = (bytes / (1024 * 1024.0))
    gb = (bytes / (1024 * 1024 * 1024.0))
    return "memory_info: %s MB (%s GB)" % (int(mb), round(gb, 2))


def random_rows(df, n):

    # only sample if df has more than n rows
    if len(df.index) > n:
        prng = pipeline.get_rn_generator().get_global_rng()
        return df.take(prng.choice(len(df), size=n, replace=False))

    else:
        return df


def read_model_spec(fname,
                    description_name="Description",
                    expression_name="Expression"):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    The CSV is expected to have columns for component descriptions
    and expressions, plus one or more alternatives.

    The CSV is required to have a header with column names. For example:

        Description,Expression,alt0,alt1,alt2

    Parameters
    ----------
    fname : str
        Name of a CSV spec file.
    description_name : str, optional
        Name of the column in `fname` that contains the component description.
    expression_name : str, optional
        Name of the column in `fname` that contains the component expression.

    Returns
    -------
    spec : pandas.DataFrame
        The description column is dropped from the returned data and the
        expression values are set as the table index.
    """

    cfg = pd.read_csv(fname, comment='#')

    cfg = cfg.dropna(subset=[expression_name])

    # don't need description and set the expression to the index
    cfg = cfg.drop(description_name, axis=1).set_index(expression_name)
    return cfg


def eval_variables(exprs, df, locals_d=None):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    There are two kinds of supported expressions: "simple" expressions are
    evaluated in the context of the DataFrame using DataFrame.eval.
    This is the default type of expression.

    Python expressions are evaluated in the context of this function using
    Python's eval function. Because we use Python's eval this type of
    expression supports more complex operations than a simple expression.
    Python expressions are denoted by beginning with the @ character.
    Users should take care that these expressions must result in
    a Pandas Series.

    Parameters
    ----------
    exprs : sequence of str
    df : pandas.DataFrame
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns of `exprs`.
    """

    # avoid altering caller's passed-in locals_d parameter (they may be looping)
    locals_d = locals_d.copy() if locals_d is not None else {}
    locals_d.update(locals())

    def to_series(x):
        if np.isscalar(x):
            return pd.Series([x] * len(df), index=df.index)
        return x

    l = []
    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously
    for e in exprs:
        try:
            l.append((e, to_series(eval(e[1:], globals(), locals_d))
                     if e.startswith('@') else df.eval(e)))
        except Exception as err:
            logger.exception("Variable evaluation failed for: %s" % str(e))
            raise err

    return pd.DataFrame.from_items(l)


def add_skims(df, skims):
    """
    Add the dataframe to the SkimDictWrapper object so that it can be dereferenced
    using the parameters of the skims object.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to which to add skim data as new columns.
        `df` is modified in-place.
    skims : SkimDictWrapper object
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    """
    if not isinstance(skims, list):
        assert isinstance(skims, SkimDictWrapper) or isinstance(skims, SkimStackWrapper)
        skims.set_df(df)
    else:
        for skim in skims:
            assert isinstance(skim, SkimDictWrapper) or isinstance(skim, SkimStackWrapper)
            skim.set_df(df)


def _check_for_variability(model_design, trace_label):
    """
    This is an internal method which checks for variability in each
    expression - under the assumption that you probably wouldn't be using a
    variable (in live simulations) if it had no variability.  This is a
    warning to the user that they might have constructed the variable
    incorrectly.  It samples 1000 rows in order to not hurt performance -
    it's likely that if 1000 rows have no variability, the whole dataframe
    will have no variability.
    """

    if trace_label is None:
        trace_label = '_check_for_variability'

    l = min(1000, len(model_design))

    sample = random_rows(model_design, l)

    no_variability = has_missing_vals = 0
    for i in range(len(sample.columns)):
        v = sample.iloc[:, i]
        if v.min() == v.max():
            col_name = sample.columns[i]
            logger.info("%s: no variability (%s) in: %s" % (trace_label, v.iloc[0], col_name))
            no_variability += 1
        # FIXME - how could this happen? Not sure it is really a problem?
        if np.count_nonzero(v.isnull().values) > 0:
            col_name = sample.columns[i]
            logger.info("%s: missing values in: %s" % (trace_label, v.iloc[0], col_name))
            has_missing_vals += 1

    if no_variability > 0:
        logger.warn("%s: %s columns have no variability" % (trace_label, no_variability))

    if has_missing_vals > 0:
        logger.warn("%s: %s columns have missing values" % (trace_label, has_missing_vals))


def compute_nested_exp_utilities(raw_utilities, nest_spec):
    """
    compute exponentiated nest utilities based on nesting coefficients

    For nest nodes this is the exponentiated logsum of alternatives adjusted by nesting coefficient

    leaf <- exp( raw_utility )
    nest <- exp( ln(sum of exponentiated raw_utility of leaves) * nest_coefficient)

    Parameters
    ----------
    raw_utilities : pandas.DataFrame
        dataframe with the raw alternative utilities of all leaves
        (what in non-nested logit would be the utilities of all the alternatives)
    nest_spec : dict
        Nest tree dict from the model spec yaml file

    Returns
    -------
    nested_utilities : pandas.DataFrame
        Will have the index of `raw_utilities` and columns for exponentiated leaf and node utilities
    """
    nested_utilities = pd.DataFrame(index=raw_utilities.index)

    for nest in logit.each_nest(nest_spec, post_order=True):

        name = nest.name

        if nest.is_leaf:
            # leaf_utility = raw_utility / nest.product_of_coefficients
            nested_utilities[name] = \
                raw_utilities[name].astype(float) / nest.product_of_coefficients

        else:
            # nest node
            # the alternative nested_utilities will already have been computed due to post_order
            # this will RuntimeWarning: divide by zero encountered in log
            # if all nest alternative utilities are zero
            # but the resulting inf will become 0 when exp is applied below
            nested_utilities[name] = \
                nest.coefficient * np.log(nested_utilities[nest.alternatives].sum(axis=1))

        # exponentiate the utility
        nested_utilities[name] = np.exp(nested_utilities[name])

    return nested_utilities


def compute_nested_probabilities(nested_exp_utilities, nest_spec, trace_label):
    """
    compute nested probabilities for nest leafs and nodes
    probability for nest alternatives is simply the alternatives's local (to nest) probability
    computed in the same way as the probability of non-nested alternatives in multinomial logit
    i.e. the fractional share of the sum of the exponentiated utility of itself and its siblings
    except in nested logit, its sib group is restricted to the nest

    Parameters
    ----------
    nested_exp_utilities : pandas.DataFrame
        dataframe with the exponentiated nested utilities of all leaves and nodes
    nest_spec : dict
        Nest tree dict from the model spec yaml file
    Returns
    -------
    nested_probabilities : pandas.DataFrame
        Will have the index of `nested_exp_utilities` and columns for leaf and node probabilities
    """

    nested_probabilities = pd.DataFrame(index=nested_exp_utilities.index)

    for nest in logit.each_nest(nest_spec, type='node', post_order=False):

        probs = logit.utils_to_probs(nested_exp_utilities[nest.alternatives],
                                     trace_label=trace_label,
                                     exponentiated=True,
                                     allow_zero_probs=True)

        nested_probabilities = pd.concat([nested_probabilities, probs], axis=1)

    return nested_probabilities


def compute_base_probabilities(nested_probabilities, nests):
    """
    compute base probabilities for nest leaves
    Base probabilities will be the nest-adjusted probabilities of all leaves
    This flattens or normalizes all the nested probabilities so that they have the proper global
    relative values (the leaf probabilities sum to 1 for each row.)

    Parameters
    ----------
    nested_probabilities : pandas.DataFrame
        dataframe with the nested probabilities for nest leafs and nodes
    nest_spec : dict
        Nest tree dict from the model spec yaml file
    Returns
    -------
    base_probabilities : pandas.DataFrame
        Will have the index of `nested_probabilities` and columns for leaf base probabilities
    """

    base_probabilities = pd.DataFrame(index=nested_probabilities.index)

    for nest in logit.each_nest(nests, type='leaf', post_order=False):

        # skip root: it has a prob of 1 but we didn't compute a nested probability column for it
        ancestors = nest.ancestors[1:]

        base_probabilities[nest.name] = nested_probabilities[ancestors].prod(axis=1)

    return base_probabilities


def eval_mnl(choosers, spec, locals_d=None, trace_label=None, trace_choice_name=None):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Each row in spec computes a partial utility for each alternative,
    by providing a spec expression (often a boolean 0-1 trigger)
    and a column of utility coefficients for each alternative.

    We compute the utility of each alternative by matrix-multiplication of eval results
    with the utility coefficients in the spec alternative columns
    yielding one row per chooser and one column per alternative

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """

    trace_label = tracing.extend_trace_label(trace_label, 'mnl')
    check_for_variability = tracing.check_for_variability()

    model_design = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(model_design, trace_label)

    # matrix product of spec expression evals with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas dot matrix-multiply depends on column names of model_design matching spec index values

    utilities = model_design.dot(spec)

    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)
    choices, rands = logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)

    if trace_label:

        tracing.trace_df(choosers, '%s.choosers' % trace_label)
        tracing.trace_df(utilities, '%s.utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(probs, '%s.probs' % trace_label,
                         column_labels=['alternative', 'probability'])
        tracing.trace_df(choices, '%s.choices' % trace_label,
                         columns=[None, trace_choice_name])
        tracing.trace_df(rands, '%s.rands' % trace_label,
                         columns=[None, 'rand'])
        tracing.trace_df(model_design, '%s.model_design' % trace_label,
                         column_labels=['expression', None])

    return choices


def eval_nl(choosers, spec, nest_spec, locals_d=None, trace_label=None, trace_choice_name=None):
    """
    Run a nested-logit simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec:
        dictionary specifying nesting structure and nesting coefficients
        (from the model spec yaml file)
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """

    trace_label = tracing.extend_trace_label(trace_label, 'nl')
    check_for_variability = tracing.check_for_variability()

    # column names of model_design match spec index values
    model_design = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(model_design, trace_label)

    # raw utilities of all the leaves

    # matrix product of spec expression evals with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas dot matrix-multiply depends on column names of model_design matching spec index values
    raw_utilities = model_design.dot(spec)

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)

    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = compute_nested_probabilities(nested_exp_utilities, nest_spec,
                                                        trace_label=trace_label)

    # global (flattened) leaf probabilities based on relative nest coefficients
    base_probabilities = compute_base_probabilities(nested_probabilities, nest_spec)

    # note base_probabilities could all be zero since we allowed all probs for nests to be zero
    # check here to print a clear message but make_choices will raise error if probs don't sum to 1
    BAD_PROB_THRESHOLD = 0.001
    no_choices = \
        base_probabilities.sum(axis=1).sub(np.ones(len(base_probabilities.index))).abs() \
        > BAD_PROB_THRESHOLD * np.ones(len(base_probabilities.index))

    if no_choices.any():
        logit.report_bad_choices(
            no_choices, base_probabilities,
            tracing.extend_trace_label(trace_label, 'eval_nl'),
            tag='bad_probs',
            msg="base_probabilities all zero")

    choices, rands = logit.make_choices(base_probabilities, trace_label, trace_choosers=choosers)

    if trace_label:
        tracing.trace_df(choosers, '%s.choosers' % trace_label)
        tracing.trace_df(raw_utilities, '%s.raw_utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(nested_exp_utilities, '%s.nested_exp_utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(nested_probabilities, '%s.nested_probabilities' % trace_label,
                         column_labels=['alternative', 'probability'])
        tracing.trace_df(base_probabilities, '%s.base_probabilities' % trace_label,
                         column_labels=['alternative', 'probability'])
        tracing.trace_df(choices, '%s.choices' % trace_label,
                         columns=[None, trace_choice_name])
        tracing.trace_df(rands, '%s.rands' % trace_label,
                         columns=[None, 'rand'])
        tracing.trace_df(model_design, '%s.model_design' % trace_label,
                         column_labels=['expression', None])

        # dump whole df - for software development debugging
        # tracing.trace_df(raw_utilities, "%s.raw_utilities" % trace_label,
        #                  slicer='NONE', transpose=False)
        # tracing.trace_df(nested_exp_utilities, "%s.nested_exp_utilities" % trace_label,
        #                  slicer='NONE', transpose=False)
        # tracing.trace_df(nested_probabilities, "%s.nested_probabilities" % trace_label,
        #                  slicer='NONE', transpose=False)
        # tracing.trace_df(base_probabilities, "%s.base_probabilities" % trace_label,
        #                  slicer='NONE', transpose=False)
        # tracing.trace_df(unnested_probabilities, "%s.unnested_probabilities" % trace_label,
        #                  slicer='NONE', transpose=False)

    return choices


def simple_simulate(choosers, spec, nest_spec, skims=None, locals_d=None,
                    trace_label=None, trace_choice_name=None):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec:
        for nested logit (nl): dictionary specifying nesting structure and nesting coefficients
        for multinomial logit (mnl): None
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
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """
    if skims:
        add_skims(choosers, skims)

    trace_label = tracing.extend_trace_label(trace_label, 'simple_simulate')

    if nest_spec is None:
        choices = eval_mnl(choosers, spec, locals_d, trace_label, trace_choice_name)
    else:
        choices = eval_nl(choosers, spec, nest_spec, locals_d, trace_label, trace_choice_name)

    return choices


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
        trace_eval_results = []
    else:
        trace_eval_results = None

    check_for_variability = tracing.check_for_variability()

    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously

    utilities = pd.DataFrame({'utility': 0.0}, index=df.index)
    no_variability = has_missing_vals = 0

    for expr, coefficient in zip(spec.index, spec.iloc[:, 0]):
        try:

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
                trace_eval_results.append((expr,
                                           v[trace_rows]))
                trace_eval_results.append(('partial utility (coefficient = %s)' % coefficient,
                                           v[trace_rows]*coefficient))
                # trace_eval_results.append(('cumulative utility',
                #                            utilities.utility[trace_rows]))

        except Exception as err:
            logger.exception("Variable evaluation failed for: %s" % str(expr))
            raise err

    if no_variability > 0:
        logger.warn("%s: %s columns have no variability" % (trace_label, no_variability))

    if has_missing_vals > 0:
        logger.warn("%s: %s columns have missing values" % (trace_label, has_missing_vals))

    if trace_eval_results is not None:

        trace_eval_results.append(('total utility',
                                   utilities.utility[trace_rows]))

        trace_eval_results = pd.DataFrame.from_items(trace_eval_results)
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
    have_trace_targets = trace_label and tracing.has_trace_targets(choosers)

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
    if skims:
        alternatives[alternatives.index.name] = alternatives.index

    # cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    interaction_df = logit.interaction_dataset(choosers, alternatives, sample_size)

    if skims:
        add_skims(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of model_design match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in model_design
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

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.as_matrix().reshape(len(choosers), sample_size),
        index=choosers.index)

    if have_trace_targets:
        tracing.trace_df(utilities, tracing.extend_trace_label(trace_label, 'utilities'),
                         column_labels=['alternative', 'utility'])

    # tracing.trace_df(utilities, '%s.DUMP.utilities' % trace_label, transpose=False, slicer='NONE')

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)

    if have_trace_targets:
        tracing.trace_df(probs, tracing.extend_trace_label(trace_label, 'probs'),
                         column_labels=['alternative', 'probability'])

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)

    # need to get from an integer offset into the alternative sample to the alternative index
    # that is, we want the index value of the row that is offset by <position> rows into the
    # tranche of this choosers alternatives created by cross join of alternatives and choosers

    # offsets is the offset into model_design df of first row of chooser alternatives
    offsets = np.arange(len(positions)) * sample_size
    # resulting pandas Int64Index has one element per chooser row and is in same order as choosers
    choices = interaction_utilities.index.take(positions + offsets)

    # create a series with index from choosers and the index of the chosen alternative
    choices = pd.Series(choices, index=choosers.index)

    if have_trace_targets:
        tracing.trace_df(choices, tracing.extend_trace_label(trace_label, 'choices'),
                         columns=[None, trace_choice_name])
        tracing.trace_df(rands, tracing.extend_trace_label(trace_label, 'rands'),
                         columns=[None, 'rand'])

    #
    # if have_trace_targets:
    #     tracing.trace_df(choosers, '%s.choosers' % trace_label)
    #     tracing.trace_df(utilities, '%s.utilities' % trace_label,
    #                      column_labels=['alternative', 'utility'])
    #     tracing.trace_df(probs, '%s.probs' % trace_label,
    #                      column_labels=['alternative', 'probability'])
    #     tracing.trace_df(choices, '%s.choices' % trace_label,
    #                      columns=[None, trace_choice_name])
    #     tracing.trace_interaction_eval_results(trace_eval_results, trace_ids,
    #                                            '%s.eval' % trace_label)

    return choices


def chunked_choosers(choosers, chunk_size):
    # generator to iterate over chooses in chunk_size chunks
    chunk_size = int(chunk_size)
    num_choosers = len(choosers.index)

    i = offset = 0
    while offset < num_choosers:
        yield i, choosers[offset: offset+chunk_size]
        offset += chunk_size
        i += 1


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
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    # FIXME - chunk size should take number of chooser and alternative columns into account
    # FIXME - that is, chunk size should represent memory footprint (rows X columns) not just rows

    chunk_size = int(chunk_size)

    if (chunk_size == 0) or (chunk_size >= len(choosers.index)):
        choices = _interaction_simulate(choosers, alternatives, spec,
                                        skims, locals_d, sample_size,
                                        trace_label, trace_choice_name)
        return choices

    logger.info("interaction_simulate chunk_size %s num_choosers %s" %
                (chunk_size, len(choosers.index)))

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, chooser_chunk in chunked_choosers(choosers, chunk_size):

        logger.info("Running chunk %s of size %d" % (i, len(chooser_chunk)))

        choices = _interaction_simulate(chooser_chunk, alternatives, spec,
                                        skims, locals_d, sample_size,
                                        tracing.extend_trace_label(trace_label, 'chunk_%s' % i),
                                        trace_choice_name)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    assert len(choices.index == len(choosers.index))

    return choices
