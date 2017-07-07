# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from .skim import SkimDictWrapper, SkimStackWrapper
from . import logit
from . import tracing
from . import pipeline


logger = logging.getLogger(__name__)


def random_rows(df, n):

    # only sample if df has more than n rows
    if len(df.index) > n:
        prng = pipeline.get_rn_generator().get_global_rng()
        return df.take(prng.choice(len(df), size=n, replace=False))

    else:
        return df


def read_model_spec(fpath, fname,
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
    fpath : str
        path to directory containing file.
    fname : str
        Name of a CSV spec file
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

    with open(os.path.join(fpath, fname)) as f:
        spec = pd.read_csv(f, comment='#')

    spec = spec.dropna(subset=[expression_name])

    # don't need description and set the expression to the index
    if description_name in spec.columns:
        spec = spec.drop(description_name, axis=1)

    spec = spec.set_index(expression_name).fillna(0)

    return spec


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


def _check_for_variability(expression_values, trace_label):
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

    l = min(1000, len(expression_values))

    sample = random_rows(expression_values, l)

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


def eval_mnl(choosers, spec, locals_d,
             trace_label=None, trace_choice_name=None):
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
    locals_d : Dict or None
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

    expression_values = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # matrix product of spec expression_values with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas.dot depends on column names of expression_values matching spec index values

    utilities = expression_values.dot(spec)

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
        tracing.trace_df(expression_values, '%s.expression_values' % trace_label,
                         column_labels=['expression', None])

    return choices


def eval_nl(choosers, spec, nest_spec, locals_d,
            trace_label=None, trace_choice_name=None):
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
    locals_d : Dict or None
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

    # column names of expression_values match spec index values
    expression_values = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # raw utilities of all the leaves

    # matrix product of spec expression evals with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
        # pandas.dot depends on column names of expression_values matching spec index values
    raw_utilities = expression_values.dot(spec)

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
        tracing.trace_df(expression_values, '%s.expression_values' % trace_label,
                         column_labels=['expression', None])

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
        choices = eval_mnl(choosers, spec, locals_d,
                           trace_label=trace_label, trace_choice_name=trace_choice_name)
    else:
        choices = eval_nl(choosers, spec, nest_spec, locals_d,
                          trace_label=trace_label, trace_choice_name=trace_choice_name)

    return choices


def eval_mnl_logsums(choosers, spec, locals_d, trace_label=None):
    """
    like eval_nl except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be logsum across spec column values
    """

    trace_label = tracing.extend_trace_label(trace_label, 'mnl')
    check_for_variability = tracing.check_for_variability()

    expression_values = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # matrix product of spec expression evals with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas dot() depends on column names of expression_values matching spec index values
    utilities = expression_values.dot(spec)

    # logsum is log of exponentiated utilities summed across columns of each chooser row
    utils_arr = utilities.as_matrix().astype('float')
    logsums = np.log(np.exp(utils_arr).sum(axis=1))
    logsums = pd.Series(logsums, index=choosers.index)

    if trace_label:
        # add logsum to utilities for tracing
        utilities['logsum'] = logsums

        tracing.trace_df(choosers, '%s.choosers' % trace_label)
        tracing.trace_df(utilities, '%s.utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(logsums, '%s.logsums' % trace_label,
                         column_labels=['alternative', 'logsum'])
        tracing.trace_df(expression_values, '%s.expression_values' % trace_label,
                         column_labels=['expression', None])

    return logsums


def eval_nl_logsums(choosers, spec, nest_spec, locals_d, trace_label=None):
    """
    like eval_nl except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """

    trace_label = tracing.extend_trace_label(trace_label, 'nl_logsums')
    check_for_variability = tracing.check_for_variability()

    # column names of expression_values match spec index values
    expression_values = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # raw utilities of all the leaves

    # matrix product of spec expression evals with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas dot() depends on column names of expression_values matching spec index values
    raw_utilities = expression_values.dot(spec)

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)

    logsums = np.log(nested_exp_utilities.root)
    logsums = pd.Series(logsums, index=choosers.index)

    if trace_label:
        # add logsum to nested_exp_utilities for tracing
        nested_exp_utilities['logsum'] = logsums

        tracing.trace_df(choosers, '%s.choosers' % trace_label)
        tracing.trace_df(raw_utilities, '%s.raw_utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(nested_exp_utilities, '%s.nested_exp_utilities' % trace_label,
                         column_labels=['alternative', 'utility'])
        tracing.trace_df(logsums, '%s.logsums' % trace_label,
                         column_labels=['alternative', 'logsum'])

    return logsums


def simple_simulate_logsums(choosers, spec, nest_spec, skims=None, locals_d=None, trace_label=None):
    """
    like simple_simulate except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """
    if skims:
        add_skims(choosers, skims)

    trace_label = tracing.extend_trace_label(trace_label, 'simple_simulate_logsums')

    if nest_spec is None:
        logsums = eval_mnl_logsums(choosers, spec, locals_d, trace_label=trace_label)
    else:
        logsums = eval_nl_logsums(choosers, spec, nest_spec, locals_d, trace_label=trace_label)

    return logsums
