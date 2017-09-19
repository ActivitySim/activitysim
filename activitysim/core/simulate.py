# ActivitySim
# See full license in LICENSE.txt.

from math import ceil
import os
import logging

import numpy as np
import pandas as pd

from .skim import SkimDictWrapper, SkimStackWrapper
from . import logit
from . import tracing
from . import pipeline


logger = logging.getLogger(__name__)


def num_chunk_rows_for_chunk_size(chunk_size, choosers, alternatives=None, by_chunk_id=False):

    # FIXME - chunk size should take number of chooser and alternative columns into account
    # FIXME - that is, chunk size should represent memory footprint (rows X columns) not just rows

    if by_chunk_id:
        num_choosers = choosers['chunk_id'].max() + 1
    else:
        num_choosers = len(choosers.index)

    # if not chunking, then return num_choosers
    if chunk_size == 0:
        return num_choosers

    row_size = len(choosers.columns)

    if alternatives is not None:
        alt_row_size = len(alternatives.columns)
        # logger.debug('num_chunk_rows_for_chunk_size row_size %s alt_row_size %s'
        #              % (row_size, alt_row_size,))
        row_size = row_size * alt_row_size

    if by_chunk_id:
        # scale row_size by average number of chooser rows per chunk_id
        rows_per_chunk_id = len(choosers.index) / float(num_choosers)
        row_size = int(rows_per_chunk_id * row_size)
        # logger.debug('num_chunk_rows_for_chunk_size by_chunk_id rows_per_chunk_id %s'
        #              % (rows_per_chunk_id,))

    # closest number of chooser rows to achieve chunk_size
    rows_per_chunk = int(round(chunk_size / float(row_size)))
    rows_per_chunk = max(rows_per_chunk, 1)

    logger.info("num_chunk_rows_for_chunk_size %s row_size %s rows_per_chunk %s "
                "num_choosers %s chunks %s"
                % (chunk_size, row_size, rows_per_chunk,
                   num_choosers, int(ceil(num_choosers / float(rows_per_chunk)))))

    return rows_per_chunk


def chunked_choosers(choosers, rows_per_chunk):
    # generator to iterate over chooses in chunk_size chunks
    num_choosers = len(choosers.index)
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    i = offset = 0
    while offset < num_choosers:
        yield i+1, num_chunks, choosers[offset: offset+rows_per_chunk]
        offset += rows_per_chunk
        i += 1


def chunked_choosers_and_alts(choosers, alternatives, rows_per_chunk):
    """
    like chunked_choosers, but also chunks alternatives
    for use with sampled alternatives which will have different alternatives (and numbers of alts)

    There may be up to sample_size (or as few as one) alternatives for each chooser
    because alternatives may have been sampled more than once,  but pick_count for those
    alternatives will always sum to sample_size.

    When we chunk the choosers, we need to take care chunking the alternatives as there are
    varying numbers of them for each chooser. Since alternatives appear in the same order
    as choosers, we can use cumulative pick_counts to identify boundaries of sets of alternatives

    Parameters
    ----------
    choosers
    alternatives : pandas DataFrame
        sample alternatives including pick_count column in same order as choosers
    rows_per_chunk : int

    Yields
    -------
    i : int
        one-based index of current chunk
    num_chunks : int
        total number of chunks that will be yielded
    choosers : pandas DataFrame slice
        chunk of choosers
    alternatives : pandas DataFrame slice
        chunk of alternatives for chooser chunk
    """

    assert 'cum_pick_count' not in alternatives.columns
    alternatives['cum_pick_count'] = alternatives['pick_count'].cumsum()

    # currently no convenient way to remember sample_size across steps
    pick_count = alternatives.cum_pick_count.iat[-1]
    sample_size = pick_count / len(choosers.index)
    assert pick_count % sample_size == 0

    # generator to iterate over choosers and alternatives in chunk_size chunks
    num_choosers = len(choosers.index)
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    alt_chunk_size = rows_per_chunk * sample_size

    # array of indices of starts of alt chunks
    alt_chunk_end = np.where(alternatives['cum_pick_count'] % alt_chunk_size == 0)[0] + 1
    # plus index of end of array for any final partial chunk
    alt_chunk_end = np.append(alt_chunk_end, [len(alternatives.index)])

    i = offset = alt_offset = 0
    while offset < num_choosers:

        alt_end = alt_chunk_end[i]

        chooser_chunk = choosers[offset: offset + rows_per_chunk]
        alternative_chunk = alternatives[alt_offset: alt_end]

        assert len(chooser_chunk.index) == len(np.unique(alternative_chunk.index.values))

        yield i+1, num_chunks, chooser_chunk, alternative_chunk

        i += 1
        offset += rows_per_chunk
        alt_offset = alt_end


def hh_chunked_choosers(choosers, rows_per_chunk):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    num_choosers = choosers['chunk_id'].max() + 1
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    i = offset = 0
    while offset < num_choosers:
        chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]
        yield i+1, num_chunks, chooser_chunk
        offset += rows_per_chunk
        i += 1


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


def eval_variables(exprs, df, locals_d=None, target_type=np.float64):
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
    target_type: dtype or None
        type to coerce results or None if no coercion desired

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns of eval results of `exprs`.
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
    for expr in exprs:
        try:
            if expr.startswith('@'):
                expr_values = to_series(eval(expr[1:], globals(), locals_d))
            else:
                expr_values = df.eval(expr)
            l.append((expr, expr_values))
        except Exception as err:
            logger.exception("Variable evaluation failed for: %s" % str(expr))
            raise err

    values = pd.DataFrame.from_items(l)

    # FIXME - for performance, it is essential that spec and expression_values
    # FIXME - not contain booleans when dotted with spec values
    # FIXME - or the arrays will be converted to dtype=object within dot()
    if target_type is not None:
        values = values.astype(target_type)

    return values


def compute_utilities(expression_values, spec):

    # matrix product of spec expression_values with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas.dot depends on column names of expression_values matching spec index values

    # FIXME - for performance, it is essential that spec and expression_values
    # FIXME - not contain booleans when dotted with spec values
    # FIXME - or the arrays will be converted to dtype=object within dot()

    spec = spec.astype(np.float64)

    utilities = expression_values.dot(spec)

    return utilities


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

    t0 = tracing.print_elapsed_time()

    expression_values = eval_variables(spec.index, choosers, locals_d)
    t0 = tracing.print_elapsed_time("eval_variables", t0, debug=True)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # matrix product of spec expression_values with utility coefficients of alternatives
    # sums the partial utilities (represented by each spec row) of the alternatives
    # resulting in a dataframe with one row per chooser and one column per alternative
    # pandas.dot depends on column names of expression_values matching spec index values

    utilities = compute_utilities(expression_values, spec)
    t0 = tracing.print_elapsed_time("expression_values.dot", t0, debug=True)

    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)
    t0 = tracing.print_elapsed_time("logit.utils_to_probs", t0, debug=True)

    choices, rands = logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)
    t0 = tracing.print_elapsed_time("logit.make_choices", t0, debug=True)

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

    t0 = tracing.print_elapsed_time()

    # column names of expression_values match spec index values
    expression_values = eval_variables(spec.index, choosers, locals_d)
    t0 = tracing.print_elapsed_time("eval_variables", t0, debug=True)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)
    t0 = tracing.print_elapsed_time("_check_for_variability", t0, debug=True)

    # raw utilities of all the leaves
    raw_utilities = compute_utilities(expression_values, spec)
    t0 = tracing.print_elapsed_time("expression_values.dot", t0, debug=True)

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    t0 = tracing.print_elapsed_time("compute_nested_exp_utilities", t0, debug=True)

    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = compute_nested_probabilities(nested_exp_utilities, nest_spec,
                                                        trace_label=trace_label)
    t0 = tracing.print_elapsed_time("compute_nested_probabilities", t0, debug=True)

    # global (flattened) leaf probabilities based on relative nest coefficients
    base_probabilities = compute_base_probabilities(nested_probabilities, nest_spec)
    t0 = tracing.print_elapsed_time("compute_base_probabilities", t0, debug=True)

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

    t0 = tracing.print_elapsed_time("report_bad_choices", t0, debug=True)

    choices, rands = logit.make_choices(base_probabilities, trace_label, trace_choosers=choosers)
    t0 = tracing.print_elapsed_time("logit.make_choices", t0, debug=True)

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


def _simple_simulate(choosers, spec, nest_spec, skims=None, locals_d=None,
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


def simple_simulate(choosers, spec, nest_spec, skims=None, locals_d=None, chunk_size=0,
                    trace_label=None, trace_choice_name=None):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.
    """

    assert len(choosers) > 0

    num_chunk_rows = num_chunk_rows_for_chunk_size(chunk_size, choosers)

    logger.info("simple_simulate num_chunk_rows %s num_choosers %s" %
                (num_chunk_rows, len(choosers.index)))

    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, num_chunks, chooser_chunk in chunked_choosers(choosers, num_chunk_rows):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(chooser_chunk)))

        choices = _simple_simulate(
            chooser_chunk, spec, nest_spec,
            skims, locals_d,
            tracing.extend_trace_label(trace_label, 'chunk_%s' % i),
            trace_choice_name)

        result_list.append(choices)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

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

    logger.debug("running eval_mnl_logsums")

    expression_values = eval_variables(spec.index, choosers, locals_d)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)

    # utility values
    utilities = compute_utilities(expression_values, spec)

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

    logger.debug("running eval_nl_logsums")
    t0 = tracing.print_elapsed_time()

    # column names of expression_values match spec index values
    expression_values = eval_variables(spec.index, choosers, locals_d)
    t0 = tracing.print_elapsed_time("eval_variables", t0, debug=True)

    if check_for_variability:
        _check_for_variability(expression_values, trace_label)
        t0 = tracing.print_elapsed_time("_check_for_variability", t0, debug=True)

    # raw utilities of all the leaves
    raw_utilities = compute_utilities(expression_values, spec)
    t0 = tracing.print_elapsed_time("expression_values.dot", t0, debug=True)

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    t0 = tracing.print_elapsed_time("compute_nested_exp_utilities", t0, debug=True)

    logsums = np.log(nested_exp_utilities.root)
    logsums = pd.Series(logsums, index=choosers.index)
    t0 = tracing.print_elapsed_time("logsums", t0, debug=True)

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


def _simple_simulate_logsums(choosers, spec, nest_spec,
                             skims=None, locals_d=None, trace_label=None):
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


def simple_simulate_logsums(choosers, spec, nest_spec,
                            skims=None, locals_d=None, chunk_size=0,
                            trace_label=None):
    """
    like simple_simulate except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """

    assert len(choosers) > 0

    num_chunk_rows = num_chunk_rows_for_chunk_size(chunk_size, choosers)

    logger.info("simple_simulate_logsums chunk_size %s num_choosers %s, num_chunk_rows %s" %
                (chunk_size, len(choosers.index), num_chunk_rows))

    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, num_chunks, chooser_chunk in chunked_choosers(choosers, num_chunk_rows):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(chooser_chunk)))

        logsums = _simple_simulate_logsums(
            chooser_chunk, spec, nest_spec,
            skims, locals_d,
            tracing.extend_trace_label(trace_label, 'chunk_%s' % i))

        result_list.append(logsums)

    if len(result_list) > 1:
        logsums = pd.concat(result_list)

    assert len(logsums.index == len(choosers.index))

    return logsums
