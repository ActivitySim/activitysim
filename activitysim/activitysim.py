# ActivitySim
# See full license in LICENSE.txt.

import logging
from operator import itemgetter

import numpy as np
import pandas as pd
from zbox import toolz as tz

from .skim import Skims, Skims3D
from .mnl import utils_to_probs, make_choices, interaction_dataset
import tracing

import os
import psutil
import gc


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
        return df.take(np.random.choice(len(df), size=n, replace=False))
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
    if locals_d is None:
        locals_d = {}
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
    Add the dataframe to the Skims object so that it can be dereferenced
    using the parameters of the skims object.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to which to add skim data as new columns.
        `df` is modified in-place.
    skims : Skims object
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    """
    if not isinstance(skims, list):
        assert isinstance(skims, Skims) or isinstance(skims, Skims3D)
        skims.set_df(df)
    else:
        for skim in skims:
            assert isinstance(skim, Skims) or isinstance(skim, Skims3D)
            skim.set_df(df)


def _check_for_variability(model_design):
    """
    This is an internal method which checks for variability in each
    expression - under the assumption that you probably wouldn't be using a
    variable (in live simulations) if it had no variability.  This is a
    warning to the user that they might have constructed the variable
    incorrectly.  It samples 1000 rows in order to not hurt performance -
    it's likely that if 1000 rows have no variability, the whole dataframe
    will have no variability.
    """
    l = min(1000, len(model_design))
    sample = random_rows(model_design, l)

    # convert to float so describe works uniformly on bools
    sample = sample.astype('float')
    sample = sample.describe().transpose()

    error = sample[sample["std"] == 0]
    if len(error):
        logger.warn("%s columns have no variability" % len(error))
        for v in error.index.values:
            logger.info("no variability in: %s" % v)
    error = sample[sample["count"] < l]
    if len(error):
        logger.warn("%s columns have missing values" % len(error))
        for v in error.index.values:
            logger.info("missing values in: %s" % v)


EMPTY_NEST = {'level': 0, 'product_of_coefficients': 1, 'ancestors': []}


def each_nest(node, nest=EMPTY_NEST, post_order=False):

    pre_order = not post_order
    level = nest['level']

    if isinstance(node, list):
        # only needed if we allow root level to be a list, instead of a dict
        for item in node:
            for sub_node, sub_nest in each_nest(item, nest, post_order):
                yield sub_node, sub_nest
    elif isinstance(node, dict):
        name = node['name']
        coefficient = node['coefficient']
        product_of_coefficients = nest['product_of_coefficients']*coefficient
        alternatives = [a['name'] if isinstance(a, dict) else a for a in node['alternatives']]

        nest = {
            'name': name,
            'coefficient': coefficient,
            'product_of_coefficients': product_of_coefficients,
            'alternatives': alternatives,
            'level': level + 1,
            'leaf': False,
            'ancestors': nest['ancestors'] + [name]
        }

        if pre_order:
            yield node, nest

        # recursively iterate the list of alternatives
        for alternative in node['alternatives']:
            for sub_node, sub_nest in each_nest(alternative, nest, post_order):
                yield sub_node, sub_nest

        if post_order:
            yield node, nest

    elif isinstance(node, str):
        nest = {
            'name': node,
            'product_of_coefficients': nest['product_of_coefficients'],
            'level': level + 1,
            'leaf': True,
            'ancestors': nest['ancestors'] + [node]
        }
        yield node, nest


def trace_nests(nests, trace_label):

    for node, nest in each_nest(nests):

        indent = "   " * nest['level']

        if nest['leaf']:
            tracing.info(trace_label,
                         "%s leaf name: %s level %s" % (indent, nest['name'], nest['level']))
            tracing.info(trace_label,
                         "%s ... product_of_coefficients: %s"
                         % (indent, nest['product_of_coefficients']))
        else:
            tracing.info(trace_label,
                         "%s node name: %s level %s" % (indent, nest['name'], nest['level']))
            tracing.info(trace_label,
                         "%s ... coefficient: %s" % (indent, nest['coefficient']))
            tracing.info(trace_label,
                         "%s ... product_of_coefficients: %s"
                         % (indent, nest['product_of_coefficients']))
            tracing.info(trace_label,
                         "%s ... alternatives: %s" % (indent, nest['alternatives']))

        tracing.info(trace_label,
                     "%s ... ancestors: %s" % (indent, nest['ancestors']))


def compute_nested_exp_utilities(raw_utilities, nests, trace_label=None):

    # leaf: exp( raw_utility )
    # nest: exp( ln(sum of exponentiated raw_utility of leaves) * nest_coefficient)

    trace_label = "%s.compute_nested_exp_utilities" % trace_label

    nested_utilities = pd.DataFrame(index=raw_utilities.index)

    for node, nest in each_nest(nests, post_order=True):

        name = nest['name']

        if nest['leaf']:
            product_of_nest_coefficients = nest['product_of_coefficients']
            nested_utilities[name] = \
                raw_utilities[name].astype(float) / product_of_nest_coefficients

            if trace_label:
                tracing.info(trace_label,
                             "nested_utilities[%s] = exp(utilities[%s] / %s)"
                             % (name, name, product_of_nest_coefficients))
        else:
            nest_coefficient = nest['coefficient']
            alternatives = nest['alternatives']
            nested_utilities[name] = \
                nest_coefficient * np.log(nested_utilities[alternatives].sum(axis=1))

            if trace_label:
                tracing.info(trace_label,
                             "nested_utilities[%s] = %s * nested_utilities[%s]"
                             % (name, nest_coefficient, alternatives))

        nested_utilities[name] = np.exp(nested_utilities[name])

    return nested_utilities


def compute_nested_probabilities(nested_exp_utilities, nests):

    # probability for nest alternatives is simply the alternatives's local (to nest) probability
    # computed in the same way as the probability of non-nested alternatives in multinomial logit
    # i.e. the fractional share of the sum of the exponentiated utility of itself and its siblings
    # except in nested logit, its sib group is restricted to the nest

    nested_probabilities = pd.DataFrame(index=nested_exp_utilities.index)

    for node, nest in each_nest(nests, post_order=False):

        if not nest['leaf']:
            name = nest['name']
            alternatives = nest['alternatives']

            probs = utils_to_probs(nested_exp_utilities[alternatives], exponentiated=True)

            nested_probabilities = pd.concat([nested_probabilities, probs], axis=1)

    return nested_probabilities


def compute_base_probabilities(nested_probabilities, nests):

    base_probabilities = pd.DataFrame(index=nested_probabilities.index)

    for node, nest in each_nest(nests, post_order=False):

        if nest['leaf']:

            name = nest['name']

            # skip root: it has a prob of 1 but we didn't compute a nested probability column for it
            ancestors = nest['ancestors'][1:]

            base_probabilities[name] = nested_probabilities[ancestors].prod(axis=1)

    return base_probabilities


def nested_simulate(choosers, spec, nests, skims=None, locals_d=None,
                    trace_label=None, trace_choice_name=None):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest:
        dictionary specifying nesting structure and nesting coefficients
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

    tracing.warn(__name__, "nested_simulate")

    if trace_label:
        trace_nests(nests, "%s.trace_nests" % trace_label)

    if skims:
        add_skims(choosers, skims)

    model_design = eval_variables(spec.index, choosers, locals_d)

    _check_for_variability(model_design)

    # raw utilities of all the leaves
    raw_utilities = model_design.dot(spec)

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nests, trace_label)

    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = compute_nested_probabilities(nested_exp_utilities, nests)

    # global leaf probabilities based on relative nest coefficients
    base_probabilities = compute_base_probabilities(nested_probabilities, nests)

    choices = make_choices(base_probabilities)

    if trace_label:

        trace_label = "%s.nested_simulate" % trace_label
        tracing.trace_choosers(choosers, trace_label)

        tracing.trace_df(raw_utilities, '%s.raw_utilities' % trace_label,
                         column_labels=['alternative', 'utility'], warn=True)
        tracing.trace_df(nested_exp_utilities, '%s.nested_exp_utilities' % trace_label,
                         column_labels=['alternative', 'utility'], warn=True)
        tracing.trace_df(nested_probabilities, '%s.nested_probabilities' % trace_label,
                         column_labels=['alternative', 'probability'], warn=True)
        tracing.trace_df(base_probabilities, '%s.base_probabilities' % trace_label,
                         column_labels=['alternative', 'probability'], warn=True)
        tracing.trace_choices(choices, trace_label, columns=[None, trace_choice_name])
        tracing.trace_model_design(model_design, trace_label)

        # to facilitiate debugging, compare to multinomial (non-nested) logit utilities
        unnested_probabilities = utils_to_probs(raw_utilities)
        unnested_choices = make_choices(unnested_probabilities)
        tracing.trace_df(unnested_probabilities, '%s.unnested_probabilities' % trace_label,
                         column_labels=['alternative', 'probability'], warn=True)
        tracing.trace_choices(unnested_choices, '%s.unnested_choices' % trace_label,
                              columns=[None, trace_choice_name])

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


def simple_simulate(choosers, spec, skims=None, locals_d=None,
                    trace_label=None, trace_choice_name=None):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
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

    model_design = eval_variables(spec.index, choosers, locals_d)

    _check_for_variability(model_design)

    utilities = model_design.dot(spec)

    probs = utils_to_probs(utilities)
    choices = make_choices(probs)

    if trace_label:
        trace_label = "%s.simple_simulate" % trace_label
        tracing.trace_choosers(choosers, trace_label)
        tracing.trace_utilities(utilities, trace_label)
        tracing.trace_probs(probs, trace_label)
        # FIXME - couldn't we always just use trace_label as choice column header?
        tracing.trace_choices(choices, trace_label, columns=[None, trace_choice_name])
        tracing.trace_model_design(model_design, trace_label)

    return choices, model_design


def chunked_choosers(choosers, chunk_size):
    # generator to iterate over chooses in chunk_size chunks
    chunk_size = int(chunk_size)
    num_choosers = len(choosers.index)

    i = 0
    while i < num_choosers:
        yield i, choosers[i: i+chunk_size]
        i += chunk_size


def interaction_simulate(
        choosers, alternatives, spec,
        skims=None, locals_d=None, sample_size=None, chunk_size=0,
        trace_label=None, trace_choice_name=None):

    # like _interaction_simulate but iterates over choosers in chunk_size chunks
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

        logger.info("Running chunk =%s of size %d" % (i, len(chooser_chunk)))

        choices = _interaction_simulate(chooser_chunk, alternatives, spec,
                                        skims, locals_d, sample_size,
                                        trace_label, trace_choice_name)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    assert len(choices.index == len(choosers.index))

    return choices


def _interaction_simulate(
        choosers, alternatives, spec,
        skims=None, locals_d=None, sample_size=None,
        trace_label=None, trace_choice_name=None):
    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

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

    Returns
    -------
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """
    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    sample_size = sample_size or len(alternatives)

    # FIXME - is this correct?
    if sample_size > len(alternatives):
        logger.warn("clipping sample size %s to len(alternatives) %s" %
                    (sample_size, len(alternatives)))
        sample_size = min(sample_size, len(alternatives))

    # now the index is also in the dataframe, which means it will be
    # available as the "destination" for the skims dereference below
    alternatives[alternatives.index.name] = alternatives.index

    # merge choosers and alternatives
    df = interaction_dataset(choosers, alternatives, sample_size)

    if skims:
        add_skims(df, skims)

    # evaluate variables from the spec
    model_design = eval_variables(spec.index, df, locals_d)

    _check_for_variability(model_design)

    # multiply by coefficients and reshape into choosers by alts
    utilities = model_design.dot(spec).astype('float')
    utilities = pd.DataFrame(
        utilities.as_matrix().reshape(len(choosers), sample_size),
        index=choosers.index)

    # convert to probabilities and make choices
    probs = utils_to_probs(utilities)
    positions = make_choices(probs)

    # positions come back between zero and num alternatives in the sample -
    # need to get back to the indexes
    offsets = np.arange(len(positions)) * sample_size
    choices = model_design.index.take(positions + offsets)

    choices = pd.Series(choices, index=choosers.index)

    if trace_label:
        trace_label = "%s.interaction_simulate" % trace_label
        tracing.trace_choosers(choosers, trace_label)
        tracing.trace_utilities(utilities, trace_label)
        tracing.trace_probs(probs, trace_label)
        tracing.trace_choices(choices, trace_label, columns=[None, trace_choice_name])
        tracing.trace_interaction_model_design(model_design, choosers, label=trace_label)

    return choices


def other_than(groups, bools):
    """
    Construct a Series that has booleans indicating the presence of
    something- or someone-else with a certain property within a group.

    Parameters
    ----------
    groups : pandas.Series
        A column with the same index as `bools` that defines the grouping
        of `bools`. The `bools` Series will be used to index `groups` and
        then the grouped values will be counted.
    bools : pandas.Series
        A boolean Series indicating where the property of interest is present.
        Should have the same index as `groups`.

    Returns
    -------
    others : pandas.Series
        A boolean Series with the same index as `groups` and `bools`
        indicating whether there is something- or something-else within
        a group with some property (as indicated by `bools`).

    """
    counts = groups[bools].value_counts()
    merge_col = groups.to_frame(name='right')
    pipeline = tz.compose(
        tz.curry(pd.Series.fillna, value=False),
        itemgetter('left'),
        tz.curry(
            pd.DataFrame.merge, right=merge_col, how='right', left_index=True,
            right_on='right'),
        tz.curry(pd.Series.to_frame, name='left'))
    gt0 = pipeline(counts > 0)
    gt1 = pipeline(counts > 1)

    return gt1.where(bools, other=gt0)
