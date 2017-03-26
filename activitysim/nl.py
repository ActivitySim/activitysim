# ActivitySim
# See full license in LICENSE.txt.

from __future__ import division

import logging

import numpy as np
import pandas as pd

import tracing
import pipeline

logger = logging.getLogger(__name__)


def report_bad_choices(bad_row_map, df, trace_label, msg, trace_choosers=None):
    """

    Parameters
    ----------
    bad_row_map
    df : pandas.DataFrame
        utils or probs dataframe
    msg : str
        message describing the type of bad choice that necessitates error being thrown
    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        because we  can't deduce hh_id from the interaction_dataset which is indexed on index
        values from alternatives df

    Returns
    -------
    raises RuntimeError
    """
    MAX_DUMP = 1000
    MAX_PRINT = 10

    msg_with_count = "%s for %s rows" % (msg, bad_row_map.sum())
    logger.critical(msg_with_count)

    if trace_label:
        logger.critical("dumping %s" % trace_label)
        tracing.write_csv(df[bad_row_map][:MAX_DUMP],
                          file_name=trace_label,
                          transpose=False)

    # log the indexes of the first MAX_DUMP offending rows
    for idx in df.index[bad_row_map][:MAX_PRINT].values:

        if trace_choosers is None:
            hh_id = tracing.hh_id_for_chooser(idx, df)
        else:
            hh_id = tracing.hh_id_for_chooser(idx, trace_choosers)

        row_msg = "%s : %s in: %s = %s (hh_id = %s)" % (trace_label, msg, df.index.name, idx, hh_id)
        logger.critical(row_msg)

    raise RuntimeError(msg_with_count)


def utils_to_probs(utils, trace_label=None, exponentiated=False, allow_zero_probs=False,
                   trace_choosers=None):
    """
    Convert a table of utilities to probabilities.

    Parameters
    ----------
    utils : pandas.DataFrame
        Rows should be choosers and columns should be alternatives.

    trace_label : str
        label for tracing bad utility or probability values

    exponentiated : bool
        True if utilities have already been exponentiated

    allow_zero_probs : bool
        if True value rows in which all utility alts are EXP_UTIL_MIN will result
        in rows in probs to have all zero probability (and not sum to 1.0)
        This si for hte benefit of calculating probabilities of nested logit nests

    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        by report_bad_choices because it can't deduce hh_id from the interaction_dataset
        which is indexed on index values from alternatives df

    Returns
    -------
    probs : pandas.DataFrame
        Will have the same index and columns as `utils`.

    """
    trace_label = tracing.extend_trace_label(trace_label, 'utils_to_probs')

    utils_arr = utils.as_matrix().astype('float')
    if not exponentiated:
        utils_arr = np.exp(utils_arr)

    EXP_UTIL_MIN = 1e-300
    EXP_UTIL_MAX = np.inf
    np.clip(utils_arr, EXP_UTIL_MIN, EXP_UTIL_MAX, out=utils_arr)

    # FIXME
    utils_arr = np.where(utils_arr == EXP_UTIL_MIN, 0.0, utils_arr)

    arr_sum = utils_arr.sum(axis=1)

    zero_probs = (arr_sum == 0.0)
    if zero_probs.any() and not allow_zero_probs:

        report_bad_choices(zero_probs, utils,
                           tracing.extend_trace_label(trace_label, 'zero_prob_utils'),
                           msg="all probabilities are zero",
                           trace_choosers=trace_choosers)

    inf_utils = np.isinf(arr_sum)
    if inf_utils.any():
        report_bad_choices(inf_utils, utils,
                           tracing.extend_trace_label(trace_label, 'inf_exp_utils'),
                           msg="infinite exponentiated utilities",
                           trace_choosers=trace_choosers)

    # if allow_zero_probs, this may cause a RuntimeWarning: invalid value encountered in divide
    np.divide(utils_arr, arr_sum.reshape(len(utils_arr), 1), out=utils_arr)

    PROB_MIN = 0.0
    PROB_MAX = 1.0

    # if allow_zero_probs, this will cause EXP_UTIL_MIN util rows to have all zero probabilities
    utils_arr[np.isnan(utils_arr)] = PROB_MIN

    np.clip(utils_arr, PROB_MIN, PROB_MAX, out=utils_arr)

    probs = pd.DataFrame(utils_arr, columns=utils.columns, index=utils.index)

    return probs


def make_choices(probs, trace_label=None, trace_choosers=None):
    """
    Make choices for each chooser from among a set of alternatives.

    Parameters
    ----------
    probs : pandas.DataFrame
        Rows for choosers and columns for the alternatives from which they
        are choosing. Values are expected to be valid probabilities across
        each row, e.g. they should sum to 1.

    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        by report_bad_choices because it can't deduce hh_id from the interaction_dataset
        which is indexed on index values from alternatives df

    Returns
    -------
    choices : pandas.Series
        Maps chooser IDs (from `probs` index) to a choice, where the choice
        is an index into the columns of `probs`.

    """
    trace_label = tracing.extend_trace_label(trace_label, 'make_choices')

    # probs should sum to 1 across each row

    BAD_PROB_THRESHOLD = 0.001
    bad_probs = \
        probs.sum(axis=1).sub(np.ones(len(probs.index))).abs() \
        > BAD_PROB_THRESHOLD * np.ones(len(probs.index))

    if bad_probs.any():

        report_bad_choices(bad_probs, probs,
                           tracing.extend_trace_label(trace_label, 'bad_probs'),
                           msg="probabilities do not add up to 1",
                           trace_choosers=trace_choosers)

    rands = pipeline.get_rn_generator().random_for_df(probs)
    probs_arr = probs.as_matrix().cumsum(axis=1) - rands

    rows, cols = np.where(probs_arr > 0)
    choices = (s.iat[0] for _, s in pd.Series(cols).groupby(rows))

    choices = pd.Series(choices, index=probs.index)
    rands = pd.Series(np.asanyarray(rands).flatten(), index=probs.index)

    return choices, rands


def interaction_dataset(choosers, alternatives, sample_size=None):
    """
    Combine choosers and alternatives into one table for the purposes
    of creating interaction variables and/or sampling alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    alternatives : pandas.DataFrame
    sample_size : int, optional
        If sampling from alternatives for each chooser, this is
        how many to sample.

    Returns
    -------
    interacted : pandas.DataFrame
        Merged choosers and alternatives with data repeated either
        len(alternatives) or `sample_size` times.

    """
    if not choosers.index.is_unique:
        raise RuntimeError(
            "ERROR: choosers index is not unique, "
            "sample will not work correctly")
    if not alternatives.index.is_unique:
        raise RuntimeError(
            "ERROR: alternatives index is not unique, "
            "sample will not work correctly")

    numchoosers = len(choosers)
    numalts = len(alternatives)
    sample_size = sample_size or numalts

    # FIXME - is this faster or just dumb?
    alts_idx = np.arange(numalts)

    if sample_size < numalts:
        sample = pipeline.get_rn_generator().choice_for_df(choosers,
                                                           alts_idx, sample_size, replace=False)
    else:
        sample = np.tile(alts_idx, numchoosers)

    alts_sample = alternatives.take(sample)
    alts_sample['chooser_idx'] = np.repeat(choosers.index.values, sample_size)

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='chooser_idx', right_index=True,
        suffixes=('', '_r'))

    return alts_sample


class Nest(object):
    """
    Data for a nest-logit node or leaf

    This object is passed on yield when iterate over nest nodes (branch or leaf)
    The nested logit design is stored in a yaml file as a tree of dict objects,
    but using an object to pass the nest data makes the code a little more readable

    An example nest specification is in the example tour mode choice model
    yaml configuration file - example/configs/tour_mode_choice.yaml.
    """

    def __init__(self, name=None, level=0):
        self.name = name
        self.level = level
        self.product_of_coefficients = 1
        self.ancestors = []
        self.alternatives = None
        self.coefficient = 0

    @property
    def is_leaf(self):
        return (self.alternatives is None)

    @property
    def type(self):
        return 'leaf' if self.is_leaf else 'node'

    @classmethod
    def nest_types(cls):
        return ['leaf', 'node']


def _each_nest(spec, parent_nest, post_order):
    """
    Iterate over each nest or leaf node in the tree (of subtree)

    This internal routine is called by each_nest, which presents a slightly higer level interface

    Parameters
    ----------
    spec : dict
        Nest spec dict tree (or subtree when recursing) from the model spec yaml file
    parent_nest : Nest
        nest of parent node (passed to accumulate level, ancestors, and product_of_coefficients)
    post_order : Bool
        Should we iterate over the nodes of the tree in post-order or pre-order?
        (post-order means we yield the alternatives sub-tree before current node.)

    Yields
    -------
        spec_node : dict
            Nest tree spec dict for this node subtree
        nest : Nest
            Nest object with info about the current node (nest or leaf)
    """
    pre_order = not post_order

    level = parent_nest.level + 1

    if isinstance(spec, dict):
        name = spec['name']
        coefficient = spec['coefficient']
        alternatives = [a['name'] if isinstance(a, dict) else a for a in spec['alternatives']]

        nest = Nest(name=name)
        nest.level = parent_nest.level + 1
        nest.coefficient = coefficient
        nest.product_of_coefficients = parent_nest.product_of_coefficients * coefficient
        nest.alternatives = alternatives
        nest.ancestors = parent_nest.ancestors + [name]

        if pre_order:
            yield spec, nest

        # recursively iterate the list of alternatives
        for alternative in spec['alternatives']:
            for sub_node, sub_nest in _each_nest(alternative, nest, post_order):
                yield sub_node, sub_nest

        if post_order:
            yield spec, nest

    elif isinstance(spec, str):
        name = spec

        nest = Nest(name=name)
        nest.level = parent_nest.level + 1
        nest.product_of_coefficients = parent_nest.product_of_coefficients
        nest.ancestors = parent_nest.ancestors + [name]

        yield spec, nest


def each_nest(nest_spec, type=None, post_order=False):
    """
    Iterate over each nest or leaf node in the tree (of subtree)

    Parameters
    ----------
    nest_spec : dict
        Nest tree dict from the model spec yaml file
    type : str
        Nest class type to yield
        None yields all nests
        'leaf' yields only leaf nodes
        'branch' yields only branch nodes
    post_order : Bool
        Should we iterate over the nodes of the tree in post-order or pre-order?
        (post-order means we yield the alternatives sub-tree before current node.)

    Yields
    -------
        nest : Nest
            Nest object with info about the current node (nest or leaf)
    """
    if type is not None and type not in Nest.nest_types():
        raise RuntimeError("Unknown nest type '%s' in call to each_nest" % type)

    for node, nest in _each_nest(nest_spec, parent_nest=Nest(), post_order=post_order):
        if type is None or (type == nest.type):
            yield nest
