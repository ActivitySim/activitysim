# ActivitySim
# See full license in LICENSE.txt.

from __future__ import division

import numpy as np
import pandas as pd


def utils_to_probs(utils):
    """
    Convert a table of utilities to exponentiated probabilities.

    Parameters
    ----------
    utils : pandas.DataFrame
        Rows should be choosers and columns should be alternatives.

    Returns
    -------
    probs : pandas.DataFrame
        Will have the same index and columns as `utils`.

    """
    prob_min = 1e-300
    prob_max = np.inf

    utils_arr = np.exp(utils.as_matrix().astype('float'))
    arr_sum = utils_arr.sum(axis=1)

    if np.isinf(arr_sum).any():
        raise RuntimeError('utilities have infinite values')

    np.clip(utils_arr, prob_min, prob_max, out=utils_arr)
    np.divide(
        utils_arr, arr_sum.reshape(len(utils_arr), 1),
        out=utils_arr)
    utils_arr[np.isnan(utils_arr)] = prob_min
    np.clip(utils_arr, prob_min, prob_max, out=utils_arr)

    return pd.DataFrame(utils_arr, columns=utils.columns, index=utils.index)


def make_choices(probs):
    """
    Make choices for each chooser from among a set of alternatives.

    Parameters
    ----------
    probs : pandas.DataFrame
        Rows for choosers and columns for the alternatives from which they
        are choosing. Values are expected to be valid probabilities across
        each row, e.g. they should sum to 1.

    Returns
    -------
    choices : pandas.Series
        Maps chooser IDs (from `probs` index) to a choice, where the choice
        is an index into the columns of `probs`.

    """
    nchoosers = len(probs)
    probs_arr = (
        probs.as_matrix().cumsum(axis=1) - np.random.random((nchoosers, 1)))
    rows, cols = np.where(probs_arr > 0)
    choices = (s.iat[0] for _, s in pd.Series(cols).groupby(rows))
    return pd.Series(choices, index=probs.index)


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

    alts_idx = np.arange(numalts)

    if sample_size < numalts:
        sample = np.concatenate(tuple(
            np.random.choice(alts_idx, sample_size, replace=False)
            for _ in range(numchoosers)))
    else:
        sample = np.tile(alts_idx, numchoosers)

    alts_sample = alternatives.take(sample)
    alts_sample['chooser_idx'] = np.repeat(choosers.index.values, sample_size)

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='chooser_idx', right_index=True,
        suffixes=('', '_r'))

    return alts_sample
