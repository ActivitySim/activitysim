# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
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
    prob_max = 1e20

    utils_arr = np.exp(utils.as_matrix().astype('float'))
    np.clip(utils_arr, prob_min, prob_max, out=utils_arr)
    np.divide(
        utils_arr, utils_arr.sum(axis=1).reshape(len(utils_arr), 1),
        out=utils_arr)
    utils_arr[np.isnan(utils_arr)] = prob_min
    np.clip(utils_arr, prob_min, prob_max, out=utils_arr)

    return pd.DataFrame(utils_arr, columns=utils.columns, index=utils.index)
