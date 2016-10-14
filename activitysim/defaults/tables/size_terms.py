# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd


logger = logging.getLogger(__name__)


def size_term(land_use, destination_choice_coeffs):
    """
    This method takes the land use data and multiplies various columns of the
    land use data by coefficients from the spec table in order
    to yield a size term (a linear combination of land use variables).

    Parameters
    ----------
    land_use : DataFrame
        A dataframe of land use attributes - the column names should match
        the index of destination_choice_coeffs
    destination_choice_coeffs : Series
        A series of coefficients for the land use attributes - the index
        describes the link to the land use table, and the values are floating
        points numbers used to do the linear combination

    Returns
    -------
    values : Series
        The index will be the same as land use, and the values will the
        linear combination of the land use table columns specified by the
        coefficients series.
    """
    coeffs = destination_choice_coeffs

    # first check for missing column in the land_use table
    missing = coeffs[~coeffs.index.isin(land_use.columns)]

    if len(missing) > 0:
        logger.warn("%s  missing columns in land use" % len(missing.index))
        for v in missing.index.values:
            logger.warn("missing: %s" % v)

    return land_use[coeffs.index].dot(coeffs)


@orca.table()
def size_terms(configs_dir):
    f = os.path.join(configs_dir, 'destination_choice_size_terms.csv')
    return pd.read_csv(f, index_col='segment')


@orca.table()
def destination_size_terms(land_use, size_terms):
    land_use = land_use.to_frame()
    size_terms = size_terms.to_frame()
    df = pd.DataFrame({key: size_term(land_use, row) for key, row in size_terms.iterrows()},
                      index=land_use.index)
    df.index.name = "TAZ"
    return df
