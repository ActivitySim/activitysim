# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import config, inject

logger = logging.getLogger(__name__)


@inject.injectable(cache=True)
def size_terms():
    f = config.config_file_path("destination_choice_size_terms.csv")
    return pd.read_csv(f, comment="#", index_col="segment")


def size_term(land_use, destination_choice_coefficients):
    """
    This method takes the land use data and multiplies various columns of the
    land use data by coefficients from the spec table in order
    to yield a size term (a linear combination of land use variables).

    Parameters
    ----------
    land_use : DataFrame
        A dataframe of land use attributes - the column names should match
        the index of destination_choice_coefficients
    destination_choice_coefficients : Series
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
    coefficients = destination_choice_coefficients

    # first check for missing column in the land_use table
    missing = coefficients[~coefficients.index.isin(land_use.columns)]

    if len(missing) > 0:
        logger.warning("%s  missing columns in land use" % len(missing.index))
        for v in missing.index.values:
            logger.warning("missing: %s" % v)

    return land_use[coefficients.index].dot(coefficients)


def tour_destination_size_terms(land_use, size_terms, model_selector):
    """

     Parameters
     ----------
     land_use - pipeline table
     size_terms - pipeline table
     model_selector - str

     Returns
     -------

    ::

      pandas.dataframe
         one column per model_selector segment with index of land_use
         e.g. for model_selector 'workplace', columns will be work_low, work_med, ...
         and for model_selector 'trip', columns will be eatout, escort, othdiscr, ...

                  work_low    work_med  work_high   work_veryhigh
         zone_id                                         ...
         1      1267.00000     522.000  1108.000  1540.0000 ...
         2      1991.00000     824.500  1759.000  2420.0000 ...
     ...
    """

    land_use = land_use.to_frame()

    # don't count on land_use being sorted by index
    if not land_use.index.is_monotonic_increasing:
        land_use = land_use.sort_index()

    size_terms = size_terms[size_terms.model_selector == model_selector].copy()
    del size_terms["model_selector"]

    df = pd.DataFrame(
        {key: size_term(land_use, row) for key, row in size_terms.iterrows()},
        index=land_use.index,
    )

    assert land_use.index.name == "zone_id"
    df.index.name = land_use.index.name

    if not (df.dtypes == "float64").all():
        logger.warning("Surprised to find that not all size_terms were float64!")

    if df.isna().any(axis=None):
        logger.warning(
            f"tour_destination_size_terms with NAN values\n{df[df.isna().any(axis=1)]}"
        )
        assert not df.isna().any(axis=None)

    return df
