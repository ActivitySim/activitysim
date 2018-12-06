# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import logging
import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import config


logger = logging.getLogger(__name__)


@inject.table()
def size_terms():
    f = config.config_file_path('destination_choice_size_terms.csv')
    return pd.read_csv(f, comment='#', index_col='segment')


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
        logger.warning("%s  missing columns in land use" % len(missing.index))
        for v in missing.index.values:
            logger.warning("missing: %s" % v)

    return land_use[coeffs.index].dot(coeffs)


def tour_destination_size_terms(land_use, size_terms, selector):
    """

    Parameters
    ----------
    land_use - pipeline table
    size_terms - pipeline table
    selector - str

    Returns
    -------

   ::

     pandas.dataframe
        one column per selector segment with index of land_use
        e.g. for selector 'work', columns will be work_low, work_med, work_high, work_veryhigh
        and for selector 'trip', columns will be eatout, escort, othdiscr, othmaint, ...

                 work_low    work_med  work_high   work_veryhigh
        TAZ                                            ...
        1      1267.00000     522.000  1108.000  1540.0000 ...
        2      1991.00000     824.500  1759.000  2420.0000 ...
    ...
    """

    land_use = land_use.to_frame()
    size_terms = size_terms.to_frame()

    size_terms = size_terms[size_terms.selector == selector].copy()
    del size_terms['selector']

    df = pd.DataFrame({key: size_term(land_use, row) for key, row in size_terms.iterrows()},
                      index=land_use.index)
    df.index.name = 'TAZ'

    if not (df.dtypes == 'float64').all():
        logger.warning('Surprised to find that not all size_terms were float64!')

    # - NARROW
    # float16 has 3.3 decimal digits of precision, float32 has 7.2
    df = df.astype(np.float16, errors='raise')
    assert np.isfinite(df.values).all()

    return df


def destination_predicted_size(choosers_table, selector, chooser_segment_column, segment_ids):

    land_use = inject.get_table('land_use')
    size_terms = inject.get_table('size_terms')
    choosers_df = inject.get_table(choosers_table).to_frame()

    # - raw_predicted_size
    raw_size = tour_destination_size_terms(land_use, size_terms, selector)
    assert set(raw_size.columns) == set(segment_ids.keys())

    segment_chooser_counts = \
        {segment_name: (choosers_df[chooser_segment_column] == segment_id).sum()
         for segment_name, segment_id in iteritems(segment_ids)}

    # - segment scale factor (modeled / predicted) keyed by segment_name
    # scaling reconciles differences between synthetic population and zone demographics
    # in a partial sample, it also scales predicted_size targets to sample population
    segment_scale_factors = {}
    for c in raw_size:
        segment_predicted_size = raw_size[c].astype(np.float64).sum()
        segment_scale_factors[c] = \
            segment_chooser_counts[c] / np.maximum(segment_predicted_size, 1)

    # - scaled_size = zone_size * (total_segment_modeled / total_segment_predicted)
    predicted_size = raw_size.astype(np.float64)
    for c in predicted_size:
        predicted_size[c] *= segment_scale_factors[c]

    # trace_label = "destination_predicted_size %s" % (selector)
    # print("%s raw_predicted_size\n" % (trace_label,), raw_size.head(20))
    # print("%s segment_scale_factors" % (trace_label,), segment_scale_factors)
    # print("%s predicted_size\n" % (trace_label,), predicted_size)

    return predicted_size
