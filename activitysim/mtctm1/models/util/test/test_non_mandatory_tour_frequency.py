# ActivitySim
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
from ..tour_frequency import process_non_mandatory_tours


def test_nmtf():

    non_mandatory_tour_frequency = pd.Series([0, 3, 2, 1])

    non_mandatory_tour_frequency_alts = pd.DataFrame(
        {
            "escort": [0, 0, 2, 0],
            "shopping": [1, 0, 0, 0],
            "othmaint": [0, 1, 0, 0]
        },
        index=[0, 1, 2, 3]
    )

    nmt = process_non_mandatory_tours(non_mandatory_tour_frequency,
                                      non_mandatory_tour_frequency_alts)

    idx = pd.Index([7, 23, 24, 37], name="tour_id")

    pdt.assert_series_equal(
        nmt.person_id,
        pd.Series(
            [0, 2, 2, 3], index=idx, name='person_id'))

    pdt.assert_series_equal(
        nmt.tour_type,
        pd.Series(
            ["shopping", "escort", "escort", "othmaint"],
            index=idx, name='tour_type'))
