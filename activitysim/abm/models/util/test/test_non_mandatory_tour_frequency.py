# ActivitySim
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
from ..tour_frequency import process_non_mandatory_tours


def test_nmtf():

    persons = pd.DataFrame(
        {
            'non_mandatory_tour_frequency': [0, 3, 2, 1],
            'household_id': [1, 1, 2, 4],
            'home_taz': [100, 100, 200, 400]
        },
        index=[0, 1, 2, 3]
    )

    non_mandatory_tour_frequency_alts = pd.DataFrame(
        {
            "escort": [0, 0, 2, 0],
            "shopping": [1, 0, 0, 0],
            "othmaint": [0, 1, 0, 0]
        },
        index=[0, 1, 2, 3]
    )

    nmt = process_non_mandatory_tours(persons,
                                      non_mandatory_tour_frequency_alts)

    idx = nmt.index

    pdt.assert_series_equal(
        nmt.person_id,
        pd.Series(
            [0, 2, 2, 3], index=idx, name='person_id'))

    pdt.assert_series_equal(
        nmt.tour_type,
        pd.Series(
            ["shopping", "escort", "escort", "othmaint"],
            index=idx, name='tour_type'))
