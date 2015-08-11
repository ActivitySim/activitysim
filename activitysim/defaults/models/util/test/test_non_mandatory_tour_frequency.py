# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# Copyright (C) 2015 Autodesk
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
from ..non_mandatory_tour_frequency import process_non_mandatory_tours


def test_nmtf():

    non_mandatory_tour_frequency = pd.Series([0, 3, 2, 1])

    non_mandatory_tour_frequency_alts = pd.DataFrame(
        {
            "escort": [0, 0, 1, 0],
            "shopping": [1, 0, 0, 0],
            "random": [0, 2, 0, 0]
        },
        index=[0, 1, 2, 3]
    )

    nmt = process_non_mandatory_tours(non_mandatory_tour_frequency,
                                      non_mandatory_tour_frequency_alts)

    pdt.assert_series_equal(
        nmt.person_id,
        pd.Series(
            [0, 2, 3, 3], index=[0, 1, 2, 3], name='person_id'))

    pdt.assert_series_equal(
        nmt.tour_type,
        pd.Series(
            ["shopping", "escort", "random", "random"],
            index=[0, 1, 2, 3], name='tour_type'))
