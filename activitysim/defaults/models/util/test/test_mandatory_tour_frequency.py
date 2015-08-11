# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# Copyright (C) 2015 Autodesk
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
from ..mandatory_tour_frequency import process_mandatory_tours


def test_mtf():
    persons = pd.DataFrame({
        "is_worker": [True, True, False],
        "mandatory_tour_frequency": ["work1", "work_and_school", "school2"]
    }, index=[10, 20, 30])

    mandatory_tours = process_mandatory_tours(persons)

    pdt.assert_series_equal(
        mandatory_tours.person_id,
        pd.Series(
            [10, 20, 20, 30, 30], index=[0, 1, 2, 3, 4], name='person_id'))

    pdt.assert_series_equal(
        mandatory_tours.tour_type,
        pd.Series(
            ['work', 'work', 'school', 'school', 'school'],
            index=[0, 1, 2, 3, 4], name='tour_type'))

    pdt.assert_series_equal(
        mandatory_tours.tour_num,
        pd.Series(
            [1, 1, 2, 1, 2], index=[0, 1, 2, 3, 4], name='tour_num'))
