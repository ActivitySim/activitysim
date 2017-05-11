# ActivitySim
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
from ..tour_frequency import process_mandatory_tours


def test_mtf():
    persons = pd.DataFrame({
        "is_worker": [True, True, False],
        "mandatory_tour_frequency": ["work1", "work_and_school", "school2"],
        "school_taz": [1, 2, 3],
        "workplace_taz": [10, 20, 30],
    }, index=[10, 20, 30])

    mandatory_tours = process_mandatory_tours(persons)

    idx = pd.Index([119, 229, 226, 335, 336], name="tour_id")

    pdt.assert_series_equal(
        mandatory_tours.person_id,
        pd.Series([10, 20, 20, 30, 30], index=idx, name='person_id'))

    pdt.assert_series_equal(
        mandatory_tours.tour_type,
        pd.Series(['work', 'work', 'school', 'school', 'school'], index=idx, name='tour_type'))

    pdt.assert_series_equal(
        mandatory_tours.tour_num,
        pd.Series([1, 1, 2, 1, 2], index=idx, name='tour_num'))

    pdt.assert_series_equal(
        mandatory_tours.destination,
        pd.Series([10, 20, 2, 3, 3], index=idx, name='destination'))
