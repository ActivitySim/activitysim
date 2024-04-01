# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd
import pandas.testing as pdt

from activitysim.abm.models.util.overlap import person_available_periods
from activitysim.core import workflow


def test_person_available_periods():
    state = workflow.State.make_default(__file__)

    # state.add_injectable("timetable", timetable)

    persons = pd.DataFrame(index=[1, 2, 3, 4])

    state.add_table("persons", persons)

    timetable = state.get_injectable("timetable")

    # first testing scenario with no tours assigned
    all_open = person_available_periods(
        state, persons, start_bin=None, end_bin=None, continuous=False
    )

    all_open_expected = pd.Series([19, 19, 19, 19], index=[1, 2, 3, 4])
    pdt.assert_series_equal(all_open, all_open_expected, check_dtype=False)

    # adding tours to the timetable

    tours = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3, 4],
            "tour_num": [1, 2, 1, 2, 1, 1],
            "start": [5, 10, 5, 20, 10, 20],
            "end": [6, 14, 18, 21, 23, 23],
            "tdds": [1, 89, 13, 181, 98, 183],
        },
        index=[1, 2, 3, 4, 5, 6],
    )
    # timetable.assign requires only 1 tour per person, so need to loop through tour nums
    for tour_num, nth_tours in tours.groupby("tour_num", sort=True):
        timetable.assign(
            window_row_ids=nth_tours["person_id"],
            tdds=nth_tours.tdds,
        )

    # testing time bins now available
    tours_all_bins = person_available_periods(
        state, persons, start_bin=None, end_bin=None, continuous=False
    )
    tours_all_bins_expected = pd.Series([16, 7, 7, 17], index=[1, 2, 3, 4])
    pdt.assert_series_equal(tours_all_bins, tours_all_bins_expected, check_dtype=False)

    # continuous time bins available
    continuous_test = person_available_periods(
        state, persons, start_bin=None, end_bin=None, continuous=True
    )
    continuous_test_expected = pd.Series([10, 6, 6, 16], index=[1, 2, 3, 4])
    pdt.assert_series_equal(
        continuous_test, continuous_test_expected, check_dtype=False
    )

    # start bin test
    start_test = person_available_periods(
        state, persons, start_bin=11, end_bin=None, continuous=True
    )
    start_test_expected = pd.Series([8, 6, 1, 5], index=[1, 2, 3, 4])
    pdt.assert_series_equal(start_test, start_test_expected, check_dtype=False)

    # end bin test
    end_test = person_available_periods(
        state, persons, start_bin=None, end_bin=11, continuous=False
    )
    end_test_expected = pd.Series([9, 1, 6, 12], index=[1, 2, 3, 4])
    pdt.assert_series_equal(end_test, end_test_expected, check_dtype=False)

    # assortment settings test
    assortment_test = person_available_periods(
        state, persons, start_bin=8, end_bin=15, continuous=True
    )
    assortment_test_expected = pd.Series([7, 3, 0, 8], index=[1, 2, 3, 4])
    pdt.assert_series_equal(
        assortment_test, assortment_test_expected, check_dtype=False
    )


if "__main__" == __name__:
    test_person_available_periods()
