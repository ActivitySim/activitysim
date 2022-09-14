# ActivitySim
# See full license in LICENSE.txt.

from builtins import range

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from .. import chunk
from .. import timetable as tt


@pytest.fixture
def persons():

    df = pd.DataFrame(index=list(range(6)))

    return df


@pytest.fixture
def tdd_alts():
    alts = pd.DataFrame(
        data=[
            [5, 5],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 9],
            [5, 10],
            [6, 6],
            [6, 7],
            [6, 8],
            [6, 9],
            [6, 10],
            [7, 7],
            [7, 8],
            [7, 9],
            [7, 10],
            [8, 8],
            [8, 9],
            [8, 10],
            [9, 9],
            [9, 10],
            [10, 10],
        ],
        columns=["start", "end"],
    )
    alts["duration"] = alts.end - alts.start

    return alts


def test_basic(persons, tdd_alts):

    with chunk.chunk_log("test_basic", base=True):

        person_windows = tt.create_timetable_windows(persons, tdd_alts)

        timetable = tt.TimeTable(person_windows, tdd_alts, "person_windows")

        # print "\ntdd_footprints_df\n", timetable.tdd_footprints_df
        #     0  1  2  3  4  5  6  7
        # 0   0  6  0  0  0  0  0  0
        # 1   0  2  4  0  0  0  0  0
        # 2   0  2  7  4  0  0  0  0
        # 3   0  2  7  7  4  0  0  0
        # 4   0  2  7  7  7  4  0  0
        # 5   0  2  7  7  7  7  4  0
        # 6   0  0  6  0  0  0  0  0
        # 7   0  0  2  4  0  0  0  0
        # 8   0  0  2  7  4  0  0  0
        # 9   0  0  2  7  7  4  0  0
        # 10  0  0  2  7  7  7  4  0
        # 11  0  0  0  6  0  0  0  0
        # 12  0  0  0  2  4  0  0  0
        # 13  0  0  0  2  7  4  0  0
        # 14  0  0  0  2  7  7  4  0
        # 15  0  0  0  0  6  0  0  0
        # 16  0  0  0  0  2  4  0  0
        # 17  0  0  0  0  2  7  4  0
        # 18  0  0  0  0  0  6  0  0
        # 19  0  0  0  0  0  2  4  0
        # 20  0  0  0  0  0  0  6  0

        num_alts = len(tdd_alts.index)
        num_persons = len(persons.index)

        person_ids = pd.Series(list(range(num_persons)) * num_alts)
        tdds = pd.Series(np.repeat(list(range(num_alts)), num_persons))

        assert timetable.tour_available(person_ids, tdds).all()

        person_ids = pd.Series([0, 1, 2, 3, 4, 5])
        tdds = pd.Series([0, 1, 2, 15, 16, 17])
        timetable.assign(person_ids, tdds)

        # print "\nupdated_person_windows\n", timetable.get_person_windows_df()
        #    4  5  6  7  8  9  10  11
        # 0  0  6  0  0  0  0   0   0
        # 1  0  2  4  0  0  0   0   0
        # 2  0  2  7  4  0  0   0   0
        # 3  0  0  0  0  6  0   0   0
        # 4  0  0  0  0  2  4   0   0
        # 5  0  0  0  0  2  7   4   0

        person_ids = pd.Series([0, 1, 1, 0, 1, 3, 4])
        tdds = pd.Series(
            [
                0,  # tdd START_END does not collide with START_END
                0,  # tdd START_END does not collide with START
                6,  # tdd START_END does not collide with END
                1,  # tdd START does not collide with START_END
                7,  # tdd START does not collide with END
                3,  # tdd END does not collide with START_END
                3,  # tdd END does not collide with START
            ]
        )
        assert timetable.tour_available(person_ids, tdds).all()

        # print "\nupdated_person_windows\n", timetable.get_person_windows_df()
        #    4  5  6  7  8  9  10  11
        # 0  0  6  0  0  0  0   0   0
        # 1  0  2  4  0  0  0   0   0
        # 2  0  2  7  4  0  0   0   0
        # 3  0  0  0  0  6  0   0   0
        # 4  0  0  0  0  2  4   0   0
        # 5  0  0  0  0  2  7   4   0

        person_ids = pd.Series([1, 5, 2, 2])
        tdds = pd.Series(
            [
                1,  # tdd START + END collides with START + END
                17,  # START + MIDDLE + END collides with same
                6,  # tdd START_END collides with MIDDLE
                1,  # tdd START + END collides with START + MIDDLE
            ]
        )
        assert not timetable.tour_available(person_ids, tdds).any()

        # ensure that tour_available handles heterogeneous results
        person_ids = pd.Series([0, 1, 1, 5])
        tdds = pd.Series(
            [
                0,  # tdd START_END does not collide with START_END
                0,  # tdd START_END does not collide with START
                1,  # tdd START + END collides with START + END
                17,  # START + MIDDLE + END collides with same
            ]
        )
        pdt.assert_series_equal(
            timetable.tour_available(person_ids, tdds),
            pd.Series([True, True, False, False], index=person_ids.index),
        )

        # assigning overlapping trip END,START should convert END to START_END
        person_ids = pd.Series([2])
        tdds = pd.Series([13])
        assert timetable.tour_available(person_ids, tdds).all()
        assert timetable.windows[2, 3] == tt.I_END
        timetable.assign(person_ids, tdds)
        assert timetable.windows[2, 3] == tt.I_START_END

        # print "\nupdated_person_windows\n", timetable.get_person_windows_df()
        #    4  5  6  7  8  9  10  11
        # 0  0  6  0  0  0  0   0   0
        # 1  0  2  4  0  0  0   0   0
        # 2  0  2  7  6  7  4   0   0
        # 3  0  0  0  0  6  0   0   0
        # 4  0  0  0  0  2  4   0   0
        # 5  0  0  0  0  2  7   4   0

        # - previous_tour_ends

        person_ids = pd.Series([0, 1, 2, 3, 4, 5, 2])
        periods = pd.Series([5, 6, 9, 8, 9, 10, 7])
        assert timetable.previous_tour_ends(person_ids, periods).all()

        person_ids = pd.Series([0, 1, 2])
        periods = pd.Series([9, 5, 8])
        assert not timetable.previous_tour_ends(person_ids, periods).any()

        # - previous_tour_begins

        person_ids = pd.Series([0, 1, 2, 3, 4, 5, 2])
        periods = pd.Series([5, 5, 5, 8, 8, 8, 7])
        assert timetable.previous_tour_begins(person_ids, periods).all()

        person_ids = pd.Series([0, 1, 2])
        periods = pd.Series([9, 6, 8])
        assert not timetable.previous_tour_begins(person_ids, periods).any()

        # - adjacent_window_after
        person_ids = pd.Series([0, 1, 2, 3, 4, 5])
        periods = pd.Series([5, 5, 5, 5, 5, 5])
        adjacent_run_length = timetable.adjacent_window_after(person_ids, periods)
        pdt.assert_series_equal(adjacent_run_length, pd.Series([5, 5, 0, 5, 5, 3]))

        # - adjacent_window_before
        person_ids = pd.Series([0, 1, 2, 3, 4, 5])
        periods = pd.Series([10, 10, 10, 10, 10, 10])
        adjacent_run_length = timetable.adjacent_window_before(person_ids, periods)
        pdt.assert_series_equal(adjacent_run_length, pd.Series([5, 5, 1, 5, 5, 0]))

        # - remaining_periods_available
        person_ids = pd.Series([0, 1, 2, 3])
        starts = pd.Series([9, 6, 9, 5])
        ends = pd.Series([10, 10, 10, 9])
        periods_available = timetable.remaining_periods_available(
            person_ids, starts, ends
        )
        pdt.assert_series_equal(
            periods_available, pd.Series([6, 3, 4, 3]), check_dtype=False
        )
