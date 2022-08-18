# ActivitySim
# See full license in LICENSE.txt.


import os

import pandas as pd
import pandas.testing as pdt
import pytest

from ..tour_frequency import process_non_mandatory_tours


def test_nmtf():

    persons = pd.DataFrame(
        {
            "non_mandatory_tour_frequency": [0, 3, 2, 1],
            "household_id": [1, 1, 2, 4],
            "home_zone_id": [100, 100, 200, 400],
        },
        index=[0, 1, 2, 3],
    )

    non_mandatory_tour_frequency_alts = pd.DataFrame(
        {"escort": [0, 0, 2, 0], "shopping": [1, 0, 0, 0], "othmaint": [0, 1, 0, 0]},
        index=[0, 1, 2, 3],
    )

    tour_counts = non_mandatory_tour_frequency_alts.loc[
        persons.non_mandatory_tour_frequency
    ]
    tour_counts.index = persons.index  # assign person ids to the index

    # - create the non_mandatory tours
    nmt = process_non_mandatory_tours(persons, tour_counts)

    idx = nmt.index

    pdt.assert_series_equal(
        nmt.person_id, pd.Series([0, 2, 2, 3], index=idx, name="person_id")
    )

    pdt.assert_series_equal(
        nmt.tour_type,
        pd.Series(
            ["shopping", "escort", "escort", "othmaint"], index=idx, name="tour_type"
        ),
    )
