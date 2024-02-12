# ActivitySim
# See full license in LICENSE.txt.


import pandas as pd
import pandas.testing as pdt

from activitysim.abm.models.util.tour_frequency import process_non_mandatory_tours
from activitysim.core import workflow


def test_nmtf():
    state = workflow.State.make_default(__file__)

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
    nmt = process_non_mandatory_tours(state, persons, tour_counts)

    idx = nmt.index

    pdt.assert_series_equal(
        nmt.person_id, pd.Series([0, 2, 2, 3], index=idx, name="person_id")
    )

    # check if the tour_type variable is pandas categorical
    if isinstance(nmt.tour_type.dtype, pd.api.types.CategoricalDtype):
        pdt.assert_series_equal(
            nmt.tour_type.astype(str),
            pd.Series(
                ["shopping", "escort", "escort", "othmaint"],
                index=idx,
                name="tour_type",
            ),
        )
    else:
        pdt.assert_series_equal(
            nmt.tour_type,
            pd.Series(
                ["shopping", "escort", "escort", "othmaint"],
                index=idx,
                name="tour_type",
            ),
        )
