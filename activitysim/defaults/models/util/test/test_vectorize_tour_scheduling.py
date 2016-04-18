# ActivitySim
# See full license in LICENSE.txt.


import pytest
import pandas as pd
import numpy as np
import pandas.util.testing as pdt
from ..vectorize_tour_scheduling import get_previous_tour_by_tourid, \
    vectorize_tour_scheduling


def test_vts():

    np.random.seed(0)

    alts = pd.DataFrame({
        "start": [1, 2, 3],
        "end": [4, 5, 6],
    }, index=[10, 20, 30])

    current_tour_person_ids = pd.Series(['b', 'c'],
                                        index=['d', 'e'])

    previous_tour_by_personid = pd.Series([20, 20, 10],
                                          index=['a', 'b', 'c'])

    prev_tour_attrs = get_previous_tour_by_tourid(current_tour_person_ids,
                                                  previous_tour_by_personid,
                                                  alts)

    pdt.assert_series_equal(
        prev_tour_attrs.start_previous,
        pd.Series([2, 1], index=['d', 'e'], name='start_previous'))

    pdt.assert_series_equal(
        prev_tour_attrs.end_previous,
        pd.Series([5, 4], index=['d', 'e'], name='end_previous'))

    tours = pd.DataFrame({
        "person_id": [1, 1, 2, 3, 3],
        "income": [20, 20, 30, 25, 25]
    })

    spec = pd.DataFrame({"Coefficient": [1.2]},
                        index=["income"])
    spec.index.name = "Expression"

    choices = vectorize_tour_scheduling(tours, alts, spec)

    # there's no real logic here - this is just what came out of the monte carlo
    # note that the result comes out ordered by the nth trips and not ordered
    # by the trip index.  shrug?
    pdt.assert_series_equal(
        choices,
        pd.Series([20, 20, 30, 30, 20],
                  index=pd.Index([0, 2, 3, 1, 4], name='index')))
