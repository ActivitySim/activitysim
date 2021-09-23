# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from ..util import other_than, quick_loc_df, quick_loc_series, reindex


@pytest.fixture(scope="module")
def people():
    return pd.DataFrame(
        {
            "household": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            "ptype": [1, 2, 1, 3, 1, 2, 3, 2, 2, 1],
        },
        index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )


def test_other_than(people):
    expected = pd.Series(
        [False, False, True, True, True, False, True, True, True, True],
        index=people.index,
        name="left",
    )

    bools = people["ptype"] == 2
    others = other_than(people["household"], bools)

    pdt.assert_series_equal(others, expected)


def test_reindex():
    s = pd.Series([0.5, 1.0, 1.5], index=[2, 1, 3])
    s2 = pd.Series([1, 2, 3], index=["a", "b", "c"])
    assert list(reindex(s, s2).values) == [1.0, 0.5, 1.5]


def test_quick_loc_df():

    df = pd.DataFrame({"attrib": ["1", "2", "3", "4", "5"]}, index=[1, 2, 3, 4, 5])

    loc_list = np.asanyarray([2, 1, 3, 4, 4, 5, 1])
    attrib_list = [str(i) for i in loc_list]

    assert list(quick_loc_df(loc_list, df, "attrib")) == attrib_list
    assert list(quick_loc_df(loc_list, df, "attrib")) == list(
        df.loc[loc_list]["attrib"]
    )


def test_quick_loc_series():

    series = pd.Series(["1", "2", "3", "4", "5"], index=[1, 2, 3, 4, 5])

    loc_list = np.asanyarray([2, 1, 3, 4, 4, 5, 1])
    attrib_list = [str(i) for i in loc_list]

    assert list(quick_loc_series(loc_list, series)) == attrib_list
    assert list(quick_loc_series(loc_list, series)) == list(series.loc[loc_list])
