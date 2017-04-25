# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd
import pandas.util.testing as pdt
import pytest

from ..util import reindex
from ..util import other_than


@pytest.fixture(scope='module')
def people():
    return pd.DataFrame({
        'household': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'ptype':     [1, 2, 1, 3, 1, 2, 3, 2, 2, 1]},
        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])


def test_other_than(people):
    expected = pd.Series(
        [False, False, True, True, True, False, True, True, True, True],
        index=people.index, name='left')

    bools = people['ptype'] == 2
    others = other_than(people['household'], bools)

    pdt.assert_series_equal(others, expected)


def test_reindex():
    s = pd.Series([.5, 1.0, 1.5], index=[2, 1, 3])
    s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    assert list(reindex(s, s2).values) == [1.0, .5, 1.5]
