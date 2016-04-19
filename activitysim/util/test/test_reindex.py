# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd

from ..reindex import reindex


def test_reindex():
    s = pd.Series([.5, 1.0, 1.5], index=[2, 1, 3])
    s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    assert list(reindex(s, s2).values) == [1.0, .5, 1.5]
