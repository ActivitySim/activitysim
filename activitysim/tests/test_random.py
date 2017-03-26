# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
import pytest

from .. import random
from .. import pipeline


def test_basic():

    rng = random.Random()

    rng.set_base_seed(0)

    rng.begin_step('test_step')

    global_rng = rng.get_global_rng()

    npt.assert_almost_equal(global_rng.rand(1), [0.09237])

    # second call should return something different
    with pytest.raises(AssertionError) as excinfo:
        npt.assert_almost_equal(global_rng.rand(1), [0.09237])
    assert "Arrays are not almost equal" in str(excinfo.value)

    # second call should return something different
    with pytest.raises(RuntimeError) as excinfo:
        rng.set_base_seed(1)
    assert "call set_base_seed before the first step" in str(excinfo.value)


def test_channel():

    rng = random.Random({'households': 3, 'persons': 5})

    persons = pd.DataFrame({
        "household_id": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    persons.index.name = 'PERID'

    rng.begin_step('test_step')

    rng.add_channel(persons, channel_name='persons')

    rands = rng.random_for_df(persons)

    expected_rands = [0.0206057, 0.6934178, 0.6821042, 0.4591678, 0.1561672]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    households = pd.DataFrame({
        "data": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    households.index.name = 'HHID'
    rng.add_channel(households, channel_name='households', step_name='last', step_num=1)

    rands = rng.random_for_df(households)

    expected_rands = [0.8848691, 0.7472187, 0.5943395, 0.833949, 0.7222694]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)
