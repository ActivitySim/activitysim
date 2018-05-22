# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
import pytest

from activitysim.core import random
from activitysim.core import pipeline


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

    channels = {
        'households': 'HHID',
        'persons': 'PERID',
    }
    rng = random.Random()
    rng.set_channel_info(channels)

    persons = pd.DataFrame({
        "household_id": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    persons.index.name = 'PERID'

    households = pd.DataFrame({
        "data": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    households.index.name = 'HHID'

    rng.begin_step('test_step')

    rng.add_channel(persons, channel_name='persons')
    rng.add_channel(households, channel_name='households')

    rands = rng.random_for_df(persons)

    assert rands.shape == (5, 1)
    expected_rands = [0.0305274, 0.6452407, 0.1686045, 0.9529088, 0.1994755]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    expected_rands = [0.9912599, 0.5523497, 0.4580549, 0.3668453, 0.134653]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step('test_step')

    rng.begin_step('test_step2')

    rands = rng.random_for_df(households)
    expected_rands = [0.7992435, 0.5682545, 0.8956348, 0.6326098, 0.630408]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [1, 3, 3, 2, 1, 1, 1, 3, 1, 3]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [2, 3, 3, 3, 2, 2, 2, 2, 2, 4]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step('test_step2')

    rng.begin_step('test_step3')

    rands = rng.random_for_df(households, n=2)

    expected_rands = [0.4633051, 0.4924085, 0.8627697, 0.854059, 0.0689231,
                      0.3818341, 0.0301041, 0.7765588, 0.2082694, 0.4542789]

    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step('test_step3')
