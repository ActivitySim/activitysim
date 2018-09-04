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
        'households': 'household_id',
        'persons': 'person_id',
    }
    rng = random.Random()

    persons = pd.DataFrame({
        "household_id": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    persons.index.name = 'person_id'

    households = pd.DataFrame({
        "data": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    households.index.name = 'household_id'

    rng.begin_step('test_step')

    rng.add_channel(persons, channel_name='persons')
    rng.add_channel(households, channel_name='households')

    rands = rng.random_for_df(persons)

    print "rands", np.asanyarray(rands).flatten()

    assert rands.shape == (5, 1)
    test1_expected_rands = [0.9060891, 0.4576382, 0.2154094, 0.2801035, 0.6196645]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    test1_expected_rands2 = [0.5991157, 0.5516594, 0.5529548, 0.3586653, 0.5844314]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step('test_step')

    rng.begin_step('test_step2')

    rands = rng.random_for_df(households)
    expected_rands = [0.7970902, 0.2633469, 0.7662205, 0.7544782, 0.129741]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [2, 1, 3, 1, 3, 1, 3, 4, 4, 2]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [1, 1, 3, 2, 3, 2, 2, 3, 2, 3]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step('test_step2')

    rng.begin_step('test_step3')

    rands = rng.random_for_df(households, n=2)

    expected_rands = [0.8635927, 0.3258157, 0.7970902, 0.365523, 0.2633469, 0.5388047,
                      0.7662205, 0.8067344, 0.7544782, 0.024577]

    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step('test_step3')

    # if we use the same step name a second time, we should get the same results as before
    rng.begin_step('test_step')

    rands = rng.random_for_df(persons)

    print "rands", np.asanyarray(rands).flatten()
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    rands = rng.random_for_df(persons)
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step('test_step')
