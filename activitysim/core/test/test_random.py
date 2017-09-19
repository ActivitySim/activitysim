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

    channels = {
        'households': {'max_steps': 4, 'index': 'HHID'},
        'persons': {'max_steps': 2, 'index': 'PERID'},
    }
    rng = random.Random(channels)

    persons = pd.DataFrame({
        "household_id": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    persons.index.name = 'PERID'

    households = pd.DataFrame({
        "data": [1, 1, 2, 2, 2],
    }, index=[1, 2, 3, 4, 5])
    households.index.name = 'HHID'

    rng.begin_step('test_step')

    rng.add_channel(persons, channel_name='persons', step_name='last', step_num=0)
    rng.add_channel(households, channel_name='households')

    rands = rng.random_for_df(persons)

    assert rands.shape == (5, 1)
    expected_rands = [0.9374985, 0.0206057, 0.4684723,  0.246012, 0.700952]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    expected_rands = [0.719677, 0.1214514, 0.7015227, 0.8206436, 0.6126977]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step('test_step')

    rng.begin_step('test_step2')

    # should raise if max_steps exceeded
    with pytest.raises(RuntimeError) as excinfo:
        rands = rng.random_for_df(persons)
    assert "Too many steps" in str(excinfo.value)

    rands = rng.random_for_df(households)
    expected_rands = [0.122587, 0.7472187, 0.4623908, 0.4600264, 0.8385861]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [1, 2, 1, 1, 3, 1, 3, 1, 1, 4]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [1, 4, 2, 2, 1, 1, 3, 2, 2, 2]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step('test_step2')

    rng.begin_step('test_step3')

    rng.set_multi_choice_offset(households, 10)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [3, 4, 2, 4, 1, 2, 3, 3, 1, 2]
    npt.assert_almost_equal(choices, expected_choices)

    # should be SAME second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step('test_step3')

    rng.begin_step('test_step4')

    rands = rng.random_for_df(households, n=2)

    expected_rands = [0.7375634, 0.7714111, 0.8960886, 0.6161022, 0.833949,
                      0.3427474, 0.9498073, 0.1408251, 0.1759239, 0.6410704]

    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step('test_step4')
