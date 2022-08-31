# ActivitySim
# See full license in LICENSE.txt.
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from activitysim.core import random


def test_basic():

    rng = random.Random()

    rng.set_base_seed(0)

    rng.begin_step("test_step")

    global_rng = rng.get_global_rng()

    npt.assert_almost_equal(global_rng.rand(1), [0.8994663])

    # second call should return something different
    with pytest.raises(AssertionError) as excinfo:
        npt.assert_almost_equal(global_rng.rand(1), [0.8994663])
    assert "Arrays are not almost equal" in str(excinfo.value)

    # second call should return something different
    with pytest.raises(RuntimeError) as excinfo:
        rng.set_base_seed(1)
    assert "call set_base_seed before the first step" in str(excinfo.value)


def test_channel():

    channels = {
        "households": "household_id",
        "persons": "person_id",
    }
    rng = random.Random()

    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2, 2, 2],
        },
        index=[1, 2, 3, 4, 5],
    )
    persons.index.name = "person_id"

    households = pd.DataFrame(
        {
            "data": [1, 1, 2, 2, 2],
        },
        index=[1, 2, 3, 4, 5],
    )
    households.index.name = "household_id"

    rng.begin_step("test_step")

    rng.add_channel("persons", persons)
    rng.add_channel("households", households)

    rands = rng.random_for_df(persons)

    print("rands", np.asanyarray(rands).flatten())

    assert rands.shape == (5, 1)
    test1_expected_rands = [0.1733218, 0.1255693, 0.7384256, 0.3485183, 0.9012387]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    test1_expected_rands2 = [0.9105223, 0.5718418, 0.7222742, 0.9062284, 0.3929369]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step("test_step")

    rng.begin_step("test_step2")

    rands = rng.random_for_df(households)
    expected_rands = [0.417278, 0.2994774, 0.8653719, 0.4429748, 0.5101697]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [2, 1, 3, 3, 4, 2, 4, 1, 4, 1]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    expected_choices = [3, 1, 4, 3, 3, 2, 2, 1, 4, 2]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step("test_step2")

    rng.begin_step("test_step3")

    rands = rng.random_for_df(households, n=2)

    expected_rands = [
        0.3157928,
        0.3321823,
        0.5194067,
        0.9340083,
        0.9002048,
        0.8754209,
        0.3898816,
        0.4101094,
        0.7351484,
        0.1741092,
    ]

    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    rng.end_step("test_step3")

    # if we use the same step name a second time, we should get the same results as before
    rng.begin_step("test_step")

    rands = rng.random_for_df(persons)

    print("rands", np.asanyarray(rands).flatten())
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    rands = rng.random_for_df(persons)
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step("test_step")
