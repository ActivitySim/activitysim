# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from activitysim.core import random
from activitysim.core.exceptions import DuplicateLoadableObjectError


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
    with pytest.raises(DuplicateLoadableObjectError) as excinfo:
        rng.set_base_seed(1)
    assert "call set_base_seed before the first step" in str(excinfo.value)


@pytest.mark.parametrize("channel_type", ["simple", "fast", "faster"])
def test_channel(channel_type: Literal["simple", "fast", "faster"]):
    channels = {
        "households": "household_id",
        "persons": "person_id",
    }
    rng = random.Random(channel_type=channel_type)

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
    if channel_type == "fast":
        test1_expected_rands = [0.162561, 0.0355962, 0.0295559, 0.2415235, 0.755611]
    elif channel_type == "faster":
        test1_expected_rands = [0.1461754, 0.3404831, 0.2687972, 0.5384345, 0.8583329]
    else:
        test1_expected_rands = [0.1733218, 0.1255693, 0.7384256, 0.3485183, 0.9012387]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    if channel_type == "fast":
        test1_expected_rands2 = [0.7580942, 0.7115404, 0.2006981, 0.1460596, 0.5890839]
    elif channel_type == "faster":
        test1_expected_rands2 = [0.023297, 0.3941741, 0.2690284, 0.4423097, 0.9897082]
    else:
        test1_expected_rands2 = [0.9105223, 0.5718418, 0.7222742, 0.9062284, 0.3929369]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step("test_step")

    rng.begin_step("test_step2")

    rands = rng.random_for_df(households)
    if channel_type == "fast":
        expected_rands = [0.9603448, 0.2157038, 0.1521092, 0.1159168, 0.525709]
    elif channel_type == "faster":
        expected_rands = [0.8545137, 0.6644139, 0.0298825, 0.0820411, 0.5859529]
    else:
        expected_rands = [0.417278, 0.2994774, 0.8653719, 0.4429748, 0.5101697]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    if channel_type == "fast":
        expected_choices = [1, 3, 2, 2, 1, 3, 2, 1, 2, 2]
    elif channel_type == "faster":
        expected_choices = [2, 4, 1, 1, 3, 3, 3, 2, 3, 2]
    else:
        expected_choices = [2, 1, 3, 3, 4, 2, 4, 1, 4, 1]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    if channel_type == "fast":
        expected_choices = [4, 2, 1, 3, 1, 3, 3, 1, 2, 3]
    elif channel_type == "faster":
        expected_choices = [3, 3, 3, 4, 3, 1, 4, 1, 1, 4]
    else:
        expected_choices = [3, 1, 4, 3, 3, 2, 2, 1, 4, 2]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step("test_step2")

    rng.begin_step("test_step3")

    rands = rng.random_for_df(households, n=2)

    if channel_type == "fast":
        expected_rands = [
            0.6475412,
            0.2042014,
            0.996377,
            0.7417811,
            0.3613508,
            0.0233316,
            0.1154262,
            0.5316864,
            0.0239487,
            0.1951297,
        ]
    elif channel_type == "faster":
        expected_rands = [
            0.1904426,
            0.5351702,
            0.8730219,
            0.959349,
            0.7729186,
            0.425713,
            0.219114,
            0.9312722,
            0.0218454,
            0.3462854,
        ]
    else:
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


def test_gumbel_max_positions_for_df_matches_materialized_path_and_offsets():
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2]},
        index=pd.Index([11, 12, 13], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[0.5, -0.2, 1.1], [0.1, 0.2, -0.3], [2.0, 1.0, 0.0]],
        index=persons.index,
    )
    sample_size = 4
    n_alts = utilities.shape[1]

    baseline_rng = random.Random()
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.gumbel_for_df(utilities, n=n_alts * sample_size)
    expected_positions = np.argmax(
        materialized.reshape((len(utilities), sample_size, n_alts))
        + utilities.to_numpy()[:, np.newaxis, :],
        axis=2,
    )
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random()
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_max_positions_for_df(utilities, sample_size)
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


def test_gumbel_max_positions_for_df_matches_stable_alt_mapping_and_offsets():
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2]},
        index=pd.Index([41, 42, 43], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[0.5, -0.2, 1.1], [0.1, 0.2, -0.3], [2.0, 1.0, 0.0]],
        index=persons.index,
    )
    sample_size = 3
    stable_alt_positions = np.array([0, 2, 4], dtype=np.int64)
    n_total_alts = 5

    baseline_rng = random.Random()
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.gumbel_for_df(
        utilities,
        n=n_total_alts * sample_size,
    ).reshape((len(utilities), sample_size, n_total_alts))
    expected_positions = np.argmax(
        materialized[:, :, stable_alt_positions]
        + utilities.to_numpy()[:, np.newaxis, :],
        axis=2,
    )
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random()
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_max_positions_for_df(
        utilities,
        sample_size,
        stable_alt_positions=stable_alt_positions,
        n_total_alts=n_total_alts,
    )
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


def test_random_for_df_stable_alt_mapping_and_offsets():
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2]},
        index=pd.Index([51, 52, 53], name="person_id"),
    )
    active_alts = pd.DataFrame(
        np.zeros((len(persons), 3), dtype=np.float64),
        index=persons.index,
    )
    stable_alt_positions = np.array([0, 2, 4], dtype=np.int64)
    n_total_alts = 5

    baseline_rng = random.Random()
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.random_for_df(active_alts, n=n_total_alts)
    expected_rands = materialized[:, stable_alt_positions]
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random()
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_rands = fused_rng.random_for_df_stable_alt_positions(
        active_alts,
        stable_alt_positions=stable_alt_positions,
        n_total_alts=n_total_alts,
    )
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_allclose(observed_rands, expected_rands)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


def test_gumbel_choice_positions_for_df_matches_materialized_path_and_offsets():
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2]},
        index=pd.Index([21, 22, 23], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[0.5, -0.2, 1.1], [0.1, 0.2, -0.3], [2.0, 1.0, 0.0]],
        index=persons.index,
    )

    baseline_rng = random.Random()
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.gumbel_for_df(utilities, n=utilities.shape[1])
    expected_positions = np.argmax(materialized + utilities.to_numpy(), axis=1)
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random()
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_choice_positions_for_df(utilities)
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


def test_gumbel_choice_positions_for_df_matches_dense_alt_mapping():
    persons = pd.DataFrame(
        {"household_id": [1, 1]},
        index=pd.Index([31, 32], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[2.0, 1.0], [0.3, 1.2]],
        index=persons.index,
    )
    alt_nrs_df = pd.DataFrame(
        [[0, 2], [1, 2]],
        index=persons.index,
    )
    n_rands = 3

    baseline_rng = random.Random()
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    dense = baseline_rng.gumbel_for_df(utilities, n=n_rands)
    expected_positions = np.argmax(
        utilities.to_numpy() + np.take_along_axis(dense, alt_nrs_df.to_numpy(), axis=1),
        axis=1,
    )
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random()
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_choice_positions_for_df(
        utilities,
        alt_nrs_df=alt_nrs_df,
        n_rands=n_rands,
    )
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


def test_gumbel_choice_positions_for_df_masked_columns_never_win():
    # padded columns carry a high utility here, so if they were eligible they would
    # win every argmax; only the single active column of each row may be returned
    persons = pd.DataFrame(
        {"household_id": [1, 1, 1]},
        index=pd.Index([41, 42, 43], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[0.0, 99.0, 99.0], [99.0, 0.0, 99.0], [99.0, 99.0, 0.0]],
        index=persons.index,
    )
    alt_nrs_df = pd.DataFrame(
        [
            [0, random.MASKED_ALT_ID, random.MASKED_ALT_ID],
            [random.MASKED_ALT_ID, 1, random.MASKED_ALT_ID],
            [random.MASKED_ALT_ID, random.MASKED_ALT_ID, 2],
        ],
        index=persons.index,
    )

    rng = random.Random()
    rng.set_base_seed(0)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)
    positions = rng.gumbel_choice_positions_for_df(
        utilities, alt_nrs_df=alt_nrs_df, n_rands=3
    )
    rng.end_step("test_step")

    npt.assert_array_equal(positions, [0, 1, 2])


def test_gumbel_choice_positions_for_df_fully_masked_row_falls_back_to_first_column():
    # MASKED_ALT_ID marks padded *or unavailable* slots, so an all-masked row means the
    # chooser has no alternative available. That returns position 0, mirroring the Monte
    # Carlo path's probs.loc[zero_probs, 0] = 1.0, and it must not disturb the choice or
    # the random number stream of any other chooser.
    persons = pd.DataFrame(
        {"household_id": [1, 1, 1]},
        index=pd.Index([51, 52, 53], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[2.0, 1.0], [0.3, 1.2], [0.7, 0.4]],
        index=persons.index,
    )
    all_active = pd.DataFrame([[0, 2], [0, 2], [0, 2]], index=persons.index)
    with_masked_row = pd.DataFrame(
        [[0, 2], [random.MASKED_ALT_ID, random.MASKED_ALT_ID], [0, 2]],
        index=persons.index,
    )

    def run(alt_nrs_df):
        rng = random.Random()
        rng.set_base_seed(0)
        rng.begin_step("test_step")
        rng.add_channel("persons", persons)
        positions = rng.gumbel_choice_positions_for_df(
            utilities, alt_nrs_df=alt_nrs_df, n_rands=3
        )
        following = rng.random_for_df(persons)
        rng.end_step("test_step")
        return positions, following

    baseline_positions, baseline_following = run(all_active)
    masked_positions, masked_following = run(with_masked_row)

    assert masked_positions[1] == 0
    npt.assert_array_equal(masked_positions[[0, 2]], baseline_positions[[0, 2]])
    # the masked row still consumes its n_rands draws, so offsets stay aligned
    npt.assert_allclose(masked_following, baseline_following)
