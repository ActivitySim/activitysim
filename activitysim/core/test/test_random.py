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

CHANNEL_TYPES = ("simple", "fast", "faster")

_FAST_CHANNEL_GOLDENS = {
    "fast": {
        "uniform": np.array(
            [
                [0.8412170922705721, 0.8444598643290162],
                [0.4851147851611052, 0.31191389673821024],
                [0.749397127522066, 0.05566740290330263],
            ]
        ),
        "normal": np.array(
            [0.11495552520758334, -0.25428019721944395, -1.3766521717537907]
        ),
        "choice": np.array([10, 50, 10, 20, 40, 50]),
        "following": np.array(
            [
                [0.431261375505242],
                [0.47379130909715983],
                [0.23522137228134843],
            ]
        ),
    },
    "faster": {
        "uniform": np.array(
            [
                [0.35325693076713094, 0.5673606151472527],
                [0.8601996056684649, 0.5302840828934835],
                [0.09654885478368846, 0.7308784020261461],
            ]
        ),
        "normal": np.array([1.8144041669988935, 1.0751733769824134, 1.052361487133243]),
        "choice": np.array([10, 40, 30, 20, 20, 50]),
        "following": np.array(
            [
                [0.0362199624887537],
                [0.5978910808097496],
                [0.6610301299479874],
            ]
        ),
    },
}


def test_basic():
    rng = random.Random()
    assert rng.channel_type == "simple"

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


@pytest.mark.parametrize("channel_type", ("fast", "faster"))
def test_fast_channel_mixed_sequence_matches_golden(channel_type):
    """Freeze configured fast-channel streams and cross-method consumption."""
    persons = pd.DataFrame(index=pd.Index([101, 202, 303], name="person_id"))
    requested = persons.loc[[303, 101, 202]]
    rng = random.Random(channel_type=channel_type)
    rng.set_base_seed(17)
    rng.begin_step("golden_step")
    rng.add_channel("persons", persons)

    observed = {
        "uniform": rng.random_for_df(requested, n=2),
        "normal": rng.normal_for_df(requested),
        "choice": rng.choice_for_df(
            requested, np.array([10, 20, 30, 40, 50]), 2, replace=False
        ),
        "following": rng.random_for_df(requested),
    }
    rng.end_step("golden_step")

    for name, expected in _FAST_CHANNEL_GOLDENS[channel_type].items():
        npt.assert_array_equal(observed[name], expected)


@pytest.mark.parametrize("channel_type", ("fast", "faster"))
def test_fast_channel_recreation_preserves_subset_and_extension_streams(channel_type):
    """Recreating a channel at a step boundary must preserve every row stream."""

    def run(domain_order, request_order):
        persons = pd.DataFrame(index=pd.Index(domain_order, name="person_id"))
        requested = persons.loc[request_order]
        rng = random.Random(channel_type=channel_type)
        rng.set_base_seed(17)
        rng.begin_step("resume_step")
        rng.add_channel("persons", persons)
        uniform = rng.random_for_df(requested, n=2)
        normal = rng.normal_for_df(requested)

        added = pd.DataFrame(index=pd.Index([404], name="person_id"))
        rng.add_channel("persons", added)
        extended = rng.random_for_df(added, n=2)
        rng.end_step("resume_step")
        return uniform, normal, extended

    original = run([101, 202, 303], [303, 101])
    recreated = run([303, 202, 101], [101, 303])

    # The requested row order changes between runs, so reverse those results
    # before comparing. The newly extended row is independent of domain order.
    npt.assert_array_equal(original[0], recreated[0][::-1])
    npt.assert_array_equal(original[1], recreated[1][::-1])
    npt.assert_array_equal(original[2], recreated[2])


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
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
        test1_expected_rands = [0.4072658, 0.5591271, 0.0297283, 0.6235138, 0.6921163]
    elif channel_type == "faster":
        test1_expected_rands = [0.4580108, 0.531716, 0.6470319, 0.6762532, 0.7392374]
    else:
        test1_expected_rands = [0.1733218, 0.1255693, 0.7384256, 0.3485183, 0.9012387]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands)

    # second call should return something different
    rands = rng.random_for_df(persons)
    if channel_type == "fast":
        test1_expected_rands2 = [0.336963, 0.5420581, 0.4396565, 0.9702927, 0.0251327]
    elif channel_type == "faster":
        test1_expected_rands2 = [0.1690983, 0.933964, 0.3887059, 0.7922818, 0.4179632]
    else:
        test1_expected_rands2 = [0.9105223, 0.5718418, 0.7222742, 0.9062284, 0.3929369]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), test1_expected_rands2)

    rng.end_step("test_step")

    rng.begin_step("test_step2")

    rands = rng.random_for_df(households)
    if channel_type == "fast":
        expected_rands = [0.1571023, 0.2709219, 0.2515827, 0.9444831, 0.6816792]
    elif channel_type == "faster":
        expected_rands = [0.1934219, 0.3369451, 0.8455883, 0.6440651, 0.3889942]
    else:
        expected_rands = [0.417278, 0.2994774, 0.8653719, 0.4429748, 0.5101697]
    npt.assert_almost_equal(np.asanyarray(rands).flatten(), expected_rands)

    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    if channel_type == "fast":
        expected_choices = [4, 1, 4, 3, 2, 1, 3, 1, 1, 4]
    elif channel_type == "faster":
        expected_choices = [3, 4, 4, 3, 4, 2, 4, 1, 2, 3]
    else:
        expected_choices = [2, 1, 3, 3, 4, 2, 4, 1, 4, 1]
    npt.assert_almost_equal(choices, expected_choices)

    # should be DIFFERENT the second time
    choices = rng.choice_for_df(households, [1, 2, 3, 4], 2, replace=True)
    if channel_type == "fast":
        expected_choices = [1, 4, 2, 1, 2, 3, 1, 2, 2, 4]
    elif channel_type == "faster":
        expected_choices = [4, 1, 3, 3, 4, 1, 4, 2, 3, 2]
    else:
        expected_choices = [3, 1, 4, 3, 3, 2, 2, 1, 4, 2]
    npt.assert_almost_equal(choices, expected_choices)

    rng.end_step("test_step2")

    rng.begin_step("test_step3")

    rands = rng.random_for_df(households, n=2)

    if channel_type == "fast":
        expected_rands = [
            0.0728735,
            0.9764697,
            0.6611142,
            0.8802973,
            0.0122184,
            0.8770089,
            0.9944639,
            0.2064867,
            0.6051138,
            0.1666114,
        ]
    elif channel_type == "faster":
        expected_rands = [
            0.2677105,
            0.7688408,
            0.9949042,
            0.909176,
            0.9348486,
            0.069542,
            0.7039883,
            0.89629,
            0.7469927,
            0.3387263,
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


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_reset_offsets_for_step_replays_all_rows(channel_type):
    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    rng = random.Random(channel_type=channel_type)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)

    first = rng.random_for_df(persons)
    rng.random_for_df(persons)
    rng.reset_offsets_for_step("test_step")
    replay = rng.random_for_df(persons)

    npt.assert_array_equal(replay, first)


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_reset_offsets_for_df_replays_only_selected_rows(channel_type):
    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    selected = persons.loc[[1, 3]]
    unselected = persons.loc[[2]]

    rng = random.Random(channel_type=channel_type)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)
    first = rng.random_for_df(persons)
    rng.random_for_df(persons)
    rng.reset_offsets_for_df(selected)

    replay = rng.random_for_df(selected)
    unselected_next = rng.random_for_df(unselected)

    baseline = random.Random(channel_type=channel_type)
    baseline.begin_step("test_step")
    baseline.add_channel("persons", persons)
    baseline.random_for_df(persons)
    baseline.random_for_df(persons)
    expected_unselected_next = baseline.random_for_df(unselected)

    npt.assert_array_equal(replay, first[[0, 2]])
    npt.assert_array_equal(unselected_next, expected_unselected_next)


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_normal_for_df_preserves_per_row_shapes(channel_type):
    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    mu = np.array([10.0, 20.0, 30.0])
    sigma = np.zeros(3)
    rng = random.Random(channel_type=channel_type)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)

    scalar_draws = rng.normal_for_df(persons, mu=mu, sigma=sigma)
    vector_draws = rng.normal_for_df(persons, mu=mu, sigma=sigma, size=2)
    lognormal_draws = rng.lognormal_for_df(persons, mu=mu, sigma=sigma)

    assert scalar_draws.shape == (3,)
    assert vector_draws.shape == (3, 2)
    assert lognormal_draws.shape == (3,)
    npt.assert_array_equal(scalar_draws, mu)
    npt.assert_array_equal(vector_draws, np.repeat(mu[:, None], 2, axis=1))
    npt.assert_array_equal(lognormal_draws, np.exp(mu))


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_lognormal_for_df_broadcasts_over_duplicate_rows(channel_type):
    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    alternatives = pd.DataFrame(index=pd.Index([1, 1, 2, 3, 3], name="person_id"))
    rng = random.Random(channel_type=channel_type)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)

    draws = rng.lognormal_for_df(
        alternatives,
        mu=1.0,
        sigma=0.5,
        broadcast=True,
    )

    assert draws.shape == (5,)
    assert draws.iloc[0] == draws.iloc[1]
    assert draws.iloc[3] == draws.iloc[4]


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_max_positions_for_df_matches_materialized_path_and_offsets(
    channel_type,
):
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

    baseline_rng = random.Random(channel_type=channel_type)
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

    fused_rng = random.Random(channel_type=channel_type)
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_max_positions_for_df(utilities, sample_size)
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_max_positions_for_df_matches_stable_alt_mapping_and_offsets(
    channel_type,
):
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

    baseline_rng = random.Random(channel_type=channel_type)
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

    fused_rng = random.Random(channel_type=channel_type)
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


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_random_for_df_stable_alt_mapping_and_offsets(channel_type):
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

    baseline_rng = random.Random(channel_type=channel_type)
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.random_for_df(active_alts, n=n_total_alts)
    expected_rands = materialized[:, stable_alt_positions]
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random(channel_type=channel_type)
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


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_choice_positions_for_df_matches_materialized_path_and_offsets(
    channel_type,
):
    persons = pd.DataFrame(
        {"household_id": [1, 1, 2]},
        index=pd.Index([21, 22, 23], name="person_id"),
    )
    utilities = pd.DataFrame(
        [[0.5, -0.2, 1.1], [0.1, 0.2, -0.3], [2.0, 1.0, 0.0]],
        index=persons.index,
    )

    baseline_rng = random.Random(channel_type=channel_type)
    baseline_rng.set_base_seed(0)
    baseline_rng.begin_step("test_step")
    baseline_rng.add_channel("persons", persons)

    materialized = baseline_rng.gumbel_for_df(utilities, n=utilities.shape[1])
    expected_positions = np.argmax(materialized + utilities.to_numpy(), axis=1)
    next_random_after_materialized = baseline_rng.random_for_df(persons)
    baseline_rng.end_step("test_step")

    fused_rng = random.Random(channel_type=channel_type)
    fused_rng.set_base_seed(0)
    fused_rng.begin_step("test_step")
    fused_rng.add_channel("persons", persons)

    observed_positions = fused_rng.gumbel_choice_positions_for_df(utilities)
    next_random_after_fused = fused_rng.random_for_df(persons)
    fused_rng.end_step("test_step")

    npt.assert_array_equal(observed_positions, expected_positions)
    npt.assert_allclose(next_random_after_fused, next_random_after_materialized)


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_choice_positions_for_df_matches_dense_alt_mapping(channel_type):
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

    baseline_rng = random.Random(channel_type=channel_type)
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

    fused_rng = random.Random(channel_type=channel_type)
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


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_choice_positions_for_df_masked_columns_never_win(channel_type):
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

    rng = random.Random(channel_type=channel_type)
    rng.set_base_seed(0)
    rng.begin_step("test_step")
    rng.add_channel("persons", persons)
    positions = rng.gumbel_choice_positions_for_df(
        utilities, alt_nrs_df=alt_nrs_df, n_rands=3
    )
    rng.end_step("test_step")

    npt.assert_array_equal(positions, [0, 1, 2])


@pytest.mark.parametrize("channel_type", CHANNEL_TYPES)
def test_gumbel_choice_positions_for_df_fully_masked_row_falls_back_to_first_column(
    channel_type,
):
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
        rng = random.Random(channel_type=channel_type)
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
