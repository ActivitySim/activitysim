# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_simulate, workflow


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_simulate_explicit_error_terms_parity(state):
    # Run interaction_simulate with and without explicit error terms and check that results are similar.

    # Keep this large enough for stable parity checks without overloading CI.
    num_choosers = 100_000
    num_alts = 5
    sample_size = num_alts

    # Create random choosers and alternatives
    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    # Run _without_ explicit error terms
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)  # Set seed BEFORE adding channels or steps
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_mnl")

    choices_mnl = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
    )

    # Run _with_ explicit error terms
    state.init_state()  # reset the state to rerun with same seed
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_explicit")

    choices_explicit = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
    )

    assert len(choices_mnl) == num_choosers
    assert len(choices_explicit) == num_choosers
    assert choices_mnl.index.equals(choosers.index)
    assert choices_explicit.index.equals(choosers.index)
    assert not choices_mnl.isna().any()
    assert not choices_explicit.isna().any()

    mnl_counts = choices_mnl.value_counts(normalize=True).sort_index()
    explicit_counts = choices_explicit.value_counts(normalize=True).sort_index()

    # Check that they are close, relative to the number of draws
    assert np.allclose(
        mnl_counts.to_numpy(), explicit_counts.to_numpy(), atol=0.01, rtol=0.001
    )


def test_interaction_simulate_eet_unavailable_alternatives(state):
    # Test that EET handles unavailable alternatives (very low utilities)
    # similarly to MNL (zero probabilities).

    num_choosers = 100
    num_alts = 5

    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # Alt 0 and 1 are attractive, Alt 2, 3, 4 are "unavailable" (very low utility)
    alternatives = pd.DataFrame(
        {"alt_attr": [10.0, 10.0, -1000.0, -1000.0, -1000.0]},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    # Run with EET
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_unavailable_eet")

    choices_eet = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=num_alts,
    )

    assert len(choices_eet) == num_choosers
    assert choices_eet.index.equals(choosers.index)
    assert not choices_eet.isna().any()

    # Choices should only be from Alt 0 or 1
    assert choices_eet.isin(
        [0, 1]
    ).all(), f"EET picked an 'unavailable' alternative: {choices_eet[~choices_eet.isin([0, 1])]}"


def test_interaction_simulate_eet_large_utilities(state):
    # Test that EET handles very large utilities without overflow issues
    # that might occur in exp(util) calculations in standard MNL.

    num_choosers = 10
    num_alts = 2

    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # Standard MNL might struggle with exp(700) or exp(800) depending on float precision
    alternatives = pd.DataFrame(
        {"alt_attr": [700.0, 800.0]},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_large_utils_eet")

    # This should run without crashing or returning NaNs
    choices_eet = interaction_simulate.interaction_simulate(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=num_alts,
    )

    assert not choices_eet.isna().any()
    # With such a large difference, Alt 1 should be the dominant choice
    assert (choices_eet == 1).all()
