# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_sample_simulate, workflow
from activitysim.core.logit import AltsContext


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_sample_simulate_parity(state):
    # Run interaction_sample_simulate with and without explicit error terms and check that results are similar.

    num_choosers = 100_000
    num_alts_per_chooser = 5  # small sample size to keep things simple

    # Create random choosers
    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # Create random alternatives for each chooser
    # In interaction_sample_simulate, alternatives is typically a DataFrame with the same index as choosers
    # but repeated for each alternative in the sample.
    alt_ids = np.tile(np.arange(num_alts_per_chooser), num_choosers)
    alternatives = pd.DataFrame(
        {
            "alt_attr": rng.random(num_choosers * num_alts_per_chooser),
            "alt_id": alt_ids,
            "tdd": alt_ids,
        },
        index=np.repeat(choosers.index, num_alts_per_chooser),
    )
    alternatives.index.name = "person_id"

    # Simple spec: utility = chooser_attr * alt_attr
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    # Run _without_ explicit error terms
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_mnl")

    choices_mnl = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="tdd",
    )

    # Run _with_ explicit error terms
    state.init_state()
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_explicit")

    choices_explicit = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="tdd",
        alts_context=AltsContext.from_num_alts(num_alts_per_chooser, zero_based=True),
    )

    assert len(choices_mnl) == num_choosers
    assert len(choices_explicit) == num_choosers
    assert choices_mnl.index.equals(choosers.index)
    assert choices_explicit.index.equals(choosers.index)
    assert not choices_mnl.isna().any()
    assert not choices_explicit.isna().any()

    # choices are series with the same index as choosers and containing the choice (from choice_column)
    mnl_counts = choices_mnl.value_counts(normalize=True).sort_index()
    explicit_counts = choices_explicit.value_counts(normalize=True).sort_index()

    for alt in range(num_alts_per_chooser):
        share_mnl = mnl_counts.get(alt, 0)
        share_explicit = explicit_counts.get(alt, 0)
        diff = abs(share_mnl - share_explicit)
        assert diff < 0.01, (
            f"Large discrepancy at alt {alt}: "
            f"mnl={share_mnl:.4f}, explicit={share_explicit:.4f}, diff={diff:.4f}"
        )


def test_interaction_sample_simulate_eet_unavailable_alternatives(state):
    # Test that EET handles unavailable alternatives in sample simulation

    num_choosers = 10
    num_alts_per_chooser = 5

    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # For each chooser, 2 attractive alts, 3 unavailable
    alt_attrs = [10.0, 10.0, -1000.0, -1000.0, -1000.0] * num_choosers
    alt_ids = [0, 1, 2, 3, 4] * num_choosers

    alternatives = pd.DataFrame(
        {
            "alt_attr": alt_attrs,
            "alt_id": alt_ids,
            "tdd": alt_ids,
        },
        index=np.repeat(choosers.index, num_alts_per_chooser),
    )
    alternatives.index.name = "person_id"

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    # Run with EET
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_unavailable_eet")

    choices_eet = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="tdd",
        alts_context=AltsContext.from_num_alts(num_alts_per_chooser, zero_based=True),
    )

    assert len(choices_eet) == num_choosers
    assert choices_eet.index.equals(choosers.index)
    assert not choices_eet.isna().any()

    # Choices should only be 0 or 1
    assert choices_eet.isin([0, 1]).all()
    assert not choices_eet.isin([2, 3, 4]).any()


def test_interaction_sample_simulate_passes_alts_context_and_alt_nrs_df(
    state, monkeypatch
):
    state.settings.use_explicit_error_terms = True

    choosers = pd.DataFrame(
        {"chooser_attr": [1.0, 1.0]},
        index=pd.Index([100, 101], name="person_id"),
    )
    alternatives = pd.DataFrame(
        {
            "alt_attr": [1.0, 0.5, 0.8, 1.2],
            "tdd": [0, 2, 0, 2],
        },
        index=pd.Index([100, 100, 101, 101], name="person_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    captured = {}

    def fake_make_choices_utility_based(
        _state,
        utilities,
        nest_spec=None,
        trace_label=None,
        trace_choosers=None,
        alts_context=None,
        alt_nrs_df=None,
    ):
        captured["alts_context"] = alts_context
        captured["alt_nrs_df"] = alt_nrs_df.copy() if alt_nrs_df is not None else None
        return pd.Series([0, 0], index=utilities.index), pd.Series(
            np.zeros(len(utilities.index)), index=utilities.index
        )

    monkeypatch.setattr(
        interaction_sample_simulate.logit,
        "make_choices_utility_based",
        fake_make_choices_utility_based,
    )

    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_alts_context_forwarding")

    ctx = AltsContext.from_num_alts(3, zero_based=True)
    choices = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="tdd",
        alts_context=ctx,
    )

    assert len(choices) == len(choosers)
    assert captured["alts_context"] == ctx
    assert captured["alt_nrs_df"] is not None
    expected_alt_nrs = pd.DataFrame(
        [[0, 2], [0, 2]],
        index=choosers.index,
    )
    pd.testing.assert_frame_equal(captured["alt_nrs_df"], expected_alt_nrs)
