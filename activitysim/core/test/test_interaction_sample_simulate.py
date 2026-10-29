# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sharrow as sh

from activitysim.core import interaction_sample, interaction_sample_simulate, workflow
from activitysim.core.configuration.base import ComputeSettings
from activitysim.core.logit import AltsContext


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_global_constants_available_in_sampling_and_simulation(tmp_path):
    """Global constants are available in both destination-choice substeps."""
    configs_dir = tmp_path.joinpath("configs")
    configs_dir.mkdir()
    configs_dir.joinpath("constants.yaml").write_text(
        "SAMPLE_SCALE: 3.0\nSIMULATE_SCALE: 2.0\n"
    )
    tmp_path.joinpath("data").mkdir()

    state = workflow.State()
    state.initialize_filesystem(
        working_dir=tmp_path, configs_dir=("configs",)
    ).default_settings()
    state.settings.check_for_variability = False

    choosers = pd.DataFrame(
        {"chooser_attr": [1.0, 2.0]},
        index=pd.Index([0, 1], name="person_id"),
    )
    alternatives = pd.DataFrame(
        {"alt_attr": [1.0, 2.0]},
        index=pd.Index([10, 20], name="alt_id"),
    )

    # Sampling and simulation use separate specifications in location and
    # destination choice, so exercise each expression-evaluation path.
    sample_spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr * SAMPLE_SCALE"], name="Expression"),
    )
    sample = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        sample_spec,
        sample_size=0,
        alt_col_name="alt_id",
    )
    sampled_alternatives = sample.join(alternatives, on="alt_id")

    simulate_spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr * SIMULATE_SCALE"], name="Expression"),
    )
    results = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        sampled_alternatives,
        simulate_spec,
        choice_column="alt_id",
        want_logsums=True,
        skip_choice=True,
    )

    expected_logsum = np.logaddexp(2.0, 4.0)
    np.testing.assert_allclose(results["logsums"], expected_logsum)


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


def _call_with_unavailable_chooser(state, **kwargs):
    """Person 2 has no available alternative; person 1 has two at utility 0."""
    choosers = pd.DataFrame(
        {"chooser_ok": [1.0, 0.0]},
        index=pd.Index([1, 2], name="person_id"),
    )
    alternatives = pd.DataFrame(
        {"alt_id": [1, 2, 1, 2]},
        index=pd.Index([1, 1, 2, 2], name="person_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["-1000 * (1 - chooser_ok)"], name="Expression"),
    )
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_zero_prob_choice_val")

    return interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="alt_id",
        allow_zero_probs=True,
        **kwargs,
    )


def test_interaction_sample_simulate_allow_zero_probs_requires_choice_val(state):
    """Without zero_prob_choice_val, a chooser with no available alternative would
    silently keep its placeholder (first) alternative."""
    with pytest.raises(ValueError, match="zero_prob_choice_val is required"):
        _call_with_unavailable_chooser(state)


def test_interaction_sample_simulate_skip_choice_needs_no_choice_val(state):
    """Logsum-only calls make no choice, so they need no zero_prob_choice_val."""
    result = _call_with_unavailable_chooser(state, want_logsums=True, skip_choice=True)

    assert result["logsums"].iloc[0] == pytest.approx(np.log(2.0))
    assert result["logsums"].iloc[1] == -np.inf


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


@pytest.fixture(scope="module")
def sharrow_cache_dir(tmp_path_factory):
    # shared so the flow for the spec below compiles once per module
    return tmp_path_factory.mktemp("sharrow-cache")


# low but available: exp() underflows only in float32
LOW_UTILITY = -120.0


def _simulate_two_trips(
    tmp_path,
    sharrow_cache_dir,
    utilities_a,
    utilities_b,
    *,
    sharrow,
    use_explicit_error_terms,
    allow_zero_probs,
    want_logsums=False,
):
    """Trips 1 and 2 choose among sampled alternatives 1..n with the given utilities.

    Sharrow evaluates the utilities as float32, plain evaluation as float64.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    state = workflow.State().default_settings()
    state.initialize_filesystem(working_dir=tmp_path, configs_dir=("configs",))
    state.filesystem.sharrow_cache_dir = sharrow_cache_dir
    state.set("skim_dataset", sh.Dataset())
    state.settings.check_for_variability = False
    state.settings.sharrow = sharrow
    state.settings.use_explicit_error_terms = use_explicit_error_terms

    choosers = pd.DataFrame(
        {"household_id": [101, 102], "chooser_util": [0.0, 0.0]},
        index=pd.Index([1, 2], name="trip_id"),
    )
    alternatives = pd.DataFrame(
        {
            "alt_dest": [
                *range(1, len(utilities_a) + 1),
                *range(1, len(utilities_b) + 1),
            ],
            "alt_util": [*utilities_a, *utilities_b],
        },
        index=pd.Index([1] * len(utilities_a) + [2] * len(utilities_b), name="trip_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0, 1.0]},
        index=pd.Index(["alt_util", "chooser_util"], name="Expression"),
    )
    state.rng().add_channel("trip_id", choosers)
    state.rng().begin_step("test_sample_simulate_zero_probs")

    result = interaction_sample_simulate.interaction_sample_simulate(
        state,
        choosers,
        alternatives,
        spec,
        choice_column="alt_dest",
        allow_zero_probs=allow_zero_probs,
        zero_prob_choice_val=-1,
        want_logsums=want_logsums,
        compute_settings=ComputeSettings(protect_columns=["alt_dest"]),
    )
    return result, state


@pytest.mark.parametrize("num_alts", [2, 15, 20, 24, 30])
@pytest.mark.parametrize("sharrow", [False, True], ids=["float64", "float32"])
def test_interaction_sample_simulate_eet_flags_unavailable_rows(
    tmp_path, sharrow_cache_dir, sharrow, num_alts
):
    """Under EET, a trip with no available sampled alternative gets zero_prob_choice_val.

    This holds whatever the number of alternatives and the utility dtype.
    """
    choices, state = _simulate_two_trips(
        tmp_path,
        sharrow_cache_dir,
        [0.0] + [-1000.0] * (num_alts - 1),
        [-1000.0] * num_alts,
        sharrow=sharrow,
        use_explicit_error_terms=True,
        allow_zero_probs=True,
    )

    assert choices.tolist() == [1, -1]
    assert state.get("num_skipped_households", 0) == 0


@pytest.mark.parametrize("use_explicit_error_terms", [False, True], ids=["mnl", "eet"])
@pytest.mark.parametrize("sharrow", [False, True], ids=["float64", "float32"])
def test_interaction_sample_simulate_low_utilities_stay_available(
    request, tmp_path, sharrow_cache_dir, sharrow, use_explicit_error_terms
):
    """A trip whose sampled alternatives all have utility -120 still chooses one."""
    if sharrow and not use_explicit_error_terms:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="unshifted float32 exp() underflows below about -103, so "
                "MNL returns zero_prob_choice_val for an available trip (#1115)",
            )
        )

    choices, state = _simulate_two_trips(
        tmp_path,
        sharrow_cache_dir,
        [0.0, 0.0],
        [LOW_UTILITY, LOW_UTILITY - 1.0],
        sharrow=sharrow,
        use_explicit_error_terms=use_explicit_error_terms,
        allow_zero_probs=True,
    )

    assert choices.isin([1, 2]).all()
    assert state.get("num_skipped_households", 0) == 0


@pytest.mark.parametrize("use_explicit_error_terms", [False, True], ids=["mnl", "eet"])
@pytest.mark.parametrize("sharrow", [False, True], ids=["float64", "float32"])
def test_interaction_sample_simulate_logsums_of_low_utilities(
    request, tmp_path, sharrow_cache_dir, sharrow, use_explicit_error_terms
):
    """Destination logsums stay finite when every sampled alternative is low."""
    if sharrow:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="unshifted float32 exp() underflows below about -103, so "
                "the logsum is -inf (#1115)",
            )
        )

    result, state = _simulate_two_trips(
        tmp_path,
        sharrow_cache_dir,
        [0.0, 0.0],
        [LOW_UTILITY, LOW_UTILITY],
        sharrow=sharrow,
        use_explicit_error_terms=use_explicit_error_terms,
        allow_zero_probs=False,
        want_logsums=True,
    )

    assert result["logsum"].iloc[1] == pytest.approx(np.log(2.0) + LOW_UTILITY)
    assert state.get("num_skipped_households", 0) == 0
