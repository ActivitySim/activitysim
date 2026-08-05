import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.abm.models import joint_tour_participation
from activitysim.core import logit, workflow

from .test_trip_departure_choice import add_canonical_dirs


@pytest.fixture
def candidates():
    # Create synthetic candidates for Joint Tour Participation
    # JTP chooses whether each candidate participates in a joint tour.
    # We include varied compositions and preschoolers to exercise the
    # get_tour_satisfaction logic properly.
    num_tours_per_comp = 500
    compositions = ["MIXED", "ADULTS", "CHILDREN"]
    num_candidates_per_tour = 4

    total_tours = num_tours_per_comp * len(compositions)
    num_candidates = total_tours * num_candidates_per_tour

    # Ensure reproducibility
    rng = np.random.default_rng(42)

    tour_ids = np.repeat(np.arange(total_tours), num_candidates_per_tour)
    comp_values = np.repeat(compositions, num_tours_per_comp * num_candidates_per_tour)

    df = pd.DataFrame(
        {
            "tour_id": tour_ids,
            "household_id": tour_ids,  # simplified for mock
            "person_id": np.arange(num_candidates),
            "composition": comp_values,
        },
        index=pd.Index(np.arange(num_candidates), name="participant_id"),
    )

    # Assign adult and preschooler status based on composition
    # MIXED: at least one adult and one child
    # ADULTS: all adults
    # CHILDREN: all children
    df["adult"] = False
    df["person_is_preschool"] = False

    for i, comp in enumerate(compositions):
        mask = df.composition == comp
        indices = df[mask].index

        if comp == "ADULTS":
            df.loc[indices, "adult"] = True
        elif comp == "CHILDREN":
            df.loc[indices, "adult"] = False
            # Some children are preschoolers
            df.loc[
                rng.choice(indices, len(indices) // 4, replace=False),
                "person_is_preschool",
            ] = True
        elif comp == "MIXED":
            # For each tour, make the first person an adult, rest children
            tour_start_indices = indices[::num_candidates_per_tour]
            df.loc[tour_start_indices, "adult"] = True
            # Other members are children, some might be preschoolers
            other_indices = indices[~indices.isin(tour_start_indices)]
            df.loc[
                rng.choice(other_indices, len(other_indices) // 3, replace=False),
                "person_is_preschool",
            ] = True

    return df


@pytest.fixture
def model_spec():
    # Simple spec with two alternatives: 'participate' and 'not_participate'
    return pd.DataFrame(
        {"participate": [0.8, -0.2], "not_participate": [0.0, 0.0]},
        index=pd.Index(["adult", "person_is_preschool"], name="Expression"),
    )


def test_jtp_explicit_error_terms_parity(candidates, model_spec):
    """
    Test that joint tour participation results are statistically similar
    between MNL and Explicit Error Terms (EET) using realistic candidate scenarios.
    """
    # Create random utilities for the candidates that vary by attribute
    rng = np.random.default_rng(42)

    # Base utility + some noise
    base_util = (candidates.adult * 0.5) - (candidates.person_is_preschool * 1.0)
    utils = pd.DataFrame(
        {
            "participate": base_util + rng.standard_normal(len(candidates)),
            "not_participate": 0,
        },
        index=candidates.index,
    )

    # Run without EET (MNL)
    state_no_eet = add_canonical_dirs("configs_test_misc").default_settings()
    state_no_eet.settings.use_explicit_error_terms = False
    state_no_eet.rng().set_base_seed(42)
    state_no_eet.rng().begin_step("test_no_eet")
    state_no_eet.rng().add_channel("participant_id", candidates)

    # MNL path expects probabilities
    probs_no_eet = logit.utils_to_probs(state_no_eet, utils, trace_label="test_no_eet")
    choices_no_eet, _ = joint_tour_participation.participants_chooser(
        state_no_eet,
        probs_no_eet,
        candidates,
        model_spec,
        trace_label="test_no_eet",
    )

    # Run with EET
    state_eet = add_canonical_dirs("configs_test_misc").default_settings()
    state_eet.settings.use_explicit_error_terms = True
    state_eet.rng().set_base_seed(42)
    state_eet.rng().begin_step("test_eet")
    state_eet.rng().add_channel("participant_id", candidates)

    # EET path expects raw utilities
    choices_eet, _ = joint_tour_participation.participants_chooser(
        state_eet,
        utils.copy(),
        candidates,
        model_spec,
        trace_label="test_eet",
    )

    # Compare distributions of number of participants per tour
    # Choice 0 is 'participate'
    no_eet_participation_counts = (
        (choices_no_eet == 0).groupby(candidates.tour_id).sum()
    )
    eet_participation_counts = (choices_eet == 0).groupby(candidates.tour_id).sum()

    dist_no_eet = no_eet_participation_counts.value_counts(normalize=True).sort_index()
    dist_eet = eet_participation_counts.value_counts(normalize=True).sort_index()

    # Check that the distribution of participation counts is close
    pdt.assert_series_equal(dist_no_eet, dist_eet, atol=0.05, check_names=False)

    # Also check average participation by composition for deeper parity check
    comp_parity_no_eet = no_eet_participation_counts.groupby(
        candidates.groupby("tour_id")["composition"].first()
    ).mean()
    comp_parity_eet = eet_participation_counts.groupby(
        candidates.groupby("tour_id")["composition"].first()
    ).mean()

    pdt.assert_series_equal(
        comp_parity_no_eet, comp_parity_eet, atol=0.1, check_names=False
    )
