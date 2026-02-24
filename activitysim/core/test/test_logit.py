# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path
import re

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


from activitysim.core import logit, workflow, random
from activitysim.core.logit import add_ev1_random
from activitysim.core.simulate import eval_variables


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


# this is lifted straight from urbansim's test_mnl.py
@pytest.fixture(
    scope="module",
    params=[
        (
            "fish.csv",
            "fish_choosers.csv",
            pd.DataFrame(
                [[-0.02047652], [0.95309824]], index=["price", "catch"], columns=["Alt"]
            ),
            pd.DataFrame(
                [
                    [0.2849598, 0.2742482, 0.1605457, 0.2802463],
                    [0.1498991, 0.4542377, 0.2600969, 0.1357664],
                ],
                columns=["beach", "boat", "charter", "pier"],
            ),
        )
    ],
)
def test_data(request):
    data, choosers, spec, probabilities = request.param
    return {
        "data": data,
        "choosers": choosers,
        "spec": spec,
        "probabilities": probabilities,
    }


@pytest.fixture
def choosers(test_data, data_dir):
    filen = os.path.join(data_dir, test_data["choosers"])
    return pd.read_csv(filen)


@pytest.fixture
def spec(test_data):
    return test_data["spec"]


@pytest.fixture
def utilities(choosers, spec, test_data):
    state = workflow.State().default_settings()
    vars = eval_variables(state, spec.index, choosers)
    utils = vars.dot(spec).astype("float")
    return pd.DataFrame(
        utils.values.reshape(test_data["probabilities"].shape),
        columns=test_data["probabilities"].columns,
    )


# TODO-EET: Add tests here!


def test_utils_to_probs(utilities, test_data):
    state = workflow.State().default_settings()
    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    pdt.assert_frame_equal(probs, test_data["probabilities"])


def test_utils_to_probs_raises():
    state = workflow.State().default_settings()
    idx = pd.Index(name="household_id", data=[1])
    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, np.inf, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, 9999, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[-999, -999, -999, -999]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "all probabilities are zero" in str(excinfo.value)

    # test that overflow protection works
    z = logit.utils_to_probs(
        state,
        pd.DataFrame([[1, 2, 9999, 3]], index=idx),
        trace_label=None,
        overflow_protection=True,
    )
    assert np.asarray(z).ravel() == pytest.approx(np.asarray([0.0, 0.0, 1.0, 0.0]))


def test_make_choices_only_one():
    state = workflow.State().default_settings()
    probs = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0]], columns=["a", "b", "c"], index=["x", "y"]
    )
    choices, rands = logit.make_choices(state, probs)

    pdt.assert_series_equal(
        choices, pd.Series([0, 1], index=["x", "y"]), check_dtype=False
    )

def reset_step(state, name='test_step'):
    state.get_rn_generator().end_step(name)
    state.get_rn_generator().begin_step(name)

def test_make_choices_utility_based_sampled_alts():
    """Test the situation of making choices from a sampled choice set"""
    # TODO should these tests go in test_random?
    state = workflow.State().default_settings()
    # Make explicit that there's two indexing schemes - the raw alts, and the 0 based internals
    utils_project_raw = pd.DataFrame({"a":10.582999, "b":10.680792, "c":10.710443}, index=pd.Index([0], name='person_id'))
    # zero based indexes
    utils_project = utils_project_raw.rename(columns={"a":0, "b":1, "c":2})
    utils_base = utils_project_raw[["a", "c"]].rename(columns={"a":0, "c":1})

    assert utils_project.index.name == "person_id"
    state.get_rn_generator().add_channel("persons", utils_project)
    state.get_rn_generator().begin_step("test_step")
    # mock base case, where alt 1 is omitted (it was improved in the project)
    # this situation is quite common with poisson sampling with a variable choice set size,
    # but it can also happen in with-replacement EET sampling e.g. if alt 2 had a pick_count of 2 in the base case.
    # In principle, it can also be problematic for non-sampled choices where there is a base project difference in the
    # availability of alternatives .e.g a new mode was introduced in the project case

    utils_project_with_rands = add_ev1_random(state, utils_project)
    rands_project = utils_project_with_rands - utils_project
    reset_step(state)
    utils_base_with_rands = add_ev1_random(state, utils_base)
    rands_base = utils_base_with_rands - utils_base
    rands_base_labeled = rands_base.rename(columns={0:"a", 1:"c"})
    rands_project_labeled = rands_project.rename(columns={0:"a", 1:"b", 2:"c"})
    with pytest.raises(AssertionError, match=re.escape('(column name="c") are different')):
        # TODO this should pass
        pdt.assert_frame_equal(rands_base_labeled, rands_project_labeled.loc[:, rands_base_labeled.columns])
    # document incorrect invariant - first two columns have the same random numbers:
    pdt.assert_frame_equal(rands_base, rands_project.iloc[:, :2])

    # revised approach
    reset_step(state)
    alt_nrs_df = pd.DataFrame({0:0, 1:1, 2:2}, index=utils_project_raw.index)
    utils_project_with_rands = add_ev1_random(state, utils_project, n_alts=3, alt_nrs_df=alt_nrs_df)
    rands_project = utils_project_with_rands - utils_project
    reset_step(state)

    # alt "b" is missing from the sampled choice set, alt_nrs_df is set to reflect that
    alt_nrs_df = pd.DataFrame({0: 0, 1: 2}, index=utils_project_raw.index)
    utils_base_with_rands = add_ev1_random(state, utils_base, n_alts=3, alt_nrs_df=alt_nrs_df)
    rands_base = utils_base_with_rands - utils_base
    rands_base_labeled = rands_base.rename(columns={0: "a", 1: "c"})
    rands_project_labeled = rands_project.rename(columns={0: "a", 1: "b", 2: "c"})

    # Corrected invariant holds true
    pdt.assert_frame_equal(rands_base_labeled, rands_project_labeled.loc[:, rands_base_labeled.columns])







def test_make_choices_real_probs(utilities):
    state = workflow.State().default_settings()
    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    choices, rands = logit.make_choices(state, probs)

    pdt.assert_series_equal(
        choices,
        pd.Series([1, 2], index=[0, 1]),
        check_dtype=False,
    )


@pytest.fixture(scope="module")
def interaction_choosers():
    return pd.DataFrame({"attr": ["a", "b", "c", "b"]}, index=["w", "x", "y", "z"])


@pytest.fixture(scope="module")
def interaction_alts():
    return pd.DataFrame({"prop": [10, 20, 30, 40]}, index=[1, 2, 3, 4])


def test_interaction_dataset_no_sample(interaction_choosers, interaction_alts):
    expected = pd.DataFrame(
        {
            "attr": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["b"] * 4,
            "prop": [10, 20, 30, 40] * 4,
        },
        index=[1, 2, 3, 4] * 4,
    )

    interacted = logit.interaction_dataset(
        workflow.State().default_settings(), interaction_choosers, interaction_alts
    )

    interacted, expected = interacted.align(expected, axis=1)

    print("interacted\n", interacted)
    print("expected\n", expected)
    pdt.assert_frame_equal(interacted, expected)


def test_interaction_dataset_sampled(interaction_choosers, interaction_alts):
    expected = pd.DataFrame(
        {
            "attr": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["b"] * 2,
            "prop": [30, 40, 10, 30, 40, 10, 20, 10],
        },
        index=[3, 4, 1, 3, 4, 1, 2, 1],
    )

    interacted = logit.interaction_dataset(
        workflow.State().default_settings(),
        interaction_choosers,
        interaction_alts,
        sample_size=2,
    )

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)
