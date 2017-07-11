# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy as np
import pandas as pd
import orca

import pandas.util.testing as pdt
import pytest

from ..simulate import eval_variables
from .. import logit


@pytest.fixture(scope='module')
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


def add_canonical_dirs():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)


# this is lifted straight from urbansim's test_mnl.py
@pytest.fixture(scope='module', params=[
    ('fish.csv',
        'fish_choosers.csv',
        pd.DataFrame(
            [[-0.02047652], [0.95309824]], index=['price', 'catch'],
            columns=['Alt']),
        pd.DataFrame([
            [0.2849598, 0.2742482, 0.1605457, 0.2802463],
            [0.1498991, 0.4542377, 0.2600969, 0.1357664]],
            columns=['beach', 'boat', 'charter', 'pier']))])
def test_data(request):
    data, choosers, spec, probabilities = request.param
    return {
        'data': data,
        'choosers': choosers,
        'spec': spec,
        'probabilities': probabilities
    }


@pytest.fixture
def choosers(test_data, data_dir):
    filen = os.path.join(data_dir, test_data['choosers'])
    return pd.read_csv(filen)


@pytest.fixture
def spec(test_data):
    return test_data['spec']


@pytest.fixture
def choosers_dm(choosers, spec):
    return eval_variables(spec.index, choosers)


@pytest.fixture
def utilities(choosers_dm, spec, test_data):
    utils = choosers_dm.dot(spec).astype('float')
    return pd.DataFrame(
        utils.as_matrix().reshape(test_data['probabilities'].shape),
        columns=test_data['probabilities'].columns)


def test_utils_to_probs(utilities, test_data):
    probs = logit.utils_to_probs(utilities, trace_label=None)
    pdt.assert_frame_equal(probs, test_data['probabilities'])


def test_utils_to_probs_raises():

    add_canonical_dirs()

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(pd.DataFrame([[1, 2, np.inf, 3]]), trace_label=None)
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(pd.DataFrame([[-999, -999, -999, -999]]), trace_label=None)
    assert "all probabilities are zero" in str(excinfo.value)


def test_make_choices_only_one():
    probs = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0]], columns=['a', 'b', 'c'], index=['x', 'y'])
    choices, rands = logit.make_choices(probs)

    pdt.assert_series_equal(
        choices,
        pd.Series([0, 1], index=['x', 'y']))


def test_make_choices_real_probs(utilities):
    probs = logit.utils_to_probs(utilities, trace_label=None)
    choices, rands = logit.make_choices(probs)

    pdt.assert_series_equal(
        choices,
        pd.Series([1, 2], index=[0, 1]))


@pytest.fixture(scope='module')
def interaction_choosers():
    return pd.DataFrame({
        'attr': ['a', 'b', 'c', 'b']},
        index=['w', 'x', 'y', 'z'])


@pytest.fixture(scope='module')
def interaction_alts():
    return pd.DataFrame({
        'prop': [10, 20, 30, 40]},
        index=[1, 2, 3, 4])


def test_interaction_dataset_no_sample(interaction_choosers, interaction_alts):
    expected = pd.DataFrame({
        'attr': ['a'] * 4 + ['b'] * 4 + ['c'] * 4 + ['b'] * 4,
        'prop': [10, 20, 30, 40] * 4,
        'chooser_idx': ['w'] * 4 + ['x'] * 4 + ['y'] * 4 + ['z'] * 4},
        index=[1, 2, 3, 4] * 4)

    interacted = logit.interaction_dataset(
        interaction_choosers, interaction_alts)

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)


def test_interaction_dataset_sampled(
        interaction_choosers, interaction_alts):
    expected = pd.DataFrame({
        'attr': ['a'] * 2 + ['b'] * 2 + ['c'] * 2 + ['b'] * 2,
        'prop': [30, 40, 10, 30, 40, 10, 20, 10],
        'chooser_idx': ['w'] * 2 + ['x'] * 2 + ['y'] * 2 + ['z'] * 2},
        index=[3, 4, 1, 3, 4, 1, 2, 1])

    interacted = logit.interaction_dataset(
        interaction_choosers, interaction_alts, sample_size=2)

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)
