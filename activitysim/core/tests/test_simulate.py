# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

import orca

from .. import simulate as asim


@pytest.fixture(scope='module')
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def spec_name(data_dir):
    return os.path.join(data_dir, 'sample_spec.csv')


@pytest.fixture(scope='module')
def spec(spec_name):
    return asim.read_model_spec(spec_name,
                                description_name='description',
                                expression_name='expression')


@pytest.fixture(scope='module')
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, 'data.csv'))


def test_read_model_spec(spec_name):
    spec = asim.read_model_spec(
        spec_name, description_name='description', expression_name='expression')

    assert len(spec) == 4
    assert spec.index.name == 'expression'
    assert list(spec.columns) == ['alt0', 'alt1']
    npt.assert_array_equal(
        spec.as_matrix(),
        [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(spec, data):
    result = asim.eval_variables(spec.index, data)

    pdt.assert_frame_equal(
        result,
        pd.DataFrame([
            [True, False, 4, 1],
            [False, True, 4, 1],
            [False, True, 5, 1]],
            index=data.index, columns=spec.index),
        check_names=False)


def test_simple_simulate(data, spec):

    orca.add_injectable("check_for_variability", False)

    choices = asim.simple_simulate(data, spec, None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected)
