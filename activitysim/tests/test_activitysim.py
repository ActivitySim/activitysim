import os.path

import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import activitysim as asim


@pytest.fixture(scope='module')
def spec_name():
    return os.path.join(os.path.dirname(__file__), 'data', 'sample_spec.csv')


@pytest.fixture(scope='module')
def data_name():
    return os.path.join(os.path.dirname(__file__), 'data', 'data.csv')


@pytest.fixture(scope='module')
def spec(spec_name):
    return asim.read_model_spec(
        spec_name,
        description_name='description', expression_name='expression')


@pytest.fixture(scope='module')
def data(data_name):
    return pd.read_csv(data_name)


def test_read_model_spec(spec_name):
    spec = asim.read_model_spec(
        spec_name,
        description_name='description', expression_name='expression')

    assert len(spec) == 3
    assert spec.index.name == 'expression'
    assert list(spec.columns) == ['alt0', 'alt1']


def test_identity_matrix():
    names = ['a', 'b', 'c']
    i = asim.identity_matrix(names)

    assert list(i.columns) == names
    assert list(i.index) == names

    npt.assert_array_equal(
        i.as_matrix(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_eval_variables(spec, data):
    result = asim.eval_variables(spec.index, data)

    pdt.assert_frame_equal(
        result,
        pd.DataFrame([
            [True, False, 4],
            [False, True, 4],
            [False, True, 5]],
            index=data.index, columns=spec.index),
        check_names=False)
