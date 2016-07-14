# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import asim_eval as asim_eval


@pytest.fixture(scope='module')
def spec_name():
    return os.path.join(os.path.dirname(__file__), 'data', 'sample_assignment_spec.csv')


@pytest.fixture(scope='module')
def data_name():
    return os.path.join(os.path.dirname(__file__), 'data', 'data.csv')


@pytest.fixture(scope='module')
def data(data_name):
    return pd.read_csv(data_name)


def test_read_model_spec(spec_name):

    spec = asim_eval.read_assignment_spec(spec_name)

    assert len(spec) == 5

    assert list(spec.columns) == ['description', 'target', 'expression']


def test_eval_variables(capsys, spec_name, data):

    spec = asim_eval.read_assignment_spec(spec_name)

    locals_d = {'CONSTANT': 7}
    result = asim_eval.assign_variables(spec, data, locals_d)

    print result

    assert list(result.columns) == ['target3', 'target2', 'target1']
    assert list(result.target1) == [True, False, False]
    assert list(result.target2) == [53, 53, 55]
    assert list(result.target3) == [None, None, None]
