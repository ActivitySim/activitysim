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

    assert len(spec) == 7

    assert list(spec.columns) == ['description', 'target', 'expression']


def test_eval_variables(capsys, spec_name, data):

    spec = asim_eval.read_assignment_spec(spec_name)

    locals_d = {'CONSTANT': 7, '_shadow': 99}

    trace_rows = None
    results, trace_results = asim_eval.assign_variables(spec, data, locals_d, trace_rows)

    print results

    assert list(results.columns) == ['target1', 'target2', 'target3']
    assert list(results.target1) == [True, False, False]
    assert list(results.target2) == [53, 53, 55]
    assert list(results.target3) == [None, None, None]
    assert trace_results is None

    trace_rows = [False, True, False]

    results, trace_results = asim_eval.assign_variables(spec, data, locals_d, trace_rows=trace_rows)

    print trace_results

    assert trace_results is not None
    assert '_scalar' in trace_results.columns
    assert list(trace_results['_scalar']) == [42]

    # shadow should have been assigned
    assert list(trace_results['_shadow']) == [1]
    assert list(trace_results['_temp']) == [9]

    assert locals_d['_shadow'] == 99

    out, err = capsys.readouterr()
