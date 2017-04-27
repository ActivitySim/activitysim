# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy.testing as npt
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest

import orca

from .. import assign
from .. import tracing


@pytest.fixture(scope='module')
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def configs_dir():
    return os.path.join(os.path.dirname(__file__), 'configs')


@pytest.fixture(scope='module')
def spec_name(data_dir):
    return os.path.join(data_dir, 'assignment_spec.csv')


@pytest.fixture(scope='module')
def data_name(data_dir):
    return os.path.join(data_dir, 'data.csv')


@pytest.fixture(scope='module')
def data(data_name):
    return pd.read_csv(data_name)


def test_read_model_spec(spec_name):

    spec = assign.read_assignment_spec(spec_name)

    assert len(spec) == 8

    assert list(spec.columns) == ['description', 'target', 'expression']


def test_assign_variables(capsys, spec_name, data):

    spec = assign.read_assignment_spec(spec_name)

    locals_d = {'CONSTANT': 7, '_shadow': 99}

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(spec, data, locals_d, trace_rows=None)

    print results

    assert list(results.columns) == ['target1', 'target2', 'target3']
    assert list(results.target1) == [True, False, False]
    assert list(results.target2) == [53, 53, 55]
    assert list(results.target3) == [530, 530, 550]
    assert trace_results is None
    assert trace_assigned_locals is None

    trace_rows = [False, True, False]

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(spec, data, locals_d, trace_rows=trace_rows)

    # should get same results as before
    assert list(results.target3) == [530, 530, 550]

    # should assign trace_results for second row in data
    print trace_results

    assert trace_results is not None
    assert '_scalar' in trace_results.columns
    assert list(trace_results['_scalar']) == [42]

    # shadow should have been assigned
    assert list(trace_results['_shadow']) == [1]
    assert list(trace_results['_temp']) == [9]
    assert list(trace_results['target3']) == [530]

    print "trace_assigned_locals", trace_assigned_locals
    assert trace_assigned_locals['_DF_COL_NAME'] == 'thing2'

    # shouldn't have been changed even though it was a target
    assert locals_d['_shadow'] == 99

    out, err = capsys.readouterr()


def test_assign_variables_aliased(capsys, data):

    spec_name = \
        os.path.join(os.path.dirname(__file__), 'data', 'assignment_spec_alias_df.csv')

    spec = assign.read_assignment_spec(spec_name)

    locals_d = {'CONSTANT': 7, '_shadow': 99}

    trace_rows = [False, True, False]

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(spec, data, locals_d,
                                  df_alias='aliased_df', trace_rows=trace_rows)

    print results

    assert list(results.columns) == ['target1', 'target2', 'target3']
    assert list(results.target1) == [True, False, False]
    assert list(results.target2) == [53, 53, 55]
    assert list(results.target3) == [530, 530, 550]

    # should assign trace_results for second row in data
    print trace_results

    assert trace_results is not None
    assert '_scalar' in trace_results.columns
    assert list(trace_results['_scalar']) == [42]

    # shadow should have been assigned
    assert list(trace_results['_shadow']) == [1]
    assert list(trace_results['_temp']) == [9]
    assert list(trace_results['target3']) == [530]

    assert locals_d['_shadow'] == 99

    out, err = capsys.readouterr()


def test_assign_variables_failing(capsys, data):

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    tracing.config_logger(basic=True)

    spec_name = \
        os.path.join(os.path.dirname(__file__), 'data', 'assignment_spec_failing.csv')

    spec = assign.read_assignment_spec(spec_name)

    locals_d = {
        'CONSTANT': 7,
        '_shadow': 99,
        'log': np.log,
    }

    with pytest.raises(NameError) as excinfo:
        results, trace_results = assign.assign_variables(spec, data, locals_d, trace_rows=None)

    out, err = capsys.readouterr()
    # don't consume output
    print out

    # numpy warning should appear in log but not raise error
    assert 'invalid value encountered in log' in out

    # undefined variable should raise error
    assert "'undefined_variable' is not defined" in str(excinfo.value)
