# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import simulate, workflow


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def spec_name(data_dir):
    return "sample_spec.csv"


@pytest.fixture
def state(data_dir) -> workflow.State:
    state = workflow.State()
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__), data_dir=(data_dir,)
    ).default_settings()
    return state


@pytest.fixture
def spec(state, spec_name):
    return state.filesystem.read_model_spec(file_name=spec_name)


@pytest.fixture
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "data.csv"))


def test_read_model_spec(state, spec_name):
    spec = state.filesystem.read_model_spec(file_name=spec_name)

    assert len(spec) == 4
    assert spec.index.name == "Expression"
    assert list(spec.columns) == ["alt0", "alt1"]
    npt.assert_array_equal(spec.values, [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(state, spec, data):
    result = simulate.eval_variables(state, spec.index, data)

    expected = pd.DataFrame(
        [[1, 0, 4, 1], [0, 1, 4, 1], [0, 1, 5, 1]], index=data.index, columns=spec.index
    )

    expected[expected.columns[0]] = expected[expected.columns[0]].astype(np.int8)
    expected[expected.columns[1]] = expected[expected.columns[1]].astype(np.int8)
    expected[expected.columns[2]] = expected[expected.columns[2]].astype(np.int64)
    expected[expected.columns[3]] = expected[expected.columns[3]].astype(int)

    print("\nexpected\n%s" % expected.dtypes)
    print("\nresult\n%s" % result.dtypes)

    pdt.assert_frame_equal(result, expected, check_names=False)


def test_simple_simulate(state, data, spec):
    state.settings.check_for_variability = False

    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_simple_simulate_chunked(state, data, spec):
    state.settings.check_for_variability = False
    state.settings.chunk_size = 2
    choices = simulate.simple_simulate(
        state,
        choosers=data,
        spec=spec,
        nest_spec=None,
    )
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)
