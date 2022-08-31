# ActivitySim
# See full license in LICENSE.txt.

import os.path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from .. import inject, simulate


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def spec_name(data_dir):
    return "sample_spec.csv"


@pytest.fixture(scope="module")
def spec(data_dir, spec_name):
    return simulate.read_model_spec(file_name=spec_name)


@pytest.fixture(scope="module")
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "data.csv"))


def setup_function():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), f"output")
    inject.add_injectable("output_dir", output_dir)


def test_read_model_spec(spec_name):

    spec = simulate.read_model_spec(file_name=spec_name)

    assert len(spec) == 4
    assert spec.index.name == "Expression"
    assert list(spec.columns) == ["alt0", "alt1"]
    npt.assert_array_equal(spec.values, [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(spec, data):

    result = simulate.eval_variables(spec.index, data)

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


def test_simple_simulate(data, spec):

    inject.add_injectable("settings", {"check_for_variability": False})

    choices = simulate.simple_simulate(choosers=data, spec=spec, nest_spec=None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected)


def test_simple_simulate_chunked(data, spec):

    inject.add_injectable("settings", {"check_for_variability": False})

    choices = simulate.simple_simulate(
        choosers=data, spec=spec, nest_spec=None, chunk_size=2
    )
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected)
