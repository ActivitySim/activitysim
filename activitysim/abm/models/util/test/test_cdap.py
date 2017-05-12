# ActivitySim
# See full license in LICENSE.txt.

import os.path
from itertools import product

import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import cdap

from activitysim.core.simulate import read_model_spec


@pytest.fixture(scope='module')
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def people(data_dir):
    return pd.read_csv(
        os.path.join(data_dir, 'people.csv'),
        index_col='id')


@pytest.fixture(scope='module')
def cdap_indiv_and_hhsize1(data_dir):
    return read_model_spec(
        os.path.join(data_dir, 'cdap_indiv_and_hhsize1.csv'))


@pytest.fixture(scope='module')
def cdap_interaction_coefficients(data_dir):
    f = os.path.join(data_dir, 'cdap_interaction_coefficients.csv')
    coefficients = pd.read_csv(f, comment='#')
    coefficients = cdap.preprocess_interaction_coefficients(coefficients)
    return coefficients


@pytest.fixture(scope='module')
def individual_utils(
        people, cdap_indiv_and_hhsize1):
    return cdap.individual_utilities(people, cdap_indiv_and_hhsize1, locals_d=None)


# @pytest.fixture
# def hh_utils(individual_utils, people, hh_id_col):
#     hh_utils = cdap.initial_household_utilities(
#         individual_utils, people, hh_id_col)
#     return hh_utils
#
#
# @pytest.fixture
# def hh_choices(random_seed, hh_utils):
#     return cdap.make_household_choices(hh_utils)


def test_bad_coefficients(data_dir):

    f = os.path.join(data_dir, 'cdap_interaction_coefficients.csv')
    coefficients = pd.read_csv(f, comment='#')

    coefficients.loc[2, 'activity'] = 'AA'

    with pytest.raises(RuntimeError) as excinfo:
        coefficients = cdap.preprocess_interaction_coefficients(coefficients)
    assert "Expect only M, N, or H" in str(excinfo.value)


def test_assign_cdap_rank(people):

    cdap.assign_cdap_rank(people)

    expected = pd.Series(
        [1, 1, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3, 2, 4, 1, 3, 4, 2],
        index=people.index
    )

    pdt.assert_series_equal(people['cdap_rank'], expected, check_dtype=False, check_names=False)


def test_individual_utilities(people, cdap_indiv_and_hhsize1):

    cdap.assign_cdap_rank(people)
    individual_utils = cdap.individual_utilities(people, cdap_indiv_and_hhsize1, locals_d=None)
    individual_utils = individual_utils[['M', 'N', 'H']]

    expected = pd.DataFrame([
        [2, 0, 0],  # person 1
        [0, 0, 1],  # person 2
        [3, 0, 0],  # person 3
        [3, 0, 0],  # person 4
        [0, 1, 0],  # person 5
        [1, 0, 0],  # person 6
        [1, 0, 0],  # person 7
        [0, 2, 0],  # person 8
        [0, 0, 1],  # person 9
        [2, 0, 0],  # person 10
        [0, 0, 3],  # person 11
        [0, 0, 2],  # person 12
        [3, 0, 0],  # person 13
        [1, 0, 0],  # person 14
        [0, 4, 0],  # person 15
        [0, 4, 0],  # person 16
        [0, 0, 4],  # person 17
        [0, 0, 5],  # person 18
        [50, 0, 4],  # person 19
        [2, 0, 0]  # person 20
        ], index=people.index, columns=cdap_indiv_and_hhsize1.columns)

    pdt.assert_frame_equal(
        individual_utils, expected, check_dtype=False, check_names=False)


def test_build_cdap_spec_hhsize2(people, cdap_indiv_and_hhsize1, cdap_interaction_coefficients):

    hhsize = 2

    cdap.assign_cdap_rank(people)
    indiv_utils = cdap.individual_utilities(people, cdap_indiv_and_hhsize1, locals_d=None)

    choosers = cdap.hh_choosers(indiv_utils, hhsize=hhsize)

    spec = cdap.build_cdap_spec(cdap_interaction_coefficients, hhsize=hhsize)

    vars = cdap.eval_variables(spec.index, choosers)

    utils = vars.dot(spec).astype('float')

    expected = pd.DataFrame([
        [0, 3, 0, 3, 7, 3, 0, 3, 0],  # household 3
        [0, 0, 1, 1, 1, 2, 0, 0, 2],  # household 4
        ],
        index=[3, 4],
        columns=['HH', 'HM', 'HN', 'MH', 'MM', 'MN', 'NH', 'NM', 'NN']).astype('float')

    pdt.assert_frame_equal(utils, expected, check_names=False)
