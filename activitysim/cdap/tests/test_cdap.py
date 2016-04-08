import os.path
from itertools import product

import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import cdap
from ...activitysim import read_model_spec


@pytest.fixture(scope='module')
def people():
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'people.csv'),
        index_col='id')


@pytest.fixture(scope='module')
def one_spec():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_1_person.csv'))


@pytest.fixture(scope='module')
def two_spec():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_2_person.csv'))


@pytest.fixture(scope='module')
def three_spec():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_3_person.csv'))


@pytest.fixture(scope='module')
def final_rules():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_final_rules.csv'))


@pytest.fixture
def all_people():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_all_people.csv'),
        expression_name='Alternative')


@pytest.fixture(scope='module')
def hh_id_col():
    return 'household'


@pytest.fixture(scope='module')
def p_type_col():
    return 'ptype'


@pytest.fixture(scope='module')
def individual_utils(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec):
    return cdap.individual_utilities(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec)


@pytest.fixture
def hh_utils(individual_utils, people, hh_id_col):
    hh_utils = cdap.initial_household_utilities(
        individual_utils, people, hh_id_col)
    return hh_utils


@pytest.fixture
def hh_choices(random_seed, hh_utils):
    return cdap.make_household_choices(hh_utils)


def test_make_interactions(people, hh_id_col, p_type_col):
    expected_two = pd.DataFrame(
        {'interaction': [
            '11',  # household 3; person 3
            '11',  # household 3; person 4
            '32',  # household 4; person 5
            '23',  # household 4; person 6
            '32', '32',  # household 5; person 7
            '23', '22',  # household 5; person 8
            '23', '22',  # household 5; person 9
            '13', '11',  # household 6; person 10
            '31', '31',  # household 6; person 11
            '11', '13',  # household 6; person 12
            '13', '12', '12',  # household 7; person 13
            '31', '32', '32',  # household 7; person 14
            '21', '23', '22',  # household 7; person 15
            '21', '23', '22',  # household 7; person 16
            '13', '12', '11',  # household 8; person 17
            '31', '32', '31',  # household 8; person 18
            '21', '23', '21',  # household 8; person 19
            '11', '13', '12'   # household 8; person 20
        ]},
        index=[
            3, 4, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
            13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
            17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20
        ])

    expected_three = pd.DataFrame(
        {'interaction': [
            '322', '322', '322',  # household 5; people 7, 8, 9
            '131', '131', '131',  # household 6; people 10, 11, 12
            '132', '132', '132',  # household 7; people 13, 14, 15
            '132', '132', '132',  # household 7; people 13, 14, 16
            '122', '122', '122',  # household 7; people 13, 15, 16
            '322', '322', '322',  # household 7; people 14, 15, 16
            '132', '132', '132',  # household 8; people 17, 18, 19
            '131', '131', '131',  # household 8; people 17, 18, 20
            '121', '121', '121',  # household 8; people 17, 19, 20
            '321', '321', '321'   # household 8; people 18, 19, 20
        ]},
        index=[
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 13, 14, 16, 13, 15, 16, 14, 15, 16,
            17, 18, 19, 17, 18, 20, 17, 19, 20, 18, 19, 20
        ])

    two, three = cdap.make_interactions(people, hh_id_col, p_type_col)

    pdt.assert_frame_equal(two, expected_two)
    pdt.assert_frame_equal(three, expected_three)


def test_make_interactions_no_interactions(people, hh_id_col, p_type_col):
    people = people.loc[[1, 2, 3]]

    two, three = cdap.make_interactions(people, hh_id_col, p_type_col)

    pdt.assert_frame_equal(
        two, pd.DataFrame(columns=['interaction']), check_dtype=False)
    pdt.assert_frame_equal(
        three, pd.DataFrame(columns=['interaction']), check_dtype=False)


def test_make_interactions_only_twos(people, hh_id_col, p_type_col):
    people = people.loc[[1, 2, 3, 4, 5, 6]]

    expected_two = pd.DataFrame(
        {'interaction': [
            '11',  # household 3; person 3
            '11',  # household 3; person 4
            '32',  # household 4; person 5
            '23',  # household 4; person 6
        ]},
        index=[3, 4, 5, 6]
    )

    two, three = cdap.make_interactions(people, hh_id_col, p_type_col)

    pdt.assert_frame_equal(two, expected_two)
    pdt.assert_frame_equal(
        three, pd.DataFrame(columns=['interaction']), check_dtype=False)


def test_individual_utilities(people, one_spec, individual_utils):
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
        ], index=people.index, columns=one_spec.columns)

    pdt.assert_frame_equal(
        individual_utils, expected, check_dtype=False, check_names=False)


def test_initial_household_utilities(hh_utils):
    alts = ['Mandatory', 'NonMandatory', 'Home']
    one_alts = list(product(alts, repeat=1))
    two_alts = list(product(alts, repeat=2))
    three_alts = list(product(alts, repeat=3))
    four_alts = list(product(alts, repeat=4))

    expected = {
        1: pd.Series([2, 0, 0], index=one_alts),
        2: pd.Series([0, 0, 1], index=one_alts),
        3: pd.Series([6, 3, 3, 3, 0, 0, 3, 0, 0], index=two_alts),
        4: pd.Series([1, 0, 0, 2, 1, 1, 1, 0, 0], index=two_alts),
        5: pd.Series([
            1, 1, 2, 3, 3, 4, 1, 1, 2,
            0, 0, 1, 2, 2, 3, 0, 0, 1,
            0, 0, 1, 2, 2, 3, 0, 0, 1,
            ], index=three_alts),
        6: pd.Series([
            2, 2, 4, 2, 2, 4, 5, 5, 7,
            0, 0, 2, 0, 0, 2, 3, 3, 5,
            0, 0, 2, 0, 0, 2, 3, 3, 5,
            ], index=three_alts),
        7: pd.Series([
            4, 8, 4, 8, 12, 8, 4, 8, 4,
            3, 7, 3, 7, 11, 7, 3, 7, 3,
            3, 7, 3, 7, 11, 7, 3, 7, 3,
            1, 5, 1, 5, 9, 5, 1, 5, 1,
            0, 4, 0, 4, 8, 4, 0, 4, 0,
            0, 4, 0, 4, 8, 4, 0, 4, 0,
            1, 5, 1, 5, 9, 5, 1, 5, 1,
            0, 4, 0, 4, 8, 4, 0, 4, 0,
            0, 4, 0, 4, 8, 4, 0, 4, 0
            ], index=four_alts),
        8: pd.Series([
            52, 50, 50, 2, 0, 0, 6, 4, 4,
            52, 50, 50, 2, 0, 0, 6, 4, 4,
            57, 55, 55, 7, 5, 5, 11, 9, 9,
            52, 50, 50, 2, 0, 0, 6, 4, 4,
            52, 50, 50, 2, 0, 0, 6, 4, 4,
            57, 55, 55, 7, 5, 5, 11, 9, 9,
            56, 54, 54, 6, 4, 4, 10, 8, 8,
            56, 54, 54, 6, 4, 4, 10, 8, 8,
            61, 59, 59, 11, 9, 9, 15, 13, 13
            ], index=four_alts)
    }

    assert list(hh_utils.keys()) == list(expected.keys())
    for k in expected:
        pdt.assert_series_equal(hh_utils[k], expected[k], check_dtype=False)


def test_apply_final_rules(hh_utils, final_rules, people, hh_id_col):
    expected = hh_utils.copy()
    expected[8] = pd.Series([
        0, 0, 0, 2, 0, 0, 6, 4, 4,
        0, 0, 0, 2, 0, 0, 6, 4, 4,
        0, 0, 0, 7, 5, 5, 11, 9, 9,
        0, 0, 0, 2, 0, 0, 6, 4, 4,
        0, 0, 0, 2, 0, 0, 6, 4, 4,
        0, 0, 0, 7, 5, 5, 11, 9, 9,
        0, 0, 0, 6, 4, 4, 10, 8, 8,
        0, 0, 0, 6, 4, 4, 10, 8, 8,
        0, 0, 0, 11, 9, 9, 15, 13, 13
        ], index=expected[7].index)

    cdap.apply_final_rules(hh_utils, people, hh_id_col, final_rules)

    for k in expected:
        pdt.assert_series_equal(hh_utils[k], expected[k], check_dtype=False)


def test_apply_all_people(hh_utils, all_people):
    all_people.at["('Mandatory',) * 3", 'Value'] = 300
    all_people.at["('Home',) * 4", 'Value'] = 500

    expected = hh_utils.copy()
    expected[5] = pd.Series([
        301, 1, 2, 3, 3, 4, 1, 1, 2,
        0, 0, 1, 2, 2, 3, 0, 0, 1,
        0, 0, 1, 2, 2, 3, 0, 0, 1,
        ], index=hh_utils[5].index)
    expected[6] = pd.Series([
        302, 2, 4, 2, 2, 4, 5, 5, 7,
        0, 0, 2, 0, 0, 2, 3, 3, 5,
        0, 0, 2, 0, 0, 2, 3, 3, 5,
        ], index=hh_utils[6].index)
    expected[7] = pd.Series([
        4, 8, 4, 8, 12, 8, 4, 8, 4,
        3, 7, 3, 7, 11, 7, 3, 7, 3,
        3, 7, 3, 7, 11, 7, 3, 7, 3,
        1, 5, 1, 5, 9, 5, 1, 5, 1,
        0, 4, 0, 4, 8, 4, 0, 4, 0,
        0, 4, 0, 4, 8, 4, 0, 4, 0,
        1, 5, 1, 5, 9, 5, 1, 5, 1,
        0, 4, 0, 4, 8, 4, 0, 4, 0,
        0, 4, 0, 4, 8, 4, 0, 4, 500
        ], index=hh_utils[7].index)
    expected[8] = pd.Series([
        52, 50, 50, 2, 0, 0, 6, 4, 4,
        52, 50, 50, 2, 0, 0, 6, 4, 4,
        57, 55, 55, 7, 5, 5, 11, 9, 9,
        52, 50, 50, 2, 0, 0, 6, 4, 4,
        52, 50, 50, 2, 0, 0, 6, 4, 4,
        57, 55, 55, 7, 5, 5, 11, 9, 9,
        56, 54, 54, 6, 4, 4, 10, 8, 8,
        56, 54, 54, 6, 4, 4, 10, 8, 8,
        61, 59, 59, 11, 9, 9, 15, 13, 513
        ], index=hh_utils[8].index)

    cdap.apply_all_people(hh_utils, all_people)

    for k in expected:
        pdt.assert_series_equal(hh_utils[k], expected[k], check_dtype=False)


def test_make_household_choices(hh_choices):
    expected = pd.Series([
        ('Mandatory',),
        ('Home',),
        ('Mandatory', 'Mandatory'),
        ('NonMandatory', 'NonMandatory'),
        ('Mandatory', 'NonMandatory', 'Home'),
        ('Mandatory', 'Home', 'Home'),
        ('Mandatory', 'Mandatory', 'NonMandatory', 'NonMandatory'),
        ('Home', 'Home', 'Mandatory', 'NonMandatory')],
        index=range(1, 9))
    pdt.assert_series_equal(hh_choices, expected)


def test_household_choices_to_people(hh_choices, people):
    people_choices = cdap.household_choices_to_people(hh_choices, people)
    expected = pd.Series([
        'Mandatory',
        'Home',
        'Mandatory', 'Mandatory',
        'NonMandatory', 'NonMandatory',
        'Mandatory', 'NonMandatory', 'Home',
        'Mandatory', 'Home', 'Home',
        'Mandatory', 'Mandatory', 'NonMandatory', 'NonMandatory',
        'Home', 'Home', 'Mandatory', 'NonMandatory'],
        index=people.index)
    pdt.assert_series_equal(people_choices, expected)


def test_run_cdap(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
        final_rules, all_people, random_seed):
    people_choices = cdap.run_cdap(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
        final_rules, all_people)
    expected = pd.Series([
        'Mandatory',
        'Home',
        'Mandatory', 'Mandatory',
        'NonMandatory', 'NonMandatory',
        'Mandatory', 'NonMandatory', 'Home',
        'Mandatory', 'Home', 'Home',
        'Mandatory', 'Mandatory', 'NonMandatory', 'NonMandatory',
        'Home', 'Home', 'Home', 'NonMandatory'],
        index=people.index)
    pdt.assert_series_equal(people_choices, expected)
