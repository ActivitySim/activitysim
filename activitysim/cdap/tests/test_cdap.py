import os.path

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
def final_rules():
    return read_model_spec(
        os.path.join(
            os.path.dirname(__file__), 'data', 'cdap_final_rules.csv'))


@pytest.fixture(scope='module')
def hh_id_col():
    return 'household'


@pytest.fixture(scope='module')
def p_type_col():
    return 'ptype'


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

    pdt.assert_frame_equal(two, pd.DataFrame(columns=['interaction']))
    pdt.assert_frame_equal(three, pd.DataFrame(columns=['interaction']))


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
    pdt.assert_frame_equal(three, pd.DataFrame(columns=['interaction']))


def test_apply_final_rules(people, final_rules):
    utilities = pd.DataFrame(
        [[1, 1, 1]] * len(people), index=people.index,
        columns=['Mandatory', 'NonMandatory', 'Home'])
    cdap.apply_final_rules(people, final_rules, utilities)

    assert utilities.loc[19, 'Mandatory'] == 0
