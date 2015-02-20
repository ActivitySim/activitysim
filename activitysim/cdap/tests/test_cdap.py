import os.path

import pandas as pd
import pandas.util.testing as pdt
import pytest

from .. import cdap


@pytest.fixture(scope='module')
def people():
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'people.csv'))


@pytest.fixture(scope='module')
def hh_id_col():
    return 'household'


@pytest.fixture(scope='module')
def p_type_col():
    return 'ptype'


def test_make_interactions(people, hh_id_col, p_type_col):
    expected_two = pd.Series(
        [
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
        ],
        index=[
            3, 4, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
            13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
            17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20
        ])

    expected_three = pd.Series(
        [
            '322', '232', '232',  # household 5; people 7, 8, 9
            '131', '311', '113',  # household 6; people 10, 11, 12
            '132', '132', '122',  # household 7; person 13
            '312', '312', '322',  # household 7; person 14
            '213', '212', '232',  # household 7; person 15
            '213', '212', '232',  # household 7; person 16
            '132', '131', '121',  # household 8; person 17
            '312', '311', '321',  # household 8; person 18
            '213', '211', '231',  # household 8; person 19
            '113', '112', '132'   # household 8; person 20
        ],
        index=[
            7, 8, 9, 10, 11, 12,
            13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
            17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20
        ])

    two, three = cdap.make_interactions(people, hh_id_col, p_type_col)

    pdt.assert_series_equal(two, expected_two)
    pdt.assert_series_equal(three, expected_three)
