# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd
import pandas.util.testing as pdt
from ..mode import pre_process_expressions, evaluate_expression_list, \
    expand_alternatives, _mode_choice_spec


def test_ppe():
    ret = pre_process_expressions(
        ['1', '$expr.format(var="bar")'],
        {'expr': '@foo * {var}'}
    )
    assert ret[0] == '1'
    assert ret[1] == '@foo * bar'


def test_eel():
    ret = evaluate_expression_list(
        pd.Series(
            ['.7', 'ivt * .7 * COST'],
            index=['ivt', 'ivt_lr']
        ),
        {'COST': 2.0}
    )
    pdt.assert_series_equal(
        ret,
        pd.Series(
            [.7, .98],
            index=['ivt', 'ivt_lr']
        )
    )


def test_ea():
    df = pd.DataFrame({
        "Alternative": ["One", "One,Two"],
        "Other column": [1, 2]
    }).set_index("Alternative")

    df = expand_alternatives(df)

    pdt.assert_series_equal(
        df.reset_index().Alternative,
        pd.Series(
            ["One", "One", "Two"], index=[0, 1, 2], name='Alternative'))

    pdt.assert_series_equal(
        df.reset_index().Rowid,
        pd.Series(
            [0, 1, 1], index=[0, 1, 2], name='Rowid'))

    pdt.assert_series_equal(
        df.reset_index()["Other column"],
        pd.Series(
            [1, 2, 2], index=[0, 1, 2], name='Other column'))


def test_mode_choice_spec():

    spec = pd.DataFrame({
        "Alternative": ["One", "One,Two"],
        "Expression": ['1', '$expr.format(var="bar")'],
        "Work": ['ivt', 'ivt_lr']
    }).set_index(["Expression"])

    coeffs = pd.DataFrame({
        "Work": ['.7', 'ivt * .7 * COST']
    }, index=['ivt', 'ivt_lr'])

    settings = {
        "CONSTANTS": {
            "COST": 2.0
        },
        "VARIABLE_TEMPLATES": {
            'expr': '@foo * {var}'
        }
    }

    df = _mode_choice_spec(spec, coeffs, settings)

    pdt.assert_series_equal(
        df.reset_index().Alternative,
        pd.Series(
            ["One", "One", "Two"], index=[0, 1, 2], name='Alternative'))

    pdt.assert_series_equal(
        df.reset_index().Rowid,
        pd.Series(
            [0, 1, 1], index=[0, 1, 2], name='Rowid'))

    pdt.assert_series_equal(
        df.reset_index()["Work"],
        pd.Series(
            [.7, .98, .98], index=[0, 1, 2], name='Work'))

    pdt.assert_series_equal(
        df.reset_index()["Expression"],
        pd.Series(
            ["1", "@foo * bar", "@foo * bar"],
            index=[0, 1, 2], name='Expression'))
