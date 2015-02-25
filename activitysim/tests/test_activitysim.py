import os.path

import numpy.testing as npt

from .. import activitysim as asim


def test_read_model_spec():
    spec = asim.read_model_spec(
        os.path.join(os.path.dirname(__file__), 'data', 'sample_spec.csv'),
        description_name='description', expression_name='expression')

    assert len(spec) == 3
    assert spec.index.name == 'expression'
    assert list(spec.columns) == ['alt0', 'alt1']


def test_identity_matrix():
    names = ['a', 'b', 'c']
    i = asim.identity_matrix(names)

    assert list(i.columns) == names
    assert list(i.index) == names

    npt.assert_array_equal(
        i.as_matrix(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
