# Licensed under the Apache License, v2.0
# See activitysim/omx/LICENSE.txt
# Modified from the original OMX library available at
# https://github.com/osPlanning/omx/tree/dev/api/python/omx

import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest
import tables

from .. import open_omxfile, ShapeError


@pytest.fixture
def tmpomx(request):
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name

    def cleanup():
        if os.path.exists(fname):
            os.remove(fname)
    request.addfinalizer(cleanup)

    return fname


@pytest.fixture
def ones5x5():
    return np.ones((5, 5))


@pytest.fixture
def basic_omx(request, tmpomx, ones5x5):
    f = open_omxfile(tmpomx, mode='w')
    f['m1'] = ones5x5

    def fin():
        f.close()
    request.addfinalizer(fin)

    return f


def test_open_readonly_hdf5_file(tmpomx):
    f = tables.open_file(tmpomx, 'w')
    f.close()
    assert os.path.exists(tmpomx)
    f = open_omxfile(tmpomx, 'r')
    f.close()


def test_set_get_del(tmpomx, ones5x5):
    with open_omxfile(tmpomx, 'w') as f:
        f['m1'] = ones5x5
        npt.assert_array_equal(f['m1'], ones5x5)
        assert f.shape() == (5, 5)
        del f['m1']
        assert 'm1' not in f


def test_create_matrix(tmpomx, ones5x5):
    with open_omxfile(tmpomx, 'w') as f:
        f.create_matrix('m1', obj=ones5x5)
        npt.assert_array_equal(f['m1'], ones5x5)
        assert f.shape() == (5, 5)

        # test check for shape matching
        with pytest.raises(ShapeError):
            f.create_matrix('m2', obj=np.ones((8, 8)))


def test_add_matrix_to_readonly_file(tmpomx, ones5x5):
    with open_omxfile(tmpomx, 'w') as f:
        f['m2'] = ones5x5

    with open_omxfile(tmpomx, 'r') as f:
        with pytest.raises(tables.FileModeError):
            f['m1'] = ones5x5


def test_add_matrix_with_same_name(tmpomx, ones5x5):
    with open_omxfile(tmpomx, 'w') as f:
        f['m1'] = ones5x5

        # now add m1 again:
        with pytest.raises(tables.NodeError):
            f['m1'] = ones5x5


def test_len_list_iter(tmpomx, ones5x5):
    names = ['m{}'.format(x) for x in range(5)]
    with open_omxfile(tmpomx, 'w') as f:
        for m in names:
            f[m] = ones5x5

        for mat in f:
            npt.assert_array_equal(mat, ones5x5)

        assert len(f) == len(names)
        assert f.list_matrices() == names


def test_shape(tmpomx):
    with open_omxfile(tmpomx, mode='w', shape=(5, 5)) as f:
        assert f.shape() == (5, 5)

        with pytest.raises(ShapeError):
            f['test'] = np.ones(10)


def test_contains(basic_omx):
    assert 'm1' in basic_omx


def test_list_all_attrs(basic_omx, ones5x5):
    basic_omx['m2'] = ones5x5

    assert basic_omx.list_all_attributes() == []

    basic_omx['m1'].attrs['a1'] = 'a1'
    basic_omx['m1'].attrs['a2'] = 'a2'
    basic_omx['m2'].attrs['a2'] = 'a2'
    basic_omx['m2'].attrs['a3'] = 'a3'

    assert basic_omx.list_all_attributes() == ['a1', 'a2', 'a3']


def test_matrices_by_attr(basic_omx, ones5x5):
    bo = basic_omx
    bo['m2'] = ones5x5
    bo['m3'] = ones5x5

    for m in bo:
        m.attrs['a1'] = 'a1'
        m.attrs['a2'] = 'a2'
    bo['m3'].attrs['a2'] = 'a22'
    bo['m3'].attrs['a3'] = 'a3'

    gmba = bo.get_matrices_by_attr

    assert gmba('zz', 'zz') == []
    assert gmba('a1', 'a1') == [bo['m1'], bo['m2'], bo['m3']]
    assert gmba('a2', 'a2') == [bo['m1'], bo['m2']]
    assert gmba('a2', 'a22') == [bo['m3']]
    assert gmba('a3', 'a3') == [bo['m3']]


def test_set_with_carray(basic_omx):
    basic_omx['m2'] = basic_omx['m1']
    npt.assert_array_equal(basic_omx['m2'], basic_omx['m1'])
