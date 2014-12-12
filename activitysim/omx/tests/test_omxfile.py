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


def test_open_readonly_hdf5_file(tmpomx):
    f = tables.open_file(tmpomx, 'w')
    f.close()
    assert os.path.exists(tmpomx)
    f = open_omxfile(tmpomx, 'r')
    f.close()


def test_setitem_getitem(tmpomx, ones5x5):
    with open_omxfile(tmpomx, 'w') as f:
        f['m1'] = ones5x5
        npt.assert_array_equal(f['m1'], ones5x5)
        assert f.shape() == (5, 5)


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


def test_get_length_of_file(tmpomx, ones5x5):
    mats = ['m{}'.format(x) for x in range(5)]
    with open_omxfile(tmpomx, 'w') as f:
        for m in mats:
            f[m] = ones5x5

        assert f.list_matrices() == mats


def test_shape(tmpomx):
    with open_omxfile(tmpomx, mode='w', shape=(5, 5)) as f:
        assert f.shape() == (5, 5)

        with pytest.raises(ShapeError):
            f['test'] = np.ones(10)
