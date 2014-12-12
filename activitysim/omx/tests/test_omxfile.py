import os
import tempfile

import numpy as np
import pytest
import tables

from .. import open_omxfile, ShapeError


def add_m1_node(f):
    f.create_matrix('m1', obj=np.ones((7, 7)))


@pytest.fixture
def tmpomx(request):
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name

    def cleanup():
        if os.path.exists(fname):
            os.remove(fname)
    request.addfinalizer(cleanup)

    return fname


def test_create_file(tmpomx):
    f = open_omxfile(tmpomx, 'w')
    f.close()
    assert os.path.exists(tmpomx)


def test_open_readonly_hdf5_file(tmpomx):
    f = tables.open_file(tmpomx, 'w')
    f.close()
    f = open_omxfile(tmpomx, 'r')
    f.close()


def test_add_numpy_matrix_using_brackets(tmpomx):
    with open_omxfile(tmpomx, 'w') as f:
        f['m1'] = np.ones((5, 5))


def test_add_np_matrix_using_create_matrix(tmpomx):
    with open_omxfile(tmpomx, 'w') as f:
        f.create_matrix('m1', obj=np.ones((5, 5)))

        # test check for shape matching
        with pytest.raises(ShapeError):
            f.create_matrix('m2', obj=np.ones((8, 8)))


def test_add_matrix_to_readonly_file(tmpomx):
    with open_omxfile(tmpomx, 'w') as f:
        f['m2'] = np.ones((7, 7))

    with open_omxfile(tmpomx, 'r') as f:
        with pytest.raises(tables.FileModeError):
            add_m1_node(f)


def test_add_matrix_with_same_name(tmpomx):
    with open_omxfile(tmpomx, 'w') as f:
        add_m1_node(f)

        # now add m1 again:
        with pytest.raises(tables.NodeError):
            add_m1_node(f)


def test_get_length_of_file(tmpomx):
    mats = ['m{}'.format(x) for x in range(5)]
    with open_omxfile(tmpomx, 'w') as f:
        for m in mats:
            f[m] = np.ones((5, 5))

        assert f.list_matrices() == mats


def test_shapeerror(tmpomx):
    with open_omxfile(tmpomx, mode='w', shape=(5, 5)) as f:
        with pytest.raises(ShapeError):
            f['test'] = np.ones(10)
