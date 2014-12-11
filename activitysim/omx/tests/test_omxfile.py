import os
import tempfile

import numpy as np
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
    f = open_omxfile(tmpomx, 'w')
    f['m1'] = np.ones((5, 5))
    f.close()


def test_add_np_matrix_using_create_matrix(tmpomx):
    f = open_omxfile(tmpomx, 'w')
    f.create_matrix('m1', obj=np.ones((5, 5)))

    # test check for shape matching
    with pytest.raises(ShapeError):
        f.create_matrix('m2', obj=np.ones((8, 8)))

    f.close()


def test_add_matrix_to_readonly_file(tmpomx):
    f = open_omxfile(tmpomx, 'w')
    f['m2'] = np.ones((7, 7))
    f.close()
    f = open_omxfile(tmpomx, 'r')

    with pytest.raises(tables.FileModeError):
        add_m1_node(f)

    f.close()


def test_add_matrix_with_same_name(tmpomx):
    f = open_omxfile(tmpomx, 'w')
    add_m1_node(f)
    # now add m1 again:

    with pytest.raises(tables.NodeError):
        add_m1_node(f)

    f.close()


def test_get_length_of_file(tmpomx):
    f = open_omxfile(tmpomx, 'w')
    f['m1'] = np.ones((5, 5))
    f['m2'] = np.ones((5, 5))
    f['m3'] = np.ones((5, 5))
    f['m4'] = np.ones((5, 5))
    f['m5'] = np.ones((5, 5))
    assert len(f) == 5
    assert len(f.list_matrices()) == 5
    f.close()


def add_m1_node(f):
    f.create_matrix('m1', obj=np.ones((7, 7)))
