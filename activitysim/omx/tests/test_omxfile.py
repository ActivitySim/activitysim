import tables,omx,numpy,os
from nose.tools import *

def setup_clean():
    try:
        os.remove('test1.omx')
    except:
        pass


def setup_empty_file():
    try:
        os.remove('test1.omx')
        f = omx.openFile('test.omx','w')
        f.close()
    except:
        pass


def teardown_clean():
    try:
        os.remove('test2.omx')
    except:
        pass


@with_setup(setup_clean,teardown_clean)
def test_create_file():
    f = omx.openFile('test1.omx','w')
    f.close()
    assert(os.path.exists('test1.omx'))

def test_open_readonly_hdf5_file():
    f = tables.openFile('test2.omx','w')
    f.close()
    f = omx.openFile('test2.omx','r')
    f.close() 

def test_add_numpy_matrix_using_brackets():
    f = omx.openFile('test3.omx','w')
    f['m1'] = numpy.ones((5,5))
    f.close()

def test_add_numpy_matrix_using_create_matrix():
    f = omx.openFile('test4.omx','w')
    f.createMatrix('m1', obj=numpy.ones((5,5)))
    f.close()

def test_add_matrix_to_readonly_file():
    f = omx.openFile('test6.omx','w')
    f['m2'] = numpy.ones((5,5))
    f.close()
    f = omx.openFile('test6.omx','r')
    assert_raises(tables.FileModeError, add_m1_node, f)
    f.close() 

def test_add_matrix_with_same_name():
    f = omx.openFile('test5.omx','w')
    add_m1_node(f)
    # now add m1 again:
    assert_raises(tables.NodeError, add_m1_node, f)
    f.close()

@with_setup(setup_clean,teardown_clean)
def test_get_length_of_file():
    f = omx.openFile('test7.omx','w')
    f['m1'] = numpy.ones((5,5))
    f['m2'] = numpy.ones((5,5))
    f['m3'] = numpy.ones((5,5))
    f['m4'] = numpy.ones((5,5))
    f['m5'] = numpy.ones((5,5))
    assert(len(f)==5)
    assert(len(f.listMatrices())==5)
    f.close()

def add_m1_node(f):
    f.createMatrix('m1', obj=numpy.ones((7,7)))

