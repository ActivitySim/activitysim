# OMX package
# release 1

import tables
from File import *
from Exceptions import *

# GLOBAL VARIABLES -----------
__version__ = '0.2'

# GLOBAL FUNCTIONS -----------
def openFile(filename, mode='r', title='', root_uep='/',
             filters=tables.Filters(complevel=1,shuffle=True,fletcher32=False,complib='zlib'),
             shape=None, **kwargs):
    """Open or create a new OMX file. New files will be created with default
       zlib compression enabled."""

    f = File(filename, mode, title, root_uep, filters, **kwargs);

    # add omx structure if file is writable
    if mode != 'r':
        # version number
        if 'OMX_VERSION' not in f.root._v_attrs:
            f.root._v_attrs['OMX_VERSION'] = __version__

        # shape
        if shape:
            f.root._v_attrs['SHAPE'] = shape

        # /data and /lookup folders
        if 'data' not in f.root:
            f.createGroup(f.root,"data")
        if 'lookup' not in f.root:
            f.createGroup(f.root,"lookup")

    return f


if __name__ == "__main__":
    print 'OMX!'


