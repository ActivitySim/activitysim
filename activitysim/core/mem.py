
# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from builtins import input

import psutil
import logging
import gc

logger = logging.getLogger(__name__)


def force_garbage_collect():
    gc.collect()


# def GB(bytes):
#     gb = (bytes / (1024 * 1024 * 1024.0))
#     return "%s GB" % (round(gb, 2), )


def track_memory_info():

    mi = psutil.Process().memory_info()
    # logger.debug("memory_info: rss: %s vms: %s trace_label: %s" %
    #              (GB(mi.rss), GB(mi.vms), trace_label))

    # cur_mem = mi.vms
    cur_mem = mi.rss

    return cur_mem
