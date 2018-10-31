
# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402


import time
import psutil
import logging
import gc
import sys

from activitysim.core import config

logger = logging.getLogger(__name__)

MEM = {'tick': 0}
TICK_LEN = 5


def force_garbage_collect():
    gc.collect()


def GB(bytes):
    return "%.2f" % (bytes / (1024 * 1024 * 1024.0))


def trace_memory_info(event=''):

    last_tick = MEM['tick']

    t = time.time()
    if (t - last_tick < TICK_LEN) and not event:
        return

    vmi = psutil.virtual_memory()

    if last_tick == 0:
        MEM['baseline_tick'] = t
        MEM['baseline_used'] = vmi.used
        mode = 'wb' if sys.version_info < (3,) else 'w'
        with open(config.output_file_path('mem.csv'), mode) as file:
            print("seconds,delta_used,used,available,percent,event", file=file)

    MEM['tick'] = t
    baseline_tick = MEM['baseline_tick']
    baseline_used = MEM['baseline_used']

    # logger.debug("memory_info: rss: %s available: %s percent: %s"
    #              %  (GB(mi.rss), GB(vmi.available), GB(vmi.percent)))

    mode = 'ab' if sys.version_info < (3,) else 'a'
    with open(config.output_file_path('mem.csv'), mode) as file:

        print("%s, %s, %s, %s, %s%%, %s" %
              (int(t - baseline_tick),
               GB(vmi.used - baseline_used),
               GB(vmi.used),
               GB(vmi.available),
               vmi.percent,
               event), file=file)


def get_memory_info():

    mi = psutil.Process().memory_info()

    # cur_mem = mi.vms
    cur_mem = mi.rss

    return cur_mem
