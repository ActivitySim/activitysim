
# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402


import time
import datetime
import psutil
import logging
import gc
import sys

from activitysim.core import config

logger = logging.getLogger(__name__)

MEM = {}


def force_garbage_collect():
    gc.collect()


def GB(bytes):
    return "%.2f" % (bytes / (1024 * 1024 * 1024.0))


def init_trace(tick_len=5, file_name="mem.csv"):
    MEM['tick'] = 0
    if file_name is not None:
        MEM['file_name'] = file_name
    if tick_len is not None:
        MEM['tick_len'] = tick_len


def trace_memory_info(event=''):

    if not MEM:
        return

    last_tick = MEM['tick']

    t = time.time()
    if (t - last_tick < MEM['tick_len']) and not event:
        return

    vmi = psutil.virtual_memory()

    if last_tick == 0:
        mode = 'wb' if sys.version_info < (3,) else 'w'
        with open(config.output_file_path(MEM['file_name']), mode) as file:
            print("time,rss,used,available,percent,event", file=file)

    MEM['tick'] = t

    current_process = psutil.Process()
    rss = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            pass

    # logger.debug("memory_info: rss: %s available: %s percent: %s"
    #              %  (GB(mi.rss), GB(vmi.available), GB(vmi.percent)))

    mode = 'ab' if sys.version_info < (3,) else 'a'
    with open(config.output_file_path(MEM['file_name']), mode) as file:

        print("%s, %s, %s, %s, %s%%, %s" %
              (datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
               GB(rss),
               GB(vmi.used),
               GB(vmi.available),
               vmi.percent,
               event), file=file)


def get_memory_info():

    mi = psutil.Process().memory_info()

    # cur_mem = mi.vms
    cur_mem = mi.rss

    return cur_mem
