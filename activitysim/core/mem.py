
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


from activitysim.core import config

logger = logging.getLogger(__name__)

MEM = {}
HWM = {}
DEFAULT_TICK_LEN = 30


def force_garbage_collect():
    gc.collect()


def GB(bytes):
    return (bytes / (1024 * 1024 * 1024.0))


def init_trace(tick_len=None, file_name="mem.csv"):
    MEM['tick'] = 0
    if file_name is not None:
        MEM['file_name'] = file_name
    if tick_len is None:
        MEM['tick_len'] = DEFAULT_TICK_LEN
    else:
        MEM['tick_len'] = tick_len

    logger.info("init_trace file_name %s" % file_name)


def trace_hwm(tag, value, timestamp, label):

    hwm = HWM.setdefault(tag, {})

    if value > hwm.get('mark', 0):
        hwm['mark'] = value
        hwm['timestamp'] = timestamp
        hwm['label'] = label


def log_hwm():

    for tag in HWM:
        hwm = HWM[tag]
        logger.info("high water mark %s: %.2f timestamp: %s label: %s" %
                    (tag, hwm['mark'], hwm['timestamp'], hwm['label']))

    with config.open_log_file(MEM['file_name'], 'a') as log_file:
        for tag in HWM:
            hwm = HWM[tag]
            print("high water mark %s: %.2f timestamp: %s label: %s" %
                  (tag, hwm['mark'], hwm['timestamp'], hwm['label']), file=log_file)


def trace_memory_info(event=''):

    if not MEM:
        return

    last_tick = MEM['tick']
    tick_len = MEM['tick_len'] or float('inf')

    t = time.time()
    if (t - last_tick < tick_len) and not event:
        return

    vmi = psutil.virtual_memory()

    if last_tick == 0:
        with config.open_log_file(MEM['file_name'], 'w') as log_file:
            print("time,rss,used,available,percent,event", file=log_file)

    MEM['tick'] = t

    current_process = psutil.Process()
    rss = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            pass

    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    trace_hwm('rss', GB(rss), timestamp, event)
    trace_hwm('used', GB(vmi.used), timestamp, event)

    # logger.debug("memory_info: rss: %s available: %s percent: %s"
    #              %  (GB(mi.rss), GB(vmi.available), GB(vmi.percent)))

    with config.open_log_file(MEM['file_name'], 'a') as output_file:

        print("%s, %.2f, %.2f, %.2f, %s%%, %s" %
              (timestamp,
               GB(rss),
               GB(vmi.used),
               GB(vmi.available),
               vmi.percent,
               event), file=output_file)


def get_memory_info():

    mi = psutil.Process().memory_info()

    # cur_mem = mi.vms
    cur_mem = mi.rss

    return cur_mem
