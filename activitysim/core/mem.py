
# ActivitySim
# See full license in LICENSE.txt.
import time
import datetime
import psutil
import logging
import gc
import multiprocessing
import numpy as np

from activitysim.core import config
from activitysim.core import inject

logger = logging.getLogger(__name__)


HWM = {}

MEM_TICK_LEN = 30
LAST_MEM_TICK = 0

MEM_LOG_FILE_NAME = "mem.csv"
MEM_LOG_HEADER = "process,time,rss,uss,children,percent,event,proc"


def GB(bytes):
    gb = (bytes / (1024 * 1024 * 1024.0))
    return round(gb, 2)


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


def trace_memory_info(event=''):

    global LAST_MEM_TICK

    t = time.time()
    if (t - LAST_MEM_TICK < MEM_TICK_LEN) and not event:
        return

    LAST_MEM_TICK = t

    process_name = multiprocessing.current_process().name

    percent = psutil.virtual_memory().percent

    current_process = psutil.Process()
    info = current_process.memory_full_info()
    rss = info.rss
    uss = info.uss

    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    trace_hwm('rss', rss, timestamp, event)
    trace_hwm('uss', uss, timestamp, event)

    if event:
        logger.debug(f"trace_memory_info {event} rss: {GB(rss)}GB uss: {GB(uss)}GB percent: {percent}%")

    with config.open_log_file(MEM_LOG_FILE_NAME, 'a', header=MEM_LOG_HEADER, prefix=True) as log_file:

        print("%s, %s, %.2f, %.2f, %s%%, %s" %
              (process_name,
               timestamp,
               GB(rss),
               GB(uss),
               percent,
               event), file=log_file)


def get_rss(force_garbage_collect=False):

    if force_garbage_collect:
        was_disabled = not gc.isenabled()
        if was_disabled:
            gc.enable()
        gc.collect()
        if was_disabled:
            gc.disable()

    # info = psutil.Process().memory_full_info().uss

    rss = psutil.Process().memory_info().rss

    return rss


def shared_memory_size():
    """
    multiprocessing shared memory appears in teh
    Returns
    -------

    """

    shared_size = 0

    data_buffers = inject.get_injectable('data_buffers', {})
    for k, data_buffer in data_buffers.items():
        try:
            obj = data_buffer.get_obj()
        except:
            obj = data_buffer
        data = np.ctypeslib.as_array(obj)
        data_size = data.nbytes

        shared_size += data_size

    return shared_size
