
# ActivitySim
# See full license in LICENSE.txt.

import datetime

import gc
import glob
import logging
import multiprocessing
import os
import psutil
import time

import numpy as np
import pandas as pd

from activitysim.core import config
from activitysim.core import inject
from activitysim.core import util

from activitysim.core.util import GB

logger = logging.getLogger(__name__)


HWM = {}

MEM_TICK_LEN = 30
LAST_MEM_TICK = 0

MEM_LOG_FILE_NAME = "mem.csv"
OMNIBUS_LOG_FILE_NAME = f"omnibus_mem.csv"  # overwrite when consolidating


def consolidate_logs():

    # if we are overwriting MEM_LOG_FILE then presumably we want to delete any subprocess files
    delete_originals = (MEM_LOG_FILE_NAME == OMNIBUS_LOG_FILE_NAME)

    glob_file_name = config.log_file_path(f"*{MEM_LOG_FILE_NAME}", prefix=False)
    logger.debug(f"chunk.consolidate_logs reading glob {glob_file_name}")
    glob_files = glob.glob(glob_file_name)

    if len(glob_files) > 1 or (MEM_LOG_FILE_NAME != OMNIBUS_LOG_FILE_NAME):

        omnibus_df = pd.concat((pd.read_csv(f, comment='#') for f in glob_files))
        omnibus_df = omnibus_df.sort_values(by='time')

        output_path = config.log_file_path(OMNIBUS_LOG_FILE_NAME, prefix=False)

        if delete_originals:
            util.delete_files(glob_files, 'mem.consolidate_logs')

        logger.debug(f"chunk.consolidate_logs writing omnibus log to {output_path}")
        omnibus_df.to_csv(output_path, mode='w', index=False)


def check_global_hwm(tag, value, label):

    assert value is not None

    hwm = HWM.setdefault(tag, {})

    is_new_hwm = value > hwm.get('mark', 0)
    if is_new_hwm:
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        hwm['mark'] = value
        hwm['timestamp'] = timestamp
        hwm['label'] = label

    return is_new_hwm


def log_hwm():

    for tag in HWM:
        hwm = HWM[tag]
        logger.info("high water mark %s: %.2f timestamp: %s label: %s" %
                    (tag, hwm['mark'], hwm['timestamp'], hwm['label']))


def trace_memory_info(event=''):

    global LAST_MEM_TICK

    process_name = multiprocessing.current_process().name

    percent = psutil.virtual_memory().percent

    current_process = psutil.Process()
    info = current_process.memory_full_info()

    num_children = 0
    base_rss = full_rss = info.rss
    for child in current_process.children(recursive=True):
        try:
            full_rss += child.memory_info().rss
            num_children += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            # print(f"'n###'n{e}")
            pass

    noteworthy = not event.endswith('.idle')
    noteworthy = check_global_hwm('rss', full_rss, event) or noteworthy

    t = time.time()
    if (t - LAST_MEM_TICK > MEM_TICK_LEN):
        noteworthy = True
        LAST_MEM_TICK = t

    if noteworthy:
        logger.debug(f"trace_memory_info {event} base_rss: {GB(base_rss)} full_rss: {GB(full_rss)} percent: {percent}%")

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")  # sortable

        MEM_LOG_HEADER = "process,base_rss,full_rss,percent,event,children,time"
        with config.open_log_file(MEM_LOG_FILE_NAME, 'a', header=MEM_LOG_HEADER, prefix=True) as log_file:
            print(f"{process_name},"
                  f"{GB(base_rss)},"
                  f"{GB(full_rss)},"
                  f"{round(percent,2)}%,"
                  f"{event},"
                  f"{num_children},"
                  f"{timestamp}",
                  file=log_file)


def get_rss(force_garbage_collect=False):

    if force_garbage_collect:
        was_disabled = not gc.isenabled()
        if was_disabled:
            gc.enable()
        gc.collect()
        if was_disabled:
            gc.disable()

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
        except Exception:
            obj = data_buffer
        data = np.ctypeslib.as_array(obj)
        data_size = data.nbytes

        shared_size += data_size

    return shared_size
