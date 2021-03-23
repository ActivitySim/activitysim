
# ActivitySim
# See full license in LICENSE.txt.
import time
import datetime
import psutil
import logging
import gc


from activitysim.core import config
from activitysim.core import inject

logger = logging.getLogger(__name__)

MEM = {}
HWM = {}
DEFAULT_TICK_LEN = 30


def force_garbage_collect():
    was_disabled = not gc.isenabled()
    if was_disabled:
        gc.enable()
    gc.collect()
    if was_disabled:
        gc.disable()


def GB(bytes):
    gb = (bytes / (1024 * 1024 * 1024.0))
    return round(gb, 2)


def init_trace(tick_len=None, file_name="mem.csv", write_header=False):
    MEM['tick'] = 0
    if file_name is not None:
        MEM['file_name'] = file_name
    if tick_len is None:
        MEM['tick_len'] = DEFAULT_TICK_LEN
    else:
        MEM['tick_len'] = tick_len

    logger.info("init_trace file_name %s" % file_name)

    # - check for optional process name prefix
    MEM['prefix'] = inject.get_injectable('log_file_prefix', 'main')

    if write_header:
        with config.open_log_file(file_name, 'w') as log_file:
            print("process,time,rss,used,available,percent,event", file=log_file)


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
            print("%s high water mark %s: %.2f timestamp: %s label: %s" %
                  (MEM['prefix'], tag, hwm['mark'], hwm['timestamp'], hwm['label']), file=log_file)


def trace_memory_info(event=''):

    if not MEM:
        return

    last_tick = MEM['tick']
    tick_len = MEM['tick_len'] or float('inf')

    t = time.time()
    if (t - last_tick < tick_len) and not event:
        return

    force_garbage_collect()

    vmi = psutil.virtual_memory()

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

    if event:
        logger.info(f"trace_memory_info {event} rss: {GB(rss)}GB used: {GB(vmi.used)} GB percent: {vmi.percent}%")

    with config.open_log_file(MEM['file_name'], 'a') as output_file:

        print("%s, %s, %.2f, %.2f, %.2f, %s%%, %s" %
              (MEM['prefix'],
               timestamp,
               GB(rss),
               GB(vmi.used),
               GB(vmi.available),
               vmi.percent,
               event), file=output_file)


def get_rss():

    mi = psutil.Process().memory_info()

    # cur_mem = mi.vms
    cur_mem = mi.rss

    return cur_mem
