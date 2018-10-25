
# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from builtins import input

import psutil
import logging
import contextlib
import time
import inspect
import gc

logger = logging.getLogger(__name__)

MEM = {}


def force_garbage_collect():
    gc.collect()


def GB(bytes):
    gb = (bytes / (1024 * 1024 * 1024.0))
    return "%s GB" % (round(gb, 2), )


def format_elapsed_time(t):
    return "%s seconds (%s minutes)" % (round(t, 3), round(t / 60.0, 1))


def set_pause_threshold(gb):
    bytes = gb * 1024 * 1024 * 1024
    MEM['pause'] = bytes


def _track_memory_info(trace_label):

    gc.collect()
    mi = psutil.Process().memory_info()
    # logger.debug("memory_info: rss: %s vms: %s trace_label: %s" %
    #              (GB(mi.rss), GB(mi.vms), trace_label))

    cur_mem = mi.vms

    if cur_mem > MEM.get('high_water_mark', 0):
        MEM['high_water_mark'] = cur_mem
        MEM['high_water_mark_trace_label'] = trace_label
        logger.debug(
            "memory_info new high_water_mark: %s trace_label: %s" % (GB(cur_mem), trace_label,))

    if 'pause' in MEM and cur_mem > MEM['pause']:
        MEM['pause'] = cur_mem
        input("Return to continue: ")

    return cur_mem


def log_memory_info(trace_label):

    _track_memory_info(trace_label)

    mi = psutil.Process().memory_info()
    logger.debug("memory_info: rss: %s vms: %s trace_label: %s" %
                 (GB(mi.rss), GB(mi.vms), trace_label))


def log_mem_high_water_mark():
    if 'high_water_mark' in MEM:
        logger.info("mem high_water_mark %s in %s" %
                    (GB(MEM['high_water_mark']), MEM['high_water_mark_trace_label']), )


@contextlib.contextmanager
def trace(trace_label, tag, callers_logger, level=logging.DEBUG):
    """
    A context manager to log delta time and memory to execute a block

    Parameters
    ----------
    msg : str
    callers_logger : logging.Logger
        logger passed from caller's context
    level : int, optional
        Level at which to log, passed to ``logger.log``.

    """
    callerframerecord = inspect.stack()[2]
    caller_name = inspect.getframeinfo(callerframerecord[0]).function

    trace_label = "%s.%s" % (trace_label, tag)
    msg = "%s.%s" % (caller_name, tag)

    prev_mem = _track_memory_info("%s.before" % trace_label)
    t = time.time()
    yield
    t = time.time() - t
    post_mem = _track_memory_info("%s.after" % trace_label)

    delta_mem = post_mem - prev_mem

    callers_logger.log(level, "Time to perform %s : %s memory: %s (%s)" %
                       (msg, format_elapsed_time(t), GB(post_mem), GB(delta_mem)))
