# ActivitySim
# See full license in LICENSE.txt.

import datetime
import gc
import glob
import logging
import multiprocessing
import os
import platform
import threading
import time

import numpy as np
import pandas as pd
import psutil

from activitysim.core import config, inject, util

logger = logging.getLogger(__name__)

USS = True

GLOBAL_HWM = {}  # to avoid confusion with chunk local hwm

MEM_TRACE_TICK_LEN = 5
MEM_PARENT_TRACE_TICK_LEN = 15
MEM_SNOOP_TICK_LEN = 5
MEM_TICK = 0

MEM_LOG_FILE_NAME = "mem.csv"
OMNIBUS_LOG_FILE_NAME = f"omnibus_mem.csv"

SUMMARY_BIN_SIZE_IN_SECONDS = 15

mem_log_lock = threading.Lock()


def time_bin(timestamps):
    bins_size_in_seconds = SUMMARY_BIN_SIZE_IN_SECONDS
    epoch = pd.Timestamp("1970-01-01")
    seconds_since_epoch = (timestamps - epoch) // pd.Timedelta("1s")
    bin = seconds_since_epoch - (seconds_since_epoch % bins_size_in_seconds)

    return pd.to_datetime(bin, unit="s", origin="unix")


def consolidate_logs():
    """
    Consolidate and aggregate subprocess mem logs
    """

    if not config.setting("multiprocess", False):
        return

    delete_originals = not config.setting("keep_mem_logs", False)
    omnibus_df = []

    # for each multiprocess step
    multiprocess_steps = config.setting("multiprocess_steps", [])
    for step in multiprocess_steps:
        step_name = step.get("name", None)

        logger.debug(f"mem.consolidate_logs for step {step_name}")

        glob_file_name = config.log_file_path(
            f"{step_name}*{MEM_LOG_FILE_NAME}", prefix=False
        )
        glob_files = glob.glob(glob_file_name)

        if not glob_files:
            continue

        logger.debug(
            f"mem.consolidate_logs consolidating {len(glob_files)} logs for {step_name}"
        )

        # for each individual log
        step_summary_df = []
        for f in glob_files:
            df = pd.read_csv(f, comment="#")

            df = df[["rss", "uss", "event", "time"]]

            df.rss = df.rss.astype(np.int64)
            df.uss = df.uss.astype(np.int64)

            df["time"] = time_bin(
                pd.to_datetime(df.time, errors="coerce", format="%Y/%m/%d %H:%M:%S")
            )

            # consolidate events (duplicate rows should be idle steps (e.g. log_rss)
            df = (
                df.groupby("time")
                .agg(
                    rss=("rss", "max"),
                    uss=("uss", "max"),
                )
                .reset_index(drop=False)
            )

            step_summary_df.append(df)  # add step_df to step summary

        # aggregate the individual the logs into a single step log
        step_summary_df = pd.concat(step_summary_df)
        step_summary_df = (
            step_summary_df.groupby("time")
            .agg(rss=("rss", "sum"), uss=("uss", "sum"), num_files=("rss", "size"))
            .reset_index(drop=False)
        )
        step_summary_df = step_summary_df.sort_values("time")

        step_summary_df["step"] = step_name

        # scale missing values (might be missing idle steps for some chunk_tags)
        scale = (
            1
            + (len(glob_files) - step_summary_df.num_files) / step_summary_df.num_files
        )
        for c in ["rss", "uss"]:
            step_summary_df[c] = (step_summary_df[c] * scale).astype(np.int64)

        step_summary_df["scale"] = scale
        del step_summary_df["num_files"]  # do we want to keep track of scale factor?

        if delete_originals:
            util.delete_files(glob_files, f"mem.consolidate_logs.{step_name}")

        # write aggregate step log
        output_path = config.log_file_path(f"mem_{step_name}.csv", prefix=False)
        logger.debug(
            f"chunk.consolidate_logs writing step summary log for step {step_name} to {output_path}"
        )
        step_summary_df.to_csv(output_path, mode="w", index=False)

        omnibus_df.append(step_summary_df)  # add step summary to omnibus

    # aggregate the step logs into a single omnibus log ordered by timestamp
    omnibus_df = pd.concat(omnibus_df)
    omnibus_df = omnibus_df.sort_values("time")

    output_path = config.log_file_path(OMNIBUS_LOG_FILE_NAME, prefix=False)
    logger.debug(f"chunk.consolidate_logs writing omnibus log to {output_path}")
    omnibus_df.to_csv(output_path, mode="w", index=False)


def check_global_hwm(tag, value, label):

    assert value is not None

    hwm = GLOBAL_HWM.setdefault(tag, {})

    is_new_hwm = value > hwm.get("mark", 0) or not hwm
    if is_new_hwm:
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        hwm["mark"] = value
        hwm["timestamp"] = timestamp
        hwm["label"] = label

    return is_new_hwm


def log_global_hwm():

    process_name = multiprocessing.current_process().name

    for tag in GLOBAL_HWM:
        hwm = GLOBAL_HWM[tag]
        value = hwm.get("mark", 0)
        logger.info(
            f"{process_name} high water mark {tag}: {util.INT(value)} ({util.GB(value)}) "
            f"timestamp: {hwm.get('timestamp', '<none>')} label:{hwm.get('label', '<none>')}"
        )


def trace_memory_info(event, trace_ticks=0):

    global MEM_TICK

    tick = time.time()
    if trace_ticks and (tick - MEM_TICK < trace_ticks):
        return
    MEM_TICK = tick

    process_name = multiprocessing.current_process().name
    pid = os.getpid()

    current_process = psutil.Process()

    if USS:
        info = current_process.memory_full_info()
        uss = info.uss
    else:
        info = current_process.memory_info()
        uss = 0

    full_rss = rss = info.rss

    num_children = 0
    for child in current_process.children(recursive=True):
        try:
            child_info = child.memory_info()
            full_rss += child_info.rss
            num_children += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            pass

    noteworthy = (
        True  # any reason not to always log this if we are filtering idle ticks?
    )

    noteworthy = (num_children > 0) or noteworthy
    noteworthy = check_global_hwm("rss", full_rss or rss, event) or noteworthy
    noteworthy = check_global_hwm("uss", uss, event) or noteworthy

    if noteworthy:

        # logger.debug(f"trace_memory_info {event} "
        #              f"rss: {GB(full_rss) if num_children else GB(rss)} "
        #              f"uss: {GB(rss)} ")

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")  # sortable

        with mem_log_lock:
            MEM_LOG_HEADER = "process,pid,rss,full_rss,uss,event,children,time"
            with config.open_log_file(
                MEM_LOG_FILE_NAME, "a", header=MEM_LOG_HEADER, prefix=True
            ) as log_file:
                print(
                    f"{process_name},"
                    f"{pid},"
                    f"{util.INT(rss)},"  # want these as ints so we can plot them...
                    f"{util.INT(full_rss)},"
                    f"{util.INT(uss)},"
                    f"{event},"
                    f"{num_children},"
                    f"{timestamp}",
                    file=log_file,
                )

    # return rss and uss for optional use by interested callers
    return full_rss or rss, uss


def get_rss(force_garbage_collect=False, uss=False):

    if force_garbage_collect:
        was_disabled = not gc.isenabled()
        if was_disabled:
            gc.enable()
        gc.collect()
        if was_disabled:
            gc.disable()

    if uss:
        info = psutil.Process().memory_full_info()
        return info.rss, info.uss
    else:
        info = psutil.Process().memory_info()
        return info.rss, 0


def shared_memory_size(data_buffers=None):
    """
    return total size of the multiprocessing shared memory block in data_buffers

    Returns
    -------

    """

    shared_size = 0

    if data_buffers is None:
        data_buffers = inject.get_injectable("data_buffers", {})

    for k, data_buffer in data_buffers.items():
        try:
            obj = data_buffer.get_obj()
        except Exception:
            obj = data_buffer
        data = np.ctypeslib.as_array(obj)
        data_size = data.nbytes

        shared_size += data_size

    return shared_size
