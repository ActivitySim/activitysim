import datetime
import os
import time
from multiprocessing import Pipe, Process

import psutil


def record_memory_usage(
    logstream, event="", event_idx=-1, measure_uss=False, measure_cpu=False, pid=None
):

    if pid is None:
        pid = os.getpid()
    current_process = psutil.Process(pid)
    with current_process.oneshot():
        process_name = current_process.name()

        if measure_uss:
            try:
                info = current_process.memory_full_info()
                uss = info.uss
            except (PermissionError, psutil.AccessDenied, RuntimeError):
                info = current_process.memory_info()
                uss = 0
        else:
            info = current_process.memory_info()
            uss = 0

        if measure_cpu:
            cpu_pct = current_process.cpu_percent()
        else:
            cpu_pct = -1

    full_rss = rss = info.rss

    num_children = 0
    for child in current_process.children(recursive=True):
        try:
            child_info = child.memory_info()
            full_rss += child_info.rss
            num_children += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            pass

    timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")  # sortable

    print(
        f"{process_name},"
        f"{pid},"
        f"{int(rss)},"
        f"{int(full_rss)},"
        f"{int(uss)},"
        f"{cpu_pct},"
        f"{event_idx},"
        f"{event},"
        f"{num_children},"
        f"{timestamp}",
        file=logstream,
    )


def monitor_memory_usage(
    pid,
    conn,
    interval=0.5,
    flush_interval=5,
    filename="/tmp/sidecar.csv",
    measure_uss=True,
    measure_cpu=True,
):
    event = ""
    event_idx = 0
    last_flush = time.time()
    if measure_cpu:
        psutil.cpu_percent()
    with open(filename, "w") as stream:
        MEM_LOG_HEADER = (
            "process,pid,rss,full_rss,uss,cpu,event_idx,event,children,time"
        )
        print(MEM_LOG_HEADER, file=stream)
        while True:
            # timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")  # sortable
            # print(f"{timestamp} [{event}]", file=stream)
            record_memory_usage(
                stream,
                event=event,
                event_idx=event_idx,
                measure_uss=measure_uss,
                measure_cpu=measure_cpu,
                pid=pid,
            )
            if conn.poll(interval):
                event_ = conn.recv()
                if event_ != event:
                    event_idx += 1
                    event = event_
            else:
                pass
            now = time.time()
            if now > last_flush + flush_interval:
                stream.flush()
                last_flush = now
            if event == "STOP":
                stream.flush()
                break


class MemorySidecar:
    def __init__(self, filename="/tmp/sidecar.csv"):
        self.local_conn, child_conn = Pipe()
        self.sidecar_process = Process(
            target=monitor_memory_usage,
            args=(os.getpid(), child_conn),
            kwargs=dict(filename=filename),
        )
        self.sidecar_process.start()

    def stop(self):
        self.set_event("STOP")
        self.sidecar_process.join(timeout=5)
        if self.sidecar_process.exitcode is None:
            self.sidecar_process.kill()
        print("memory sidecar stopped")

    def set_event(self, event):
        try:
            self.local_conn.send(str(event))
        except BrokenPipeError:
            pass
