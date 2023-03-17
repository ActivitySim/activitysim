import logging
import os
import time

import numpy as np
import psutil

from ...core.util import si_units
from .wrapping import workstep


def ping_mem(pid=None):
    if pid is None:
        pid = os.getpid()
    current_process = psutil.Process(pid)
    with current_process.oneshot():
        info = current_process.memory_full_info()
        uss = info.uss
        rss = info.rss

    return f"USS={si_units(uss)} RSS={si_units(rss)}"


@workstep(updates_context=True)
def memory_stress_test(n=37):

    logging.critical(f"ping_mem = {ping_mem()}")
    big = np.arange(int(2 ** float(n) / 8), dtype=np.float64)
    big *= 2.0
    time.sleep(1.0)
    logging.critical(f"ping_mem = {ping_mem()}")
    time.sleep(5.0)
    logging.critical(f"ping_mem = {ping_mem()}")
    logging.critical(f"bye")
    return {}
