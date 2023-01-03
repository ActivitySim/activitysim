import multiprocessing
import platform
import subprocess

import pandas as pd
import psutil

from ..wrapping import workstep


def get_processor_info():
    out = ""
    if platform.system() == "Windows":
        out = platform.processor()
    elif platform.system() == "Darwin":
        out = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        ).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        out = subprocess.check_output(command, shell=True).strip()
    if isinstance(out, bytes):
        out = out.decode("utf-8")
    return out


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
    unit = "P"
    return f"{bytes:.2f}{unit}{suffix}"


@workstep
def machine_info(name=None):
    uname = platform.uname()
    ram = get_size(psutil.virtual_memory().total)
    out = {
        "Processor": get_processor_info(),
        "Number of Cores": multiprocessing.cpu_count(),
        "OS": uname.system,
        "Release": uname.release,
        "RAM": ram,
    }
    if name is True:
        name = uname.node
    if name is None:
        name = ""
    return pd.DataFrame(pd.Series(out).rename(name))
