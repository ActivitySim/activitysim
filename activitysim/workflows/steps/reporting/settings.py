import multiprocessing
import platform
import subprocess

import pandas as pd
import psutil

from ..wrapping import workstep


@workstep
def settings(names, **kwargs):
    if isinstance(names, str):
        names = [names]
    out = {}
    for name in names:
        out[name] = kwargs.get(name, None)
    return pd.DataFrame(pd.Series(out).rename(""))
