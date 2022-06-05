import glob
import logging
import os

import numpy as np
import openmatrix
import sharrow as sh
import yaml

logger = logging.getLogger(__name__)


def load_skims(
    network_los_settings_filename,
    data_dir,
):
    with open(network_los_settings_filename, "rt") as f:
        settings = yaml.safe_load(f)

    skim_settings = settings["taz_skims"]
    if isinstance(skim_settings, str):
        skims_omx_fileglob = skim_settings
    else:
        skims_omx_fileglob = skim_settings.get("omx", None)
        skims_omx_fileglob = skim_settings.get("files", skims_omx_fileglob)
    skims_filenames = glob.glob(os.path.join(data_dir, skims_omx_fileglob))
    index_names = ("otaz", "dtaz", "time_period")
    indexes = None
    time_period_breaks = settings.get("skim_time_periods", {}).get("periods")
    time_periods = settings.get("skim_time_periods", {}).get("labels")
    time_period_sep = "__"

    time_window = settings.get("skim_time_periods", {}).get("time_window")
    period_minutes = settings.get("skim_time_periods", {}).get("period_minutes")
    n_periods = int(time_window / period_minutes)

    tp_map = {}
    tp_imap = {}
    label = time_periods[0]
    i = 0
    for t in range(n_periods):
        if t in time_period_breaks:
            i = time_period_breaks.index(t)
            label = time_periods[i]
        tp_map[t + 1] = label
        tp_imap[t + 1] = i

    omxs = [
        openmatrix.open_file(skims_filename, mode="r")
        for skims_filename in skims_filenames
    ]
    if isinstance(time_periods, (list, tuple)):
        time_periods = np.asarray(time_periods)
    result = sh.dataset.from_omx_3d(
        omxs,
        index_names=index_names,
        indexes=indexes,
        time_periods=time_periods,
        time_period_sep=time_period_sep,
    )
    result.attrs["time_period_map"] = tp_map
    result.attrs["time_period_imap"] = tp_imap

    return result
