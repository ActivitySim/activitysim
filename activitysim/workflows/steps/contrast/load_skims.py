import os
from pypyr.context import Context
from ..progression import reset_progress_step
from ..error_handler import error_logging
from ..wrapping import workstep
from pathlib import Path
from activitysim.standalone.utils import chdir
import logging
import yaml
import openmatrix
import sharrow as sh
import numpy as np
import glob

logger = logging.getLogger(__name__)


def load_skims_per_settings(
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


@workstep("skims")
def load_skims(
    config_dirs=("configs",),
    data_dir="data",
    working_directory=None,
    common_directory=None,
):
    """
    Open and prepare one or more sets of skims for use.

    The xarray library can use dask to delay loading skim matrices until they
    are actually needed, so this may not actually load the skims into RAM.

    Parameters
    ----------
    config_dirs : Tuple[Path-like] or Mapping[str, Tuple[Path-like]], default ('configs',)
        Location of the config directory(s).  The skims are loaded by finding
        the `network_los.yaml` definition files in one of these directories.
        If a mapping is provided, the keys label multiple different sets of skims
        to load, which can be useful in contrasting model results from different
        input values.
    data_dir : Path-like or Mapping[str, Path-like], default 'data'
        Loc
    working_directory : Path-like, optional
    common_directory : Path-like, optional
        Deprecated, use working_directory

    Returns
    -------
    xarray.Dataset or Dict[str, xarray.Dataset]
        A single dataset is returned when a single set of config and data
        directories are provided, otherwise a dict is returned with keys
        that match the inputs.
    """
    if common_directory is not None:
        working_directory = common_directory

    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]

    if working_directory is None:
        working_directory = os.getcwd()

    with chdir(working_directory):
        network_los_file = None
        for config_dir in config_dirs:
            network_los_file = os.path.join(config_dir, "network_los.yaml")
            if os.path.exists(network_los_file):
                break
        if network_los_file is None:
            raise FileNotFoundError("<<config_dir>>/network_los.yaml")
        if isinstance(data_dir, (str, Path)) and isinstance(
            network_los_file, (str, Path)
        ):
            skims = load_skims_per_settings(network_los_file, data_dir)
        else:
            skims = {}
            for k in data_dir.keys():
                skims[k] = load_skims_per_settings(network_los_file, data_dir[k])
                # TODO: allow for different network_los_file

    return skims
