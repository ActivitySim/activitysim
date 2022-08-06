# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, inject

# FIXME
# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


@inject.injectable(cache=True)
def households_sample_size(settings, override_hh_ids):

    if override_hh_ids is None:
        return settings.get("households_sample_size", 0)
    else:
        return 0 if override_hh_ids is None else len(override_hh_ids)


@inject.injectable(cache=True)
def override_hh_ids(settings):

    hh_ids_filename = settings.get("hh_ids", None)
    if hh_ids_filename is None:
        return None

    file_path = config.data_file_path(hh_ids_filename, mandatory=False)
    if not file_path:
        logger.error(
            "hh_ids file name '%s' specified in settings not found" % hh_ids_filename
        )
        return None

    df = pd.read_csv(file_path, comment="#")

    if "household_id" not in df.columns:
        logger.error("No 'household_id' column in hh_ids file %s" % hh_ids_filename)
        return None

    household_ids = df.household_id.astype(int).unique()

    if len(household_ids) == 0:
        logger.error("No households in hh_ids file %s" % hh_ids_filename)
        return None

    logger.info(
        "Using hh_ids list with %s households from file %s"
        % (len(household_ids), hh_ids_filename)
    )

    return household_ids


@inject.injectable(cache=True)
def trace_hh_id(settings):

    id = settings.get("trace_hh_id", None)

    if id and not isinstance(id, int):
        logger.warning(
            "setting trace_hh_id is wrong type, should be an int, but was %s" % type(id)
        )
        id = None

    return id


@inject.injectable(cache=True)
def trace_od(settings):

    od = settings.get("trace_od", None)

    if od and not (
        isinstance(od, list) and len(od) == 2 and all(isinstance(x, int) for x in od)
    ):
        logger.warning("setting trace_od should be a list of length 2, but was %s" % od)
        od = None

    return od


@inject.injectable(cache=True)
def chunk_size(settings):
    _chunk_size = int(settings.get("chunk_size", 0) or 0)

    return _chunk_size


@inject.injectable(cache=True)
def check_for_variability(settings):
    return bool(settings.get("check_for_variability", False))
