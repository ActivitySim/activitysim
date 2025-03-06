# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import workflow

# FIXME
# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


@workflow.cached_object
def households_sample_size(state: workflow.State, override_hh_ids) -> int:
    if override_hh_ids is None:
        return state.settings.households_sample_size
    else:
        return len(override_hh_ids)


@workflow.cached_object
def override_hh_ids(state: workflow.State) -> np.ndarray | None:
    hh_ids_filename = state.settings.hh_ids
    if hh_ids_filename is None:
        return None

    file_path = state.filesystem.get_data_file_path(hh_ids_filename, mandatory=False)
    if not file_path:
        file_path = state.filesystem.get_config_file_path(
            hh_ids_filename, mandatory=False
        )
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


@workflow.cached_object
def trace_od(state: workflow.State) -> tuple[int, int] | None:
    od = state.settings.trace_od

    if od and not (
        isinstance(od, list | tuple)
        and len(od) == 2
        and all(isinstance(x, int) for x in od)
    ):
        logger.warning(
            "setting trace_od should be a list or tuple of length 2, but was %s" % od
        )
        od = None

    return od


@workflow.cached_object
def chunk_size(state: workflow.State) -> int:
    _chunk_size = int(state.settings.chunk_size or 0)

    return _chunk_size


@workflow.cached_object
def check_for_variability(state: workflow.State) -> bool:
    return bool(state.settings.check_for_variability)
