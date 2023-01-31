import logging

import numpy as np
import pandas as pd

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


def _as_int(x):
    if x.dtype.kind == "i":
        return x
    else:
        return x.astype(np.int32)


@workstep(updates_context=True)
def attach_skim_data(
    tablesets,
    skims,
    skim_vars,
    tablename,
    otaz_col,
    dtaz_col,
    time_col=None,
    rename_skim_vars=None,
) -> dict:
    if isinstance(skim_vars, str):
        skim_vars = [skim_vars]
    if len(skim_vars) == 1:
        skim_vars_note = skim_vars[0]
    else:
        skim_vars_note = f"{len(skim_vars)} skim vars"

    reset_progress_step(
        description=f"attach skim data / {tablename} <- {skim_vars_note}"
    )

    if not isinstance(skims, dict):
        skims = {i: skims for i in tablesets.keys()}

    for key, tableset in tablesets.items():
        # skim_subset = skims[key][list(skims[key].coords) + skim_vars]
        skim_subset = skims[key][skim_vars]

        otag = "omaz" if "omaz" in skims[key].coords else "otaz"
        dtag = "dmaz" if "dmaz" in skims[key].coords else "dtaz"

        zone_ids = tableset["land_use"].index
        if (
            zone_ids.is_monotonic_increasing
            and zone_ids[-1] == len(zone_ids) + zone_ids[0] - 1
        ):
            offset = zone_ids[0]
            looks = [
                _as_int(tableset[tablename][otaz_col].rename(otag) - offset),
                _as_int(tableset[tablename][dtaz_col].rename(dtag) - offset),
            ]
        else:
            remapper = dict(zip(zone_ids, pd.RangeIndex(len(zone_ids))))
            looks = [
                _as_int(tableset[tablename][otaz_col].rename(otag).apply(remapper.get)),
                _as_int(tableset[tablename][dtaz_col].rename(dtag).apply(remapper.get)),
            ]
        if "time_period" in skim_subset.dims:
            if time_col is None:
                raise KeyError("time_period in skims to slice but time_col is missing")
            looks.append(
                tableset[tablename][time_col]
                .apply(skims[key].attrs["time_period_imap"].get)
                .rename("time_period"),
            )
        look = pd.concat(looks, axis=1)
        try:
            out = skim_subset.iat.df(look)
        except KeyError as err:
            # KeyError is triggered when reading TAZ data from MAZ-enabled skims
            lookr = {i: look[i].values for i in look.columns}
            out = skims[key].iat(**lookr, _names=skim_vars).to_dataframe()
            out = out.set_index(tablesets[key][tablename].index)
        if rename_skim_vars is not None:
            if isinstance(rename_skim_vars, str):
                rename_skim_vars = [rename_skim_vars]
            out = out.rename(columns=dict(zip(skim_vars, rename_skim_vars)))
        tablesets[key][tablename] = tablesets[key][tablename].assign(**out)

    return dict(tablesets=tablesets)
