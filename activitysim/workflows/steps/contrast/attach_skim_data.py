import logging
import pandas as pd
from pypyr.context import Context
from ..progression import reset_progress_step

logger = logging.getLogger(__name__)


def run_step(context: Context) -> None:

    contrast_data = context.get('contrast_data')
    skims = context.get('skims')
    skim_vars = context.get_formatted('skim_vars')
    tablename = context.get_formatted('tablename')
    otaz_col = context.get_formatted('otaz_col')
    dtaz_col = context.get_formatted('dtaz_col')
    time_col = context.get_formatted('time_col')

    if isinstance(skim_vars, str):
        skim_vars = [skim_vars]
    if len(skim_vars) == 1:
        skim_vars_note = skim_vars[0]
    else:
        skim_vars_note = f"{len(skim_vars)} skim vars"

    reset_progress_step(description=f"attach skim data / {tablename} <- {skim_vars_note}")

    contrast_data = attach_skim_data(
        contrast_data,
        skims,
        skim_vars,
        tablename,
        otaz_col,
        dtaz_col,
        time_col,
    )
    context['contrast_data'] = contrast_data




def attach_skim_data(
    tablesets,
    skims,
    skim_vars,
    tablename,
    otaz_col,
    dtaz_col,
    time_col,
):
    if not isinstance(skims, dict):
        skims = {i: skims for i in tablesets.keys()}

    for key, tableset in tablesets.items():
        skim_subset = skims[key][skim_vars]

        zone_ids = tableset['land_use'].index
        if zone_ids.is_monotonic_increasing and zone_ids[-1] == len(zone_ids) + zone_ids[0] - 1:
            offset = zone_ids[0]
            looks = [
                tableset[tablename][otaz_col].rename('otaz') - offset,
                tableset[tablename][dtaz_col].rename('dtaz') - offset,
            ]
        else:
            remapper = dict(zip(zone_ids, pd.RangeIndex(len(zone_ids))))
            looks = [
                tableset[tablename][otaz_col].rename('otaz').apply(remapper.get),
                tableset[tablename][dtaz_col].rename('dtaz').apply(remapper.get),
            ]
        if 'time_period' in skim_subset.dims:
            looks.append(
                tableset[tablename][time_col].apply(skims[key].attrs['time_period_imap'].get).rename('time_period'),
            )
        look = pd.concat(looks, axis=1)
        out = skim_subset.iat.df(look)
        tablesets[key][tablename] = tablesets[key][tablename].assign(**out)

    return tablesets