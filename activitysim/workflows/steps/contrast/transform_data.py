import logging
import pandas as pd
from pypyr.context import Context
from ..progression import reset_progress_step
from ..wrapping import report_step

logger = logging.getLogger(__name__)



@report_step
def transform_data(
    tablesets,
    tablename,
    column,
    out,
    qcut=None,
    cut=None,
    clip=None,
    censor=None,
) -> dict:

    if qcut is None and cut is None and clip is None and censor is None:
        raise ValueError("must give at least one of {cut, qcut, clip, censor}")

    reset_progress_step(description=f"attach transformed data / {tablename} <- {out}")

    # collect all series into a common vector, so bins are common
    pieces = {}
    for key, tableset in tablesets.items():
        pieces[key] = tableset[tablename][column]

    common = pd.concat(pieces, names=['source'])

    if clip is not None:
        common = common.clip(**clip)

    if censor is not None:
        common = common.where(common.between(**censor))

    if cut is not None:
        common = pd.cut(common, **cut)
    elif qcut is not None:
        common = pd.qcut(common, **qcut)

    pieces = {k: common.loc[k] for k in tablesets.keys()}

    for key in tablesets:
        tablesets[key][tablename] = tablesets[key][tablename].assign(**{out: pieces[key]})

    return dict(tablesets=tablesets)