import logging

import numpy as np
import pandas as pd
from pypyr.context import Context

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep(updates_context=True)
def transform_data(
    tablesets,
    tablename,
    column,
    out,
    qcut=None,
    cut=None,
    clip=None,
    censor=None,
    eval=None,
) -> dict:

    if qcut is None and cut is None and clip is None and censor is None:
        raise ValueError("must give at least one of {cut, qcut, clip, censor}")

    reset_progress_step(description=f"attach transformed data / {tablename} <- {out}")

    if eval is not None:
        for key, tableset in tablesets.items():
            for target, expr in eval.items():
                tableset[tablename][target] = tableset[tablename].eval(expr)
        return dict(tablesets=tablesets)

    # collect all series into a common vector, so bins are common
    pieces = {}
    for key, tableset in tablesets.items():
        pieces[key] = tableset[tablename][column]

    common = pd.concat(pieces, names=["source"])

    if clip is not None:
        common = common.clip(**clip)

    if censor is not None:
        common = common.where(common.between(**censor))

    use_midpoint = False
    if cut is not None:
        if cut.get("labels", None) == "midpoint":
            use_midpoint = True
            cut.pop("labels")
        common = pd.cut(common, **cut)
    elif qcut is not None:
        if qcut.get("labels", None) == "midpoint":
            use_midpoint = True
            qcut.pop("labels")
        common = pd.qcut(common, **qcut)

    if use_midpoint:
        common = common.apply(lambda x: x.mid)

    pieces = {k: common.loc[k] for k in tablesets.keys()}

    for key in tablesets:
        tablesets[key][tablename] = tablesets[key][tablename].assign(
            **{out: pieces[key]}
        )

    return dict(tablesets=tablesets)
