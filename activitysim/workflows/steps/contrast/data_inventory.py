import logging

import altair as alt
import pandas as pd
import xmle
from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

from ..error_handler import error_logging
from ..progression import reset_progress_step

logger = logging.getLogger(__name__)


@error_logging
def run_step(context: Context) -> None:
    reset_progress_step(description="report model inventory")

    context.assert_key_has_value(key="report", caller=__name__)
    report = context.get("report")
    fig = context.get("fig")
    tab = context.get("tab")

    tablesets = context.get("tablesets")
    skims = context.get("skims")
    try:
        title = context.get_formatted("title")
    except (KeyError, KeyNotInContextError):
        title = "Data Inventory"

    lens = {}
    dtypes = {}
    for source, tableset in tablesets.items():
        lens[source] = {}
        dtypes[source] = {}
        for tablename, tabledata in tableset.items():
            lens[source][tablename] = len(tabledata)
            for k, kt in tabledata.dtypes.items():
                dtypes[source][tablename, k] = kt

    with report:
        report << f"## {title}"

    with report:
        with pd.option_context(
            "display.max_rows", 1_000_000, "display.max_columns", 10_000
        ):
            report << tab("Table Row Count", level=3)
            report << pd.DataFrame(lens).applymap(
                lambda x: f"{x:,}" if isinstance(x, int) else x
            )

    with report:
        with pd.option_context(
            "display.max_rows", 1_000_000, "display.max_columns", 10_000
        ):
            report << tab("Table Contents", level=3)
            dtypes_table = (
                pd.DataFrame(dtypes).rename_axis(index=["table", "column"]).fillna("")
            )
            dtypes_table[""] = pd.Series(
                (dtypes_table.iloc[:, 0].to_frame().values == dtypes_table.values).all(
                    1
                ),
                index=dtypes_table.index,
            ).apply(lambda x: "" if x else "\u2B05")
            report << dtypes_table

    with report:
        report << tab("Skims Contents", level=3)
        ul = xmle.Elem("ul")
        for k in skims:
            i = xmle.Elem("li")
            i << xmle.Elem("b", text=k, tail=f" {skims[k].dtype} {skims[k].dims}")
            ul << i
        report << ul
