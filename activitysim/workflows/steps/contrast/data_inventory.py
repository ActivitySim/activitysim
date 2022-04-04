
import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from pypyr.errors import KeyNotInContextError
from ..error_handler import error_logging
from ..progression import reset_progress_step

logger = logging.getLogger(__name__)


@error_logging
def run_step(context: Context) -> None:
    reset_progress_step(description="report model inventory")

    context.assert_key_has_value(key='report', caller=__name__)
    report = context.get('report')
    fig = context.get('fig')
    tab = context.get('tab')

    contrast_data = context.get('contrast_data')
    try:
        title = context.get_formatted('title')
    except (KeyError, KeyNotInContextError):
        title = "Data Inventory"

    lens = {}
    dtypes = {}
    for source, tableset in contrast_data.items():
        lens[source] = {}
        dtypes[source] = {}
        for tablename, tabledata in tableset.items():
            lens[source][tablename] = len(tabledata)
            for k, kt in tabledata.dtypes.items():
                dtypes[source][tablename, k] = kt

    with report:
        report << f"## {title}"

    with report:
        with pd.option_context('display.max_rows', 1_000_000, 'display.max_columns', 10_000):
            report << tab("Table Row Count", level=3)
            report << pd.DataFrame(lens).applymap(lambda x: f"{x:,}" if isinstance(x, int) else x)

    with report:
        with pd.option_context('display.max_rows', 1_000_000, 'display.max_columns', 10_000):
            report << tab("Table Contents", level=3)
            report << pd.DataFrame(dtypes).rename_axis(index=['table','column']).fillna("")


