import logging
import pandas as pd
import altair as alt
from ....standalone.utils import chdir
from ..progression import reset_progress_step
from ..wrapping import report_step

logger = logging.getLogger(__name__)


@report_step
def contrast_runtime(
        combined_timing_log,
        include_runs=('sharrow', 'legacy'),
) -> None:
    reset_progress_step(description="report model runtime")

    include_runs = list(include_runs)

    logger.info(f"building runtime report from {combined_timing_log}")

    df = pd.read_csv(combined_timing_log, index_col='model_name')
    df1 = df[include_runs].rename_axis(columns="source").unstack().rename('seconds').reset_index()
    c = alt.Chart(df1, height={"step": 20}, )

    result = c.mark_bar(
        yOffset=-3,
        size=6,
    ).transform_filter(
        (alt.datum.source == 'legacy')
    ).encode(
        x=alt.X('seconds:Q', stack=None),
        y=alt.Y('model_name', type='nominal', sort=None),
        color="source",
        tooltip=['source', 'model_name', 'seconds']
    ) + c.mark_bar(
        yOffset=4,
        size=6,
    ).transform_filter(
        (alt.datum.source == 'sharrow')
    ).encode(
        x=alt.X('seconds:Q', stack=None),
        y=alt.Y('model_name', type='nominal', sort=None),
        color="source",
        tooltip=['source', 'model_name', 'seconds']
    ) | alt.Chart(df1).mark_bar().encode(
        color='source',
        x='source',
        y='sum(seconds)',
        tooltip=['source', 'sum(seconds)']
    )

    return result
