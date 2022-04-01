import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from ....standalone.utils import chdir

logger = logging.getLogger(__name__)

def run_step(context: Context) -> None:

    context.assert_key_has_value(key='common_output_directory', caller=__name__)
    common_output_directory = context.get_formatted('common_output_directory')

    context.assert_key_has_value(key='report', caller=__name__)
    report = context.get('report')
    fig = context.get('fig')
    tag = context.get('tag')
    timing_log = f"combined_timing_log-{tag}.csv"

    with report:
        with chdir(common_output_directory):

            report << fig("Model Runtime")

            logger.info(f"building runtime report from {common_output_directory}/{timing_log}")

            df = pd.read_csv(timing_log, index_col='model_name')
            df1 = df[['sharrow', 'legacy']].rename_axis(columns="source").unstack().rename('seconds').reset_index()
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

            report << result
