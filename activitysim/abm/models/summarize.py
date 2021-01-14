# ActivitySim
# See full license in LICENSE.txt.
import logging
import sys
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

from activitysim.core.config import setting

logger = logging.getLogger(__name__)


@inject.step()
def write_summaries(output_dir):

    summary_settings_name = 'output_summaries'
    summary_file_name = 'summaries.txt'

    summary_settings = setting(summary_settings_name)

    if summary_settings is None:
        logger.info("No {summary_settings_name} specified in settings file. Nothing to write.")
        return

    summary_dict = summary_settings

    mode = 'wb' if sys.version_info < (3,) else 'w'
    with open(config.output_file_path(summary_file_name), mode) as output_file:

        for table_name, column_names in summary_dict.items():

            df = pipeline.get_table(table_name)

            for c in column_names:
                n = 100
                empty = (df[c] == '') | df[c].isnull()

                print(f"\n### {table_name}.{c} type: {df.dtypes[c]} rows: {len(df)} ({empty.sum()} empty)\n\n",
                      file=output_file)
                print(df[c].value_counts().nlargest(n), file=output_file)
