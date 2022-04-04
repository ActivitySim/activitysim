import os
from pypyr.context import Context
from ..progression import reset_progress_step
from ..error_handler import error_logging
from pathlib import Path
from activitysim.standalone.compare import load_final_tables
from activitysim.standalone.utils import chdir

@error_logging
def run_step(context: Context) -> None:

    reset_progress_step(description="load tables")

    context.assert_key_has_value(key='common_output_directory', caller=__name__)
    common_output_directory = context.get_formatted('common_output_directory')

    databases = context.get_formatted('databases')
    # the various different output directories to process, for example:
    # {
    #     "sharrow": "output-sharrow",
    #     "legacy": "output-legacy",
    # }

    tables = context.get_formatted('tables')
    # the various tables in the output directories to read, for example:
    # trips:
    #   filename: final_trips.csv
    #   index_col: trip_id
    # persons:
    #   filename: final_persons.csv
    #   index_col: person_id
    # land_use:
    #   filename: final_land_use.csv
    #   index_col: zone_id

    tablefiles = {}
    index_cols = {}
    for t, v in tables.items():
        if isinstance(v, str):
            tablefiles[t] = v
        else:
            tablefiles[t] = v.get("filename")
            index_cols[t] = v.get("index_col", None)

    with chdir(common_output_directory):
        contrast_data = load_final_tables(
            databases,
            tablefiles,
            index_cols,
        )

    context['contrast_data'] = contrast_data
    