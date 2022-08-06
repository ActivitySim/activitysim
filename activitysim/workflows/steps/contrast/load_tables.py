import os
from pathlib import Path

from pypyr.context import Context

from activitysim.standalone.compare import load_final_tables
from activitysim.standalone.utils import chdir

from ..error_handler import error_logging
from ..progression import reset_progress_step
from ..wrapping import workstep

#     databases = context.get_formatted('databases')
#     # the various different output directories to process, for example:
#     # {
#     #     "sharrow": "output-sharrow",
#     #     "legacy": "output-legacy",
#     # }
#
#     tables = context.get_formatted('tables')
#     # the various tables in the output directories to read, for example:
#     # trips:
#     #   filename: final_trips.csv
#     #   index_col: trip_id
#     # persons:
#     #   filename: final_persons.csv
#     #   index_col: person_id
#     # land_use:
#     #   filename: final_land_use.csv
#     #   index_col: zone_id


@workstep("tablesets")
def load_tables(databases, tables, common_output_directory=None) -> dict:
    """
    Load tables from one or more tablesets.

    Parameters
    ----------
    databases : Dict[str,Path-like]
        Defines one or more tablesets to load, out of input or output
        directories.  Each included database should include all the tables
        referenced by the `tables` argument.
    tables : Dict[str,Dict[str,str]]
        The keys give names of tables to load, and the values are dictionaries
        with keys at least including `filename`, and possibly also `index_col`.
    common_output_directory : Path-like, optional
        The directory in which each of the database directories can be found.
        If they are not in the same place, the user should set this to a common
        root directory, and include the full relative path for each database.
        If not given, defaults to the current working directory.

    Returns
    -------
    dict
        The loaded tables are under the key 'tablesets'.
    """

    tablefiles = {}
    index_cols = {}
    for t, v in tables.items():
        if isinstance(v, str):
            tablefiles[t] = v
        else:
            tablefiles[t] = v.get("filename")
            index_cols[t] = v.get("index_col", None)

    if common_output_directory is None:
        common_output_directory = os.getcwd()

    with chdir(common_output_directory):
        tablesets = load_final_tables(
            databases,
            tablefiles,
            index_cols,
        )

    return tablesets
