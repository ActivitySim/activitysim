from __future__ import annotations

import os
import warnings

import pandas as pd

from activitysim.workflows.steps.wrapping import workstep
from activitysim.workflows.utils import chdir


def load_final_tables(output_dirs, tables=None, index_cols=None):
    result = {}
    for key, pth in output_dirs.items():
        if not os.path.exists(pth):
            warnings.warn(f"{key} directory does not exist: {pth}")
            continue
        result[key] = {}
        for tname, tfile in tables.items():
            tpath = os.path.join(pth, tfile)
            kwargs = {}
            if index_cols is not None and tname in index_cols:
                kwargs["index_col"] = index_cols[tname]
            if os.path.exists(tpath):
                result[key][tname] = pd.read_csv(tpath, **kwargs)
        if len(result[key]) == 0:
            # no tables were loaded, delete the entire group
            del result[key]
    return result


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
