import pandas as pd
import yaml


def read_results(json_file):
    """
    Read benchmarking results from a single commit on a single machine.

    Parameters
    ----------
    json_file : str
        Path to the json file containing the target results.

    Returns
    -------
    pandas.DataFrame
    """
    out_data = {}
    with open(json_file, "rt") as f:
        in_data = yaml.safe_load(f)
    for k, v in in_data["results"].items():
        if v is None:
            continue
        m, c, _ = k.split(".")
        if m not in out_data:
            out_data[m] = {}
        out_data[m][c] = v["result"][0]
    return pd.DataFrame(out_data)
