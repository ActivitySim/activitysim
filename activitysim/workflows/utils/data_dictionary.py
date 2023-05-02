from __future__ import annotations

import os.path
import warnings
from collections.abc import Mapping

import yaml


def check_data_dictionary(input):
    """
    Read and validate a data dictionary.

    The dictionary should be a nested mapping, with top level keys
    giving table names, second level keys giving column names, and
    then finally a mapping of codes to values.
    """
    if input is None:
        return {}
    elif isinstance(input, str):
        if not os.path.exists(input):
            warnings.warn(f"data dictionary file {input} is missing", stacklevel=2)
            return {}
        with open(input) as f:
            content = yaml.safe_load(f)
    else:
        content = input

    for i, v in content.items():
        assert isinstance(i, str)
        assert isinstance(v, Mapping)
        for c, j in v.items():
            assert isinstance(c, str)
            assert isinstance(j, Mapping)

    return content
