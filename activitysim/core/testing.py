# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import traceback
from pathlib import Path

import pytest


def run_if_exists(filename):
    stack = traceback.extract_stack()
    base_dir = Path(stack[-2].filename).parent
    target_file = base_dir.joinpath(filename)
    return pytest.mark.skipif(
        not target_file.exists(), reason=f"required file {filename} is missing"
    )
