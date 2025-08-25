# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import argparse
import sys

import extensions

from activitysim.cli.run import add_run_args, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    sys.exit(run(args))
