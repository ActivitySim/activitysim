# ActivitySim
# See full license in LICENSE.txt.

import argparse
import os
import sys

import extensions

from activitysim.cli.run import add_run_args, run

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    os.environ["MKL_NUM_THREADS"] = "1"

    sys.exit(run(args))
