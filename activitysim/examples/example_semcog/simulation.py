# ActivitySim
# See full license in LICENSE.txt.

import sys
import argparse

from activitysim.cli.run import add_run_args, run

import extensions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    sys.exit(run(args))
