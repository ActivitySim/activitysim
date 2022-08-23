# This file allows running ActivitySim as a Python module from the command line.
# For example:
#
#   python -m activitysim run ...
#
# This style of calling ActivitySim permits developers to more easily ensure
# they are calling the correct version that has been installed with the
# particular Python executable invoked, especially for debugging.  It also can
# be configured to invoke differ configurations automatically (like engaging the
# threadstopper options to prevent multithread thrashing). It is probably not
# needed by typical users with only one installed version.

import os
import sys


def main():
    # clean up message formatting
    if sys.argv and sys.argv[0].endswith("__main__.py"):
        sys.argv[0] = "activitysim"

    # threadstopper
    if "--fast" not in sys.argv:
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from .cli.main import main

    main()


if __name__ == "__main__":
    main()
