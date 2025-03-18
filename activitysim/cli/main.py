from __future__ import annotations

import logging
import os
import sys


def prog():
    from activitysim import __doc__, __version__, workflows
    from activitysim.cli import CLI, benchmark, create, exercise, run

    asim = CLI(version=__version__, description=__doc__)
    asim.add_subcommand(
        name="run",
        args_func=run.add_run_args,
        exec_func=run.run,
        description=run.run.__doc__,
    )
    asim.add_subcommand(
        name="create",
        args_func=create.add_create_args,
        exec_func=create.create,
        description=create.create.__doc__,
    )
    asim.add_subcommand(
        name="benchmark",
        args_func=benchmark.make_asv_argparser,
        exec_func=benchmark.benchmark,
        description=benchmark.benchmark.__doc__,
    )
    asim.add_subcommand(
        name="workflow",
        args_func=lambda x: None,
        exec_func=workflows.main,
        description=workflows.main.__doc__,
    )
    asim.add_subcommand(
        name="test",
        args_func=exercise.add_exercise_args,
        exec_func=exercise.main,
        description=exercise.main.__doc__,
    )
    return asim


def main():
    # set all these before we import numpy or any other math library
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    asim = prog()
    try:
        if len(sys.argv) >= 2 and sys.argv[1] in ("workflow", "workflow_"):
            from activitysim import workflows

            if sys.argv[1] == "workflow_":
                result = workflows.main(sys.argv[2:])
                # exit silently on PipelineNotFoundError
                if result == 254:
                    result = 0
                sys.exit(result)
            else:
                sys.exit(workflows.main(sys.argv[2:]))
        else:
            sys.exit(asim.execute())
    except Exception as err:
        # if we are in the debugger, re-raise the error instead of exiting
        if sys.gettrace() is not None:
            raise
        logging.exception(err)
        sys.exit(99)


def parser():
    asim = prog()
    return asim.parser
