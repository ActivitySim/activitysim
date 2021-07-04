import sys

from activitysim.cli import CLI
from activitysim.cli import run
from activitysim.cli import create
from activitysim.cli import benchmark

from activitysim import __version__, __doc__


def main():
    asim = CLI(version=__version__,
               description=__doc__)
    asim.add_subcommand(name='run',
                        args_func=run.add_run_args,
                        exec_func=run.run,
                        description=run.run.__doc__)
    asim.add_subcommand(name='create',
                        args_func=create.add_create_args,
                        exec_func=create.create,
                        description=create.create.__doc__)
    asim.add_subcommand(name='benchmark',
                        args_func=benchmark.make_asv_argparser,
                        exec_func=benchmark.benchmark,
                        description=benchmark.benchmark.__doc__)
    sys.exit(asim.execute())
