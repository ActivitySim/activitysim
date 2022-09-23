"""Naive custom loader without any error handling."""
import os
import signal
import sys
import traceback
from pathlib import Path

from .progression import get_progress


def get_pipeline_definition(pipeline_name, parent):
    """Simplified loader that gets pipeline_name.yaml in working dir."""
    import pypyr.yaml
    from pypyr.loaders.file import get_pipeline_definition

    search_dir = Path(os.path.dirname(__file__)).parent
    workflow_file = search_dir.joinpath(f"{pipeline_name}.yaml")
    if os.path.exists(workflow_file):
        with open(workflow_file) as yaml_file:
            return pypyr.yaml.get_pipeline_yaml(yaml_file)
    else:

        return get_pipeline_definition(pipeline_name, parent)


def enable_vt_support():
    # allow printing in color on windows terminal
    if os.name == "nt":
        import ctypes

        hOut = ctypes.windll.kernel32.GetStdHandle(-11)
        out_modes = ctypes.c_uint32()
        ENABLE_VT_PROCESSING = ctypes.c_uint32(0x0004)
        ctypes.windll.kernel32.GetConsoleMode(hOut, ctypes.byref(out_modes))
        out_modes = ctypes.c_uint32(out_modes.value | 0x0004)
        ctypes.windll.kernel32.SetConsoleMode(hOut, out_modes)


def main(args):
    """
    Run a named workflow.

    Each workflow defines its own arguments, refer to the workflow itself to
    learn what the arguments and options are.
    """
    if args is None:
        args = sys.argv[2:]

    if "--no-rich" in args:
        args.remove("--no-rich")
        os.environ["NO_RICH"] = "1"

    with get_progress():

        try:
            import pypyr.log.logger
            import pypyr.pipelinerunner
            import pypyr.yaml
            from pypyr.cli import get_args
            from pypyr.config import config
        except ImportError:
            raise ImportError("activitysim.workflows requires pypyr")

        parsed_args = get_args(args)

        try:
            config.init()
            pypyr.log.logger.set_root_logger(
                log_level=parsed_args.log_level, log_path=parsed_args.log_path
            )

            pypyr.pipelinerunner.run(
                pipeline_name=parsed_args.pipeline_name,
                args_in=parsed_args.context_args,
                parse_args=True,
                groups=parsed_args.groups,
                success_group=parsed_args.success_group,
                failure_group=parsed_args.failure_group,
                py_dir=parsed_args.py_dir,
                loader="activitysim.workflows",
            )

        except KeyboardInterrupt:
            # Shell standard is 128 + signum = 130 (SIGINT = 2)
            sys.stdout.write("\n")
            return 128 + signal.SIGINT
        except Exception as e:
            # stderr and exit code 255
            sys.stderr.write("\n")
            sys.stderr.write(f"\033[91m{type(e).__name__}: {str(e)}\033[0;0m")
            sys.stderr.write("\n")
            # at this point, you're guaranteed to have args and thus log_level
            if parsed_args.log_level:
                if parsed_args.log_level < 10:
                    # traceback prints to stderr by default
                    traceback.print_exc()

            # if we are in the debugger, re-raise the error instead of returning
            if sys.gettrace() is not None:
                raise
            from pypyr.errors import PipelineNotFoundError

            return 254 if isinstance(e, PipelineNotFoundError) else 255
