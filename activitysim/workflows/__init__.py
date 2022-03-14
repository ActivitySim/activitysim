"""Naive custom loader without any error handling."""
import os
import signal
import sys
import traceback
from pathlib import Path

import pypyr.log.logger
import pypyr.pipelinerunner
import pypyr.version
import pypyr.yaml
from pypyr.config import config


def get_pipeline_definition(pipeline_name, parent):
    """Simplified loader that gets pipeline_name.yaml in working dir."""
    search_dir = Path(os.path.dirname(__file__))
    workflow_file = search_dir.joinpath(f"{pipeline_name}.yaml")
    if os.path.exists(workflow_file):
        with open(workflow_file) as yaml_file:
            return pypyr.yaml.get_pipeline_yaml(yaml_file)
    else:
        from pypyr.loaders.file import get_pipeline_definition

        return get_pipeline_definition(pipeline_name, parent)


def main(args):
    """
    Run a named workflow.

    Each workflow defines its own arguments, refer to the workflow itself to
    learn what the arguments and options are.
    """
    cwd = Path.cwd()
    from pypyr.cli import get_args

    if args is None:
        args = sys.argv[2:]

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

        return 255
