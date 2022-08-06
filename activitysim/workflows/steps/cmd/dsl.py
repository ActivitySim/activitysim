"""pypyr step yaml definition for commands - domain specific language."""
import logging
import os
import shlex
import subprocess
import sys
import time

import pypyr.errors
from pypyr.config import config
from pypyr.errors import ContextError
from pypyr.utils import types

from ..progression import reset_progress_step

# logger means the log level will be set correctly
logger = logging.getLogger(__name__)

RED = "\033[91m"
END = "\033[0m"


def stream_process(process, label):
    go = True
    while go:
        go = process.poll() is None
        for line in process.stdout:
            print(line.decode().rstrip())
        for line in process.stderr:
            print(RED + line.decode().rstrip() + END, file=sys.stderr)


class CmdStep:
    """A pypyr step that represents a command runner step.

    This models a step that takes config like this:
        cmd: <<cmd string>>

        OR, as a dict
        cmd:
            run: str. mandatory. command + args to execute.
            save: bool. defaults False. save output to cmdOut.

    If save is True, will save the output to context as follows:
        cmdOut:
            returncode: 0
            stdout: 'stdout str here. None if empty.'
            stderr: 'stderr str here. None if empty.'

    cmdOut.returncode is the exit status of the called process. Typically 0
    means OK. A negative value -N indicates that the child was terminated by
    signal N (POSIX only).

    The run_step method does the actual work. init loads the yaml.
    """

    def __init__(self, name, context):
        """Initialize the CmdStep.

        The step config in the context dict looks like this:
            cmd: <<cmd string>>

            OR, as a dict
            cmd:
                run: str. mandatory. command + args to execute.
                save: bool. optional. defaults False. save output to cmdOut.
                cwd: str/path. optional. if specified, change the working
                     directory just for the duration of the command.

        Args:
            name: Unique name for step. Likely __name__ of calling step.
            context: pypyr.context.Context. Look for config in this context
                     instance.

        """
        assert name, "name parameter must exist for CmdStep."
        assert context, "context param must exist for CmdStep."
        # this way, logs output as the calling step, which makes more sense
        # to end-user than a mystery steps.dsl.blah logging output.
        self.logger = logging.getLogger(name)

        context.assert_key_has_value(key="cmd", caller=name)

        self.context = context

        cmd_config = context.get_formatted("cmd")

        if isinstance(cmd_config, str):
            self.cmd_text = cmd_config
            self.cwd = None
            self.logger.debug("Processing command string: %s", cmd_config)
            self.label = cmd_config
        elif isinstance(cmd_config, dict):
            context.assert_child_key_has_value(parent="cmd", child="run", caller=name)

            self.cmd_text = cmd_config["run"]
            self.label = cmd_config.get("label", self.cmd_text)
            self.conda_path = cmd_config.get("conda_path", None)

            cwd_string = cmd_config.get("cwd", None)
            if cwd_string:
                self.cwd = cwd_string
                self.logger.debug(
                    "Processing command string in dir " "%s: %s",
                    self.cwd,
                    self.cmd_text,
                )
            else:
                self.cwd = None
                self.logger.debug("Processing command string: %s", self.cmd_text)

        else:
            raise ContextError(
                f"{name} cmd config should be either a simple "
                "string cmd='mycommandhere' or a dictionary "
                "cmd={'run': 'mycommandhere', 'save': False}."
            )

    def run_step(self, is_shell):
        """Run a command.

        Runs a program or executable. If is_shell is True, executes the command
        through the shell.

        Args:
            is_shell: bool. defaults False. Set to true to execute cmd through
                      the default shell.
        """
        assert is_shell is not None, "is_shell param must exist for CmdStep."
        conda_path = self.conda_path or sys.exec_prefix

        # why? If shell is True, it is recommended to pass args as a string
        # rather than as a sequence.
        # But not on windows, because windows wants strings, not sequences.
        if is_shell or config.is_windows:
            args = f'conda run -p "{conda_path}" ' + self.cmd_text
        else:
            args = shlex.split(self.cmd_text)
            args = ["conda", "run", "-p", conda_path] + list(args)  # TODO windows?

        reset_progress_step(description=f"{self.label}", prefix="[bold green]")

        env = os.environ.copy()
        pythonpath = env.pop("PYTHONPATH", None)

        process = subprocess.Popen(
            args,
            shell=is_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=env,
        )
        stream_process(process, self.label)
        self.context["cmdOut"] = {
            "returncode": process.returncode,
        }

        # don't swallow the error, because it's the Step swallow decorator
        # responsibility to decide to ignore or not.
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                process.stdout,
                process.stderr,
            )
