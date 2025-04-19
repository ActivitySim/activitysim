from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING

import pandas as pd

from .util import si_units

if TYPE_CHECKING:
    from .workflow import State


class NoTiming:
    """Class that does no timing, serves as the default.

    This class is kept as simple as possible to avoid unnecessary overhead when
    no timing is requested.
    """

    @contextmanager
    def time_expression(self, expression: str):
        """Context manager to time an expression."""
        yield

    def write_log(self, state: State) -> None:
        """Write the log to a file."""
        pass


class EvalTiming(NoTiming):
    def __new__(cls, log: Path | None = None, **kwargs):
        if log is None:
            return NoTiming()
        else:
            return super().__new__(cls)

    def __init__(self, log: Path | None = None, *, overwrite: bool = False):
        """
        Timing class to log the time taken to evaluate expressions.

        Parameters
        ----------
        log : Path | None, default None
            Path to the log file. If None, no logging is done. If this is an
            absolute path, the log file is created there. If this is just a
            simple filename or a relative path, the log file is created relative
            in or relative to the usual logging directory.
        overwrite : bool, default False
            If True, overwrite the log file if it already exists. If False,
            create a new log file with a unique name.
        """
        self.log_file = log
        self.overwrite = overwrite
        self.elapsed_times = {}

    @contextmanager
    def time_expression(self, expression: str):
        """Context manager to time an expression.

        Parameters
        ----------
        expression : str
            The expression to be timed. This is used as the key in the log file.
        """

        # when performance logging is not enabled, do nothing
        if self.log_file is None:
            yield
            return

        # when performance logging is enabled, we track the time it takes to evaluate
        # the expression and store it
        start_time = time_ns()
        yield
        end_time = time_ns()
        elapsed_time = end_time - start_time
        if expression in self.elapsed_times:
            self.elapsed_times[expression] += elapsed_time
        else:
            self.elapsed_times[expression] = elapsed_time

    def write_log(self, state: State) -> None:
        """Write the log to a file.

        Parameters
        ----------
        state : State
            The state object containing configuration information. This is used
            to determine the path for the log file, when the log file is not
            given as an absolute path.
        """
        if self.log_file is None:
            return

        if self.log_file.is_absolute():
            filename = self.log_file
        else:
            filename = state.get_log_file_path(
                str(Path("expr-performance") / self.log_file)
            )

        # if the log file already exists and overwrite is false, create a new file
        proposed_filename = filename
        n = 0
        while not self.overwrite and proposed_filename.exists():
            n += 1
            proposed_filename = filename.with_stem(filename.stem + f"-{n}")
        filename = proposed_filename

        # ensure the parent directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Determine the scale for the elapsed times.  We want to use an appropriate
        # timescale for the elapsed times, which provides useful information without
        # reporting excessive precision.
        # If the smallest elapsed time is greater than 1 second, use seconds.
        # If the smallest elapsed time is greater than 1 millisecond, use milliseconds.
        # Otherwise, use microseconds, no one should care about nanoseconds.
        min_t = 1_000_000_000
        for t in self.elapsed_times.values():
            if t < min_t:
                min_t = t
        if min_t > 1_000_000_000:
            scale = 1_000_000_000
            label = "Time (sec)"
        elif min_t > 1_000_000:
            scale = 1_000_000
            label = "Time (msec)"
        else:
            scale = 1_000
            label = "Time (µsec)"

        # The timing log is written in a tab-separated format, with times in the
        # first column so they are easy to scan through for anomalies.
        with open(filename, "w") as f:
            f.write(f"{label:11}\tExpression\n")
            for expression, elapsed_time in self.elapsed_times.items():
                t = int(elapsed_time / scale)
                f.write(f"{t: 11d}\t{expression}\n")


class AnalyzeEvalTiming:
    """
    Class to analyze the timing of expressions.
    """

    def __init__(self, state: State):
        self.log_dir = state.get_log_file_path(str(Path("expr-performance")))
        raw_data = {}
        for f in self.log_dir.glob("*.log"):
            df = pd.read_csv(f, sep="\t")
            if "(msec)" in df.columns[0]:
                df.columns = ["Time (µsec)"] + df.columns[1:].tolist()
                df.iloc[:, 0] = df.iloc[:, 0].astype(int) * 1_000
            elif "(sec)" in df.columns[0]:
                df.columns = ["Time (µsec)"] + df.columns[1:].tolist()
                df.iloc[:, 0] = df.iloc[:, 0].astype(int) * 1_000_000
            else:
                df.iloc[:, 0] = df.iloc[:, 0].astype(int)
            raw_data[str(f.stem)] = df
        d = pd.concat(raw_data, names=["Component"]).reset_index()
        self.data = d[["Time (µsec)", "Component", "Expression"]]
        self.data.sort_values(by=["Time (µsec)"], ascending=[False], inplace=True)

    def to_html(
        self, filename: str | Path = "expression-timing.html", cutoff_secs=0.1
    ) -> None:
        """Write the data to an HTML file.

        Parameters
        ----------
        filename : str | Path
            The name of the file to write the HTML to. If a relative path is given,
            it will be written in the log directory.
        cutoff_secs : float
            The cutoff time in seconds. Only expressions with a runtime greater than
            this will be included in the HTML file. This is used to avoid writing a
            huge report full of expressions that run plenty fast.
        """
        self.data[self.data["Time (µsec)"] >= cutoff_secs * 1e6].to_html(
            self.log_dir.joinpath(filename), index=False
        )
