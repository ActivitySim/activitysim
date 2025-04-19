from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Literal

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


def write_sortable_table(df: pd.DataFrame, filename: str | Path) -> None:
    html_table = df.to_html(classes="sortable", index=False)
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tablesort/5.3.1/tablesort.min.css">
        </head>
        <body>
        {html_table}
        <script src="https://cdnjs.cloudflare.com/ajax/libs/tablesort/5.3.1/tablesort.min.js"></script>
        </body>
        </html>
        """
    with open(filename, "w") as f:
        f.write(html_content)


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

        # break trace labels into components and subcomponents
        d["Subcomponent"] = d["Component"].str.split(".", 1).str[1]
        d["Component"] = d["Component"].str.split(".", 1).str[0]
        self.data = d[["Time (µsec)", "Component", "Subcomponent", "Expression"]]

        self.data.sort_values(by=["Time (µsec)"], ascending=[False], inplace=True)

    def subcomponent_report(
        self,
        filename: str | Path = "expression-timing-subcomponents.html",
        cutoff_secs=0.1,
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

        # include only expressions that took longer than cutoff_secs
        df = self.data[self.data["Time (µsec)"] >= cutoff_secs * 1e6]

        # convert the time to seconds
        df["Time (µsec)"] /= 1e6
        df.rename(columns={"Time (µsec)": "Time (sec)"}, inplace=True)

        # format and write the report to HTML
        df = (
            df.style.format(
                {
                    "Time (sec)": lambda x: f"{x:.3f}",
                }
            )
            .background_gradient(
                axis=0, gmap=df["Time (sec)"], cmap="YlOrRd", subset=["Time (sec)"]
            )
            .hide(axis="index")
            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
            .set_properties(**{"padding": "0 5px"}, subset=["Time (sec)"])
        )
        df.to_html(self.log_dir.joinpath(filename), index=False)

    def component_report_data(self, cutoff_secs: float = 0.1):
        """
        Return the data for the component report.
        """
        df = (
            self.data.groupby(["Component", "Expression"])
            .agg({"Time (µsec)": "sum"})
            .reset_index()[["Time (µsec)", "Component", "Expression"]]
            .sort_values(by=["Time (µsec)"], ascending=[False])
        )

        # include only expressions that took longer than cutoff_secs
        df = df[df["Time (µsec)"] >= cutoff_secs * 1e6]

        # convert the time to seconds
        df["Time (µsec)"] /= 1e6
        df.rename(columns={"Time (µsec)": "Time (sec)"}, inplace=True)
        return df

    def component_report(
        self,
        filename: str | Path = "expression-timing-components.html",
        cutoff_secs=0.1,
        style: Literal["simple", "grid"] = "simple",
    ) -> None:
        """Write component-level aggregations to an HTML file.

        This will aggregate the expression timings by component, which may better
        reveal expressions that are more problematic because they are evaluated
        multiple times.

        Parameters
        ----------
        filename : str | Path
            The name of the file to write the HTML to. If a relative path is given,
            it will be written in the log directory.
        cutoff_secs : float
            The cutoff time in seconds. Only expressions with a runtime greater than
            this will be included in the HTML file. This is used to avoid writing a
            huge report full of expressions that run plenty fast.
        style : "simple" | "grid", default "simple"
            The style of the report. Either "simple" or "grid". "simple" is a
            simple HTML table, "grid" is a JavaScript data grid.
        """

        df = self.component_report_data(cutoff_secs=cutoff_secs)

        if style == "simple":
            # format and write the report to HTML in a simple table
            df = (
                df.style.format(
                    {
                        "Time (sec)": lambda x: f"{x:.3f}",
                    }
                )
                .background_gradient(
                    axis=0, gmap=df["Time (sec)"], cmap="YlOrRd", subset=["Time (sec)"]
                )
                .hide(axis="index")
                .set_table_styles(
                    [{"selector": "th", "props": [("text-align", "left")]}]
                )
                .set_properties(**{"padding": "0 5px"}, subset=["Time (sec)"])
            )
            df.to_html(self.log_dir.joinpath(filename), index=False)
        elif style == "grid":
            template = """<html lang="en">
            <head>
                <!-- Includes all JS & CSS for the JavaScript Data Grid -->
                <script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>
            </head>
            <body>
                <!-- Data Grid container -->
                <div id="myGrid" ></div>
                <script>
                // Grid Options: Contains all of the Data Grid configurations
                const gridOptions = {
                    // Row Data: The data to be displayed.
                    rowData: <<ROWDATA>>,
                    // Column Definitions: Defines the columns to be displayed.
                    columnDefs: [
                        { field: "Time (sec)", flex: 1, sort: "desc" },
                        { field: "Component", flex: 2, filter: true },
                        { field: "Expression", flex: 7, filter: true, sortable: false, wrapText: true, autoHeight: true }
                    ]
                };
                // Your Javascript code to create the Data Grid
                const myGridElement = document.querySelector('#myGrid');
                agGrid.createGrid(myGridElement, gridOptions);
                </script>
            </body>
            </html>
            """
            with open(self.log_dir.joinpath(filename), "w") as f:
                f.write(template.replace("<<ROWDATA>>", df.to_json(orient="records")))
        else:
            raise ValueError(f"Unknown style {style}. Must be 'simple' or 'grid'.")
