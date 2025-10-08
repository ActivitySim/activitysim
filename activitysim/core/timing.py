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
            filename = state.get_expr_performance_log_file_path(str(self.log_file))

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
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"{label:11}\tExpression\n")
                for expression, elapsed_time in self.elapsed_times.items():
                    t = int(elapsed_time / scale)
                    f.write(f"{t: 11d}\t{expression}\n")
        except FileNotFoundError as err:
            if not filename.parent.exists():
                raise FileNotFoundError(
                    f"Could not write log file {filename!r}, parent directory does not exist."
                ) from err
            else:
                raise FileNotFoundError(
                    f"Could not write log file {filename!r}\n check permissions "
                    f"or path length ({len(str(filename))} characters in relative path, "
                    f"{len(str(filename.absolute()))} in absolute path)."
                ) from err


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

    def _read_log(self, log_file: Path) -> pd.DataFrame:
        """Read the log file and return a DataFrame."""
        df = pd.read_csv(log_file, sep="\t")
        if "(msec)" in df.columns[0]:
            df.columns = ["Time (µsec)"] + df.columns[1:].tolist()
            df.iloc[:, 0] = df.iloc[:, 0].astype(int) * 1_000
        elif "(sec)" in df.columns[0]:
            df.columns = ["Time (µsec)"] + df.columns[1:].tolist()
            df.iloc[:, 0] = df.iloc[:, 0].astype(int) * 1_000_000
        else:
            df.iloc[:, 0] = df.iloc[:, 0].astype(int)
        return df

    def __init__(self, state: State, collect_mp: bool = True) -> None:
        self.log_dir = state.get_expr_performance_log_file_path(".")
        self.default_cutoff = state.settings.expression_profile_cutoff
        raw_data = {}
        for f in self.log_dir.glob("*.log"):
            raw_data[str(f.stem)] = self._read_log(f)

        if raw_data:
            d = pd.concat(raw_data, names=["Component"]).reset_index()
            d["Proc"] = "main"
        else:
            d = None

        if collect_mp:
            raw_data = {}
            mp_log_dirs = state.get_expr_performance_log_file_path(".").glob(
                "*-expr-performance"
            )
            for mp_log_dir in mp_log_dirs:
                subproc_name = "-".join(mp_log_dir.stem.split("-")[:-2])
                for f in mp_log_dir.glob("*.log"):
                    raw_data[subproc_name, str(f.stem)] = self._read_log(f)
            if raw_data:
                d_mp = pd.concat(raw_data, names=["Proc", "Component"]).reset_index()
                if d is None:
                    d = d_mp
                else:
                    d = pd.concat([d, d_mp])

        # break trace labels into components and subcomponents
        try:
            d["Subcomponent"] = d["Component"].str.split(".", n=1).str[1]
            d["Component"] = d["Component"].str.split(".", n=1).str[0]
        except TypeError:
            # if the component is not a string, we cannot split it
            d["Subcomponent"] = ""
            d["Component"] = d["Component"].astype(str)
        self.data = d[
            ["Time (µsec)", "Proc", "Component", "Subcomponent", "Expression"]
        ]
        self.data = self.data.sort_values(by=["Time (µsec)"], ascending=[False])

    def subcomponent_report(
        self,
        filename: str | Path = "expression-timing-subcomponents.html",
        cutoff_secs: float | None = None,
        style: Literal["grid", "simple"] = "grid",
    ) -> None:
        """Write the data to an HTML file.

        Parameters
        ----------
        filename : str | Path
            The name of the file to write the HTML to. If a relative path is given,
            it will be written in the log directory.
        cutoff_secs : float, optional
            The cutoff time in seconds. Only expressions with a runtime greater than
            this will be included in the HTML file. This is used to avoid writing a
            huge report full of expressions that run plenty fast. If not provided,
            the default cutoff time from the settings is used.
        style : "simple" | "grid", default "simple"
            The style of the report. Either "simple" or "grid". "simple" is a
            simple HTML table, "grid" is a JavaScript data grid.
        """
        if cutoff_secs is None:
            cutoff_secs = self.default_cutoff

        # include only expressions that took longer than cutoff_secs
        df = self.data[self.data["Time (µsec)"] >= cutoff_secs * 1e6].copy()

        # convert the time to seconds
        df["Time (µsec)"] /= 1e6
        df = df.rename(columns={"Time (µsec)": "Time (sec)"})

        if style == "simple":
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
                .set_table_styles(
                    [{"selector": "th", "props": [("text-align", "left")]}]
                )
                .set_properties(**{"padding": "0 5px"}, subset=["Time (sec)"])
            )
            dat = df.to_html(index=False)
            dat = dat.replace("<table ", "<table data-sortable ")
            template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Expression Timing Report</title>
                <style>{SORTABLE_CSS}
                table[data-sortable] {{
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                    font-size: 14px;
                    line-height: 20px;
            }}</style>
            </head>
            <body>
            {dat}
            <script>{SORTABLE_JS}</script>
            """
            self.log_dir.joinpath(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir.joinpath(filename), "w") as f:
                f.write(template)
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
                        { field: "Subcomponent", flex: 4, filter: true, wrapText: true, autoHeight: true, editable: true },
                        { field: "Expression", flex: 7, filter: true, sortable: false, wrapText: true, autoHeight: true, editable: true }
                    ]
                };
                // Your Javascript code to create the Data Grid
                const myGridElement = document.querySelector('#myGrid');
                agGrid.createGrid(myGridElement, gridOptions);
                </script>
            </body>
            </html>
            """
            self.log_dir.joinpath(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir.joinpath(filename), "w") as f:
                f.write(template.replace("<<ROWDATA>>", df.to_json(orient="records")))

    def component_report_data(self, cutoff_secs: float | None = None):
        """
        Return the data for the component report.

        Parameters
        ----------
        cutoff_secs : float, optional
            The cutoff time in seconds. Only expressions with a runtime greater than
            this will be included in the report. This is used to avoid writing a
            huge report full of expressions that run plenty fast. If not provided,
            the default cutoff time from the settings is used.
        """
        if cutoff_secs is None:
            cutoff_secs = self.default_cutoff

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
        cutoff_secs: float | None = None,
        style: Literal["grid", "simple"] = "grid",
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
            dat = df.to_html(index=False)
            dat = dat.replace("<table ", "<table data-sortable ")
            template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Expression Timing Report</title>
                <style>{SORTABLE_CSS}
                table[data-sortable] {{
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                    font-size: 14px;
                    line-height: 20px;
            }}</style>
            </head>
            <body>
            {dat}
            <script>{SORTABLE_JS}</script>
            """
            self.log_dir.joinpath(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir.joinpath(filename), "w") as f:
                f.write(template)
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
                        { field: "Expression", flex: 7, filter: true, sortable: false, wrapText: true, autoHeight: true, editable: true }
                    ]
                };
                // Your Javascript code to create the Data Grid
                const myGridElement = document.querySelector('#myGrid');
                agGrid.createGrid(myGridElement, gridOptions);
                </script>
            </body>
            </html>
            """
            self.log_dir.joinpath(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir.joinpath(filename), "w") as f:
                f.write(template.replace("<<ROWDATA>>", df.to_json(orient="records")))
        else:
            raise ValueError(f"Unknown style {style}. Must be 'simple' or 'grid'.")


# Code below is from https://github.com/HubSpot/sortable
# Copyright (C) 2013 Adam Schwartz, http://adamschwartz.co
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


SORTABLE_JS = """
(function() {
  var SELECTOR, addEventListener, clickEvents, numberRegExp, sortable, touchDevice, trimRegExp;

  SELECTOR = 'table[data-sortable]';

  numberRegExp = /^-?[£$¤]?[\d,.]+%?$/;

  trimRegExp = /^\s+|\s+$/g;

  clickEvents = ['click'];

  touchDevice = 'ontouchstart' in document.documentElement;

  if (touchDevice) {
    clickEvents.push('touchstart');
  }

  addEventListener = function(el, event, handler) {
    if (el.addEventListener != null) {
      return el.addEventListener(event, handler, false);
    } else {
      return el.attachEvent("on" + event, handler);
    }
  };

  sortable = {
    init: function(options) {
      var table, tables, _i, _len, _results;
      if (options == null) {
        options = {};
      }
      if (options.selector == null) {
        options.selector = SELECTOR;
      }
      tables = document.querySelectorAll(options.selector);
      _results = [];
      for (_i = 0, _len = tables.length; _i < _len; _i++) {
        table = tables[_i];
        _results.push(sortable.initTable(table));
      }
      return _results;
    },
    initTable: function(table) {
      var i, th, ths, _i, _len, _ref;
      if (((_ref = table.tHead) != null ? _ref.rows.length : void 0) !== 1) {
        return;
      }
      if (table.getAttribute('data-sortable-initialized') === 'true') {
        return;
      }
      table.setAttribute('data-sortable-initialized', 'true');
      ths = table.querySelectorAll('th');
      for (i = _i = 0, _len = ths.length; _i < _len; i = ++_i) {
        th = ths[i];
        if (th.getAttribute('data-sortable') !== 'false') {
          sortable.setupClickableTH(table, th, i);
        }
      }
      return table;
    },
    setupClickableTH: function(table, th, i) {
      var eventName, onClick, type, _i, _len, _results;
      type = sortable.getColumnType(table, i);
      onClick = function(e) {
        var compare, item, newSortedDirection, position, row, rowArray, sorted, sortedDirection, tBody, ths, value, _compare, _i, _j, _k, _l, _len, _len1, _len2, _len3, _len4, _m, _ref, _ref1;
        if (e.handled !== true) {
          e.handled = true;
        } else {
          return false;
        }
        sorted = this.getAttribute('data-sorted') === 'true';
        sortedDirection = this.getAttribute('data-sorted-direction');
        if (sorted) {
          newSortedDirection = sortedDirection === 'ascending' ? 'descending' : 'ascending';
        } else {
          newSortedDirection = type.defaultSortDirection;
        }
        ths = this.parentNode.querySelectorAll('th');
        for (_i = 0, _len = ths.length; _i < _len; _i++) {
          th = ths[_i];
          th.setAttribute('data-sorted', 'false');
          th.removeAttribute('data-sorted-direction');
        }
        this.setAttribute('data-sorted', 'true');
        this.setAttribute('data-sorted-direction', newSortedDirection);
        tBody = table.tBodies[0];
        rowArray = [];
        if (!sorted) {
          if (type.compare != null) {
            _compare = type.compare;
          } else {
            _compare = function(a, b) {
              return b - a;
            };
          }
          compare = function(a, b) {
            if (a[0] === b[0]) {
              return a[2] - b[2];
            }
            if (type.reverse) {
              return _compare(b[0], a[0]);
            } else {
              return _compare(a[0], b[0]);
            }
          };
          _ref = tBody.rows;
          for (position = _j = 0, _len1 = _ref.length; _j < _len1; position = ++_j) {
            row = _ref[position];
            value = sortable.getNodeValue(row.cells[i]);
            if (type.comparator != null) {
              value = type.comparator(value);
            }
            rowArray.push([value, row, position]);
          }
          rowArray.sort(compare);
          for (_k = 0, _len2 = rowArray.length; _k < _len2; _k++) {
            row = rowArray[_k];
            tBody.appendChild(row[1]);
          }
        } else {
          _ref1 = tBody.rows;
          for (_l = 0, _len3 = _ref1.length; _l < _len3; _l++) {
            item = _ref1[_l];
            rowArray.push(item);
          }
          rowArray.reverse();
          for (_m = 0, _len4 = rowArray.length; _m < _len4; _m++) {
            row = rowArray[_m];
            tBody.appendChild(row);
          }
        }
        if (typeof window['CustomEvent'] === 'function') {
          return typeof table.dispatchEvent === "function" ? table.dispatchEvent(new CustomEvent('Sortable.sorted', {
            bubbles: true
          })) : void 0;
        }
      };
      _results = [];
      for (_i = 0, _len = clickEvents.length; _i < _len; _i++) {
        eventName = clickEvents[_i];
        _results.push(addEventListener(th, eventName, onClick));
      }
      return _results;
    },
    getColumnType: function(table, i) {
      var row, specified, text, type, _i, _j, _len, _len1, _ref, _ref1, _ref2;
      specified = (_ref = table.querySelectorAll('th')[i]) != null ? _ref.getAttribute('data-sortable-type') : void 0;
      if (specified != null) {
        return sortable.typesObject[specified];
      }
      _ref1 = table.tBodies[0].rows;
      for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
        row = _ref1[_i];
        text = sortable.getNodeValue(row.cells[i]);
        _ref2 = sortable.types;
        for (_j = 0, _len1 = _ref2.length; _j < _len1; _j++) {
          type = _ref2[_j];
          if (type.match(text)) {
            return type;
          }
        }
      }
      return sortable.typesObject.alpha;
    },
    getNodeValue: function(node) {
      var dataValue;
      if (!node) {
        return '';
      }
      dataValue = node.getAttribute('data-value');
      if (dataValue !== null) {
        return dataValue;
      }
      if (typeof node.innerText !== 'undefined') {
        return node.innerText.replace(trimRegExp, '');
      }
      return node.textContent.replace(trimRegExp, '');
    },
    setupTypes: function(types) {
      var type, _i, _len, _results;
      sortable.types = types;
      sortable.typesObject = {};
      _results = [];
      for (_i = 0, _len = types.length; _i < _len; _i++) {
        type = types[_i];
        _results.push(sortable.typesObject[type.name] = type);
      }
      return _results;
    }
  };

  sortable.setupTypes([
    {
      name: 'numeric',
      defaultSortDirection: 'descending',
      match: function(a) {
        return a.match(numberRegExp);
      },
      comparator: function(a) {
        return parseFloat(a.replace(/[^0-9.-]/g, ''), 10) || 0;
      }
    }, {
      name: 'date',
      defaultSortDirection: 'ascending',
      reverse: true,
      match: function(a) {
        return !isNaN(Date.parse(a));
      },
      comparator: function(a) {
        return Date.parse(a) || 0;
      }
    }, {
      name: 'alpha',
      defaultSortDirection: 'ascending',
      match: function() {
        return true;
      },
      compare: function(a, b) {
        return a.localeCompare(b);
      }
    }
  ]);

  setTimeout(sortable.init, 0);

  if (typeof define === 'function' && define.amd) {
    define(function() {
      return sortable;
    });
  } else if (typeof exports !== 'undefined') {
    module.exports = sortable;
  } else {
    window.Sortable = sortable;
  }

}).call(this);
"""

SORTABLE_CSS = """
/* line 2, ../sass/_sortable.sass */
table[data-sortable] {
  border-collapse: collapse;
  border-spacing: 0;
}
/* line 6, ../sass/_sortable.sass */
table[data-sortable] th {
  vertical-align: bottom;
  font-weight: bold;
}
/* line 10, ../sass/_sortable.sass */
table[data-sortable] th, table[data-sortable] td {
  text-align: left;
  padding: 10px;
}
/* line 14, ../sass/_sortable.sass */
table[data-sortable] th:not([data-sortable="false"]) {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  -o-user-select: none;
  user-select: none;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
  -webkit-touch-callout: none;
  cursor: pointer;
}
/* line 26, ../sass/_sortable.sass */
table[data-sortable] th:after {
  content: "";
  visibility: hidden;
  display: inline-block;
  vertical-align: inherit;
  height: 0;
  width: 0;
  border-width: 5px;
  border-style: solid;
  border-color: transparent;
  margin-right: 1px;
  margin-left: 10px;
  float: right;
}
/* line 40, ../sass/_sortable.sass */
table[data-sortable] th[data-sorted="true"]:after {
  visibility: visible;
}
/* line 43, ../sass/_sortable.sass */
table[data-sortable] th[data-sorted-direction="descending"]:after {
  border-top-color: inherit;
  margin-top: 8px;
}
/* line 47, ../sass/_sortable.sass */
table[data-sortable] th[data-sorted-direction="ascending"]:after {
  border-bottom-color: inherit;
  margin-top: 3px;
}"""
