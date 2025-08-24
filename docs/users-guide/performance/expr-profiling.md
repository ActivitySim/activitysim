# Expression Profiling

Part of the appeal of ActivitySim is in its flexibility: it is possible to craft
a massive variety of mathematical forms and relationships through creative use
of the expressions found in most component spec files. But as we have all learned
from Spider-Man, with great power comes great responsibility. Users can write
arbitrary code in spec files, and the runtime performance of ActivitySim will
depend on the parsimony and efficiency of that code.

Sometimes these spec files can be large, and it may be difficult to determine
simply by inspection which expressions in a given spec file are faster or slower.
ActivitySim now offers an expression-level profiling tool to assist in diagnosing
performance problems that arise from inefficient spec files.

```{important}
At this time,
expression profiling only works for the evaluation of expressions in "legacy" mode.
It does not work in "sharrow" mode, as the compiled expressions run with sharrow
are not run in a serial fashion and are not able to be profiled in the same way.
```

## Profiling an Entire Model Run

The simplest way to use the expression profiler is to set the
[`expression_profile`](activitysim.core.configuration.Settings.expression_profile)
configuration setting in the top level model settings (typically `settings.yaml`):

```yaml
expression_profile: true
```

This will cause the profiler to be activated for all expressions in the model,
across all components. This includes expressions in the spec files, as well as
expressions in all preprocessors and annotators.  Each time the expressions in
any spec file are evaluated, the profiler will record the time taken to evaluate
each expression.  An "expr-performance" subdirectory will be created in the model's
logging directory, and a new log file will be created each time the expressions in
any spec file are evaluated. The file is named according to the `trace_label` found
where the expressions are being evaluated. It will include a list of all the evaluated
expressions from the spec file, along with the time taken to evaluate each expression.
For multi-processed models, each subprocess will create its own log file directory,
similar to the logging directory structure for the other model components.

## Summary Outputs

At the end of a model run where the `expression_profile` setting is active,
ActivitySim will also create a pair of summary files in the "expr-performance"
subdirectory.  The first is named "expression-timing-subcomponents.html",
and contains a simple concatenation of the runtimes of
expressions in the various subcomponents stored in the log files,
filtered to only include expressions that tool a notable amount of time.
By default, this is set to 0.1 seconds, but can be changed by setting the
[`expression_profile_cutoff`](activitysim.core.configuration.Settings.expression_profile_cutoff)
configuration setting in the model settings.

The second file, "expression-timing-components.html", shows an aggregate
summary of the runtimes for each expression,
aggregated across all the log files. The aggregation is by model component and
expression, so that this summary includes the total time taken to evaluate each
expression within each model component, recognizing that identical expressions
may be evaluated multiple times in different model subcomponents (e.g. across
different purposes, or tour numbers, etc.).  This more aggregated summary is
typically the one that will be most useful for identifying expressions that
provide the most overall potential for performance improvement via streamlining.

Users should note that the expression profiler is not a substitute for good coding
practices.  It will not necessarily identify all performance problems, and it is not
able to suggest improvements to the expressions.  It is simply a tool to help users
identify which expressions are taking the most time to evaluate, and therefore
which expressions are the best candidates for improvement.

Also, users should understand that the expression profiler is not directly measuring the
computational complexity of the expressions, but rather the time taken to evaluate
the expressions. This time can be affected by a number of factors, including the
complexity of the expression, the size of the data being processed, and whether
there are other processes running on the machine at the same time competing for
resources. For multiprocessing model runs, those other processes may include
other the subprocesses of ActivitySim, which may lead to surprising results.

There is also no adjustment made for parallelization of the expression evaluations.
For example, if the same expression is evaluated in parallel across 8 processes on
a machine with 8 cores, and each process takes 0.1 seconds to evaluate the expression,
the profiler will still show that the expression took 0.8 seconds to evaluate, even
though the total wall clock time taken to evaluate the expression across all processes
was only 0.1 seconds.

Profiling expressions also adds some overhead to the model run, increasing the
total runtime of the model by a modest but noticeable amount. In consortium
[experiments](https://github.com/ActivitySim/activitysim/pull/936#issuecomment-3165410169)
with this tool, runtime for the full-scale SANDAG model was found to
increase by approximately 12.5% when the profiler was enabled, adding more than
13 minutes to a model run that already took 105 minutes. Users should thus
be careful about using the profiler in production runs.  It is recommended turn off
the profiler in production runs, and only use it for debugging and development.

## Profiling Individual Components

The expression profiler can also be used to profile individual components, rather
than the entire model. This is done by setting the `compute_settings.performance_log`
attribute for the component in the model settings. This attribute can be set to the
filename where the profiler log file should be written, which will override
the default behavior of writing the log file to the "expr-performance" subdirectory.
This feature only works for components that are run in a single process, and which
have a `compute_settings` attribute. It is generally not recommended to use this
feature unless a specific component is suspected of having atypical performance
problems, as it will not provide the same summary reporting as profiling the entire
model.
