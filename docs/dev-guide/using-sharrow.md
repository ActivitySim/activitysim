
# Using Sharrow

This page will walk through an exercise of running a model with `sharrow`.

## How it Works

Sharrow accelerates ActivitySim in part by using numba to create optimized and
pre-compiled versions of utility specification files, and caching those bits
of code to disk.

```{important}
Running the compiler needs to be done in single-process mode, otherwise the
various process all do the compiling and compete to write to the same cache
location on disk, which is likely to fail.  You can safely run in
multiprocessing mode after all the compilation for all model components is
complete.
```


## Getting the Code

We'll assume that activitysim and sharrow have been installed in editable
mode, per the [developer install instructions](install.md).

The code to implement `sharrow` in `activitysim` is in the
[sharrow-black](https://github.com/camsys/activitysim/tree/sharrow-black) branch
of the [camsys/activitysim](https://github.com/camsys/activitysim) repository.
If you have checked out from this repository directly, you can just switch to
the correct branch:

```sh
cd activitysim
git switch sharrow-black
cd ..
```

If your local Git repository is configured with only the main ActivitySim
parent repo as its remote, you can find the code in
[PR #579](https://github.com/ActivitySim/activitysim/pull/579), accessible on the
command line using the `gh` tool:

```sh
cd activitysim
gh pr checkout 579  # use `gh auth login` first if needed
cd ..
```


## Prototype MTC Examples

Testing with sharrow requires two steps: test mode and production mode.

In test mode, the code is run to compile all the spec files and
ascertain whether the functions are working correctly.  Production mode
can then just run the pre-compiled functions with sharrow, which is much
faster.

You can run both, plus the ActivitySim in "legacy" mode that does not use sharrow,
as well as a reference implementation (version 1.0.4), all together using the
"mini" dataset for testing in one workflow.

```sh
activitysim workflow sharrow-contrast/mtc_mini
```

Alternatively, you can use the full size skims.  To test this model with
100k households and full skims (1475 zones), you can run the "mtc_full" workflow:

```sh
activitysim workflow sharrow-contrast/mtc_full
```

To use the full synthetic population as well, run the multiprocess workflow:

```sh
activitysim workflow sharrow-contrast/mtc_mp
```

Lastly, a comprehensive performance testing suite on the prototype MTC model
(warning: this takes hours!)

```sh
activitysim workflow sharrow-contrast/mtc_comprehensive
```

All these performance tests assume you have sufficient RAM to run without chunking.


## Sharrow Compatability and Limitations

In general, utility specification files contain a sequence of :ref:`expressions` that
are evaluated for each row in a main DataFrame.  In legacy ActivitySim,
there are two fundamental evaluation modes:

- `pandas.DataFrame.eval`, which is the default, and
- plain Python `eval`, which is used when the expression is prefixed with an "@" symbol.

Under the `pandas.DataFrame.eval` mode, expressions are evaluated within the context
of the current main dataframe only. References can be made to other columns
in that dataframe directly, i.e. if the dataframe contains a column named "income",
the expression can reference that value as "income".  However, the expression can
*only* reference other values that are stored in the current row of the dataframe.
The available syntax for these expressions is a subset of regular python, see the
[supported syntax](https://pandas.pydata.org/docs/user_guide/enhancingperf.html#supported-syntax)
section of the pandas documentation for details.

Under the plain `eval` mode, expressions are evaluated within a broader context
that potentially includes other variables (which other variables is component
dependent) and constants defined in the model settings file.  References to
columns in the main dataframe must be made indirectly via attribution, i.e. if
the dataframe contains a column named "income", the expression needs to reference
that value as "df.income".  Typically, references to skims, time tables, and anything
else that isn't the simple original dataframe use this mode.  While it is sometimes
not as fast as `pandas.DataFrame.eval`, this mode is far more flexible, as you
can write basically any valid Python expression, including calling other functions,
accessing or manipulating table metadata, indexes, or adjacent rows, etc.

Within sharrow, the distinction between these two modes is ignored, as sharrow
uses a completely different evaluation system routed through numba. The "@" prefix
does not need to be stripped or added anywhere, it is simply ignored. The expression
can reference other columns of the main dataframe directly or indirectly, so that
either "income" or "df.income" is a valid reference to that column in the main
dataframe.  You can also reference other variables, including skims, time tables,
and constants defined in in the model settings file.  However, you *cannot* write
any arbitrary Python code.  External functions are not allowed unless they have
already been compiled with numba's `njit` wrapper.  Typically, unless specifically
constructed to be allowed for a model component, cross-referencing, merging or
reindexing against tables other than the main dataframe is not allowed.

Sharrow only runs for utility specification files.  ActivitySim also features the
ability to apply pre-processors and post-processors to annotate tables, which use
specification files that look very similar to utility specifications.  These
functions do *not* run through sharrow, and are a great place to relocation expressions
that need to run arbitrary code or join data from other table.

### Temporary Variables

Temporary variables can be created from "@" mode expressions by adding a variable
name beginning with an underscore before the "@", e.g. "_stops_on_leg@df.trip_count-1".

In legacy mode, temporary variables are useful but they can consume substantial
memory, as the variable is computed and stored for every row in the entire dataframe
before it can be used in other expressions.  In sharrow, temporary variables are
allocated, used, and freed for each row seperately, so no extra memory is required.

### Switchable Expressions

As a general rule, it is best to write each utility expression in a manner that
is cross-compatible between sharrow and legacy evaluation modes, even if that
means transitioning a few expressions or parts of expressions into preprocessors.

However, sometimes this is not possible or makes writing the expressions excessively
complex.  In this case, it is possible to write a toggling expression, where the
individual expression evaluated is different for sharrow and legacy modes.  The
special comment string "# sharrow:" splits the expression, with everything before this
comment evaluated under the legacy process, and everything after evaluated only
when sharrow is enabled.

### Performance Considerations

Sharrow is usually expected to bring significant performance gains to ActivitySim.
However, the numba engine beneath sharrow has a few known performance limitations,
most notably when using [strings](https://numba.readthedocs.io/en/stable/reference/pysupported.html#str).
To enhance performance, it is best to limit the number of string operations,
including simple equality checks (e.g. `df.tour_purpose == 'work'`).  Ideally,
such string operations won't appear in utility specifications at all, or if they
do appear, they are executed only once and stored in a temporary value for re-use
as needed.
