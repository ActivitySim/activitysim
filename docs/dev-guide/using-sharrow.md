
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

### Top-Level Activation Options

Activating sharrow is done at the top level of the model settings file, typically
`settings.yaml`, by setting the `sharrow` configuration setting to `True`:

```yaml
sharrow: True
```

The default operation for sharrow is to attempt to use the sharrow compiler for
all model specifications, and to revert to the legacy pandas-based evaluation
if the sharrow compiler encounters a problem.  Alternatively, the `sharrow`
setting can also be set to `require` or `test`.  The `require` setting
will cause the model simply fail if sharrow encounters a problem, which is
useful if the user is interested in ensuring maximum performance.
The `test` setting will run the model in a mode where both sharrow and the
legacy pandas-based evaluation are run on each model specification, and the
results are compared to ensure they are substantially identical.  This is
useful for debugging and testing, but is not recommended for production runs
as it is much slower than running only one evaluation path or the other.

Testing is strongly recommended during model development, as it is possible
to write expressions that are valid in one evaluation mode but not the other.
This can happen if model data includes `NaN` values
(see [Performance Considerations](#performance-considerations)), or when
using arithmatic on logical values
(see [Arithmetic on Logical Values](#arithmetic-on-logical-values)).

### Caching of Precompiled Functions

The first time you run a model with sharrow enabled, the compiler will run
and create a cache of compiled functions.  This can take a long time, especially
for models with many components or complex utility specifications.  However,
once the cache is created, subsequent runs of the model will be much faster.
By default, the cached functions are stored in a subdirectory of the
`platformdirs.user_cache_dir` directory, which is located in a platform-specific
location:

- Windows: `%USERPROFILE%\AppData\Local\ActivitySim\ActivitySim\Cache\...`
- MacOS: `~/Library/Caches/ActivitySim/...`
- Linux: `~/.cache/ActivitySim/...` or `~/$XDG_CACHE_HOME/ActivitySim/...`

The cache directory can be changed from this default location by setting the
[`sharrow_cache_dir`](activitysim.core.configuration.FileSystem.sharrow_cache_dir)
setting in the `settings.yaml` file.  Note if you change this setting and provide
a relative path, it will be interpreted as relative to the model working directory,
and cached functions may not carry over to other model runs unless copied there
by the user.

## Model Design Requirements

Activating the `sharrow` optimizations also requires using the new
[`SkimDataset`](skim-datasets) interface for skims instead of the legacy
[`SkimDict`](activitysim.core.skim_dictionary.SkimDict), and internally
recoding zones into a zero-based contiguous indexing scheme.

### Zero-based Recoding of Zones

Using sharrow requires recoding zone id's to be zero-based contiguous index
values, at least for internal usage.  This recoding needs to be written into
the input table list explicitly.  For example, the following snippet of a
`settings.yaml` settings file shows the process of recoding zone ids in
the input files.

```yaml
input_table_list:
  - tablename: land_use
    filename: land_use.csv
    index_col: zone_id
    recode_columns:
      zone_id: zero-based
  - tablename: households
    filename: households.csv
    index_col: household_id
    recode_columns:
      home_zone_id: land_use.zone_id
```

For the `land_use` table, the `zone_id` field is marked for recoding explicitly
as `zero-based`, which will turn whatever nominal id's appear in that column into
zero-based index values, as well as store a mapping of the original values that
is used to recode and decode zone id's when used elsewhere.

The first "elsewhere" recoding is in the households input table, where we will
map the `home_zone_id` to the new zone id's by pointing the recoding instruction
at the `land_use.zone_id` field.  If zone id's appear in other input files, they
need to be recoded in those fields as well, using the same process.

The other places where we need to handle zone id's is in output files.  The
following snippet of a `settings.yaml` settings file shows how those id's are
decoded in various output files.  Generally, for columns that are fully populated
with zone id's (e.g. tour and trip ends) we can apply the `decode_columns` settings
to reverse the mapping and restore the nominal zone id's globally for the entire
column of data.  For columns where there is some missing data flagged by negative
values, the "nonnegative" filter is prepended to the instruction.

```yaml
output_tables:
  action: include
  tables:
    - tablename: land_use
      decode_columns:
        zone_id: land_use.zone_id
    - tablename: households
      decode_columns:
        home_zone_id: land_use.zone_id
    - tablename: persons
      decode_columns:
        home_zone_id: land_use.zone_id
        school_zone_id: nonnegative | land_use.zone_id
        workplace_zone_id: nonnegative | land_use.zone_id
    - tablename: tours
      decode_columns:
        origin: land_use.zone_id
        destination: land_use.zone_id
    - tablename: trips
      decode_columns:
        origin: land_use.zone_id
        destination: land_use.zone_id
```

## Measuring Performance

Testing with sharrow requires two steps: test mode and production mode.

In test mode, the code is run to compile all the spec files and
ascertain whether the functions are working correctly.  Test mode is expected
to be slow, potentially much slower than older versions of ActivitySim,
especially for models with small populations and zone systems, as the compile
time is a function of the complexity of the utility functions and *not* a
function of the number of households or zones. Once the compile and test is
complete, production mode can then just run the pre-compiled functions with
sharrow, which is much faster.

It is possible to run test mode and production mode independently using the
existing `activitysim run` command line tool, pointing that tool to the test
and production configurations directories as appropriate.

To generate a meaningful measure of performance enhancement, it is necessary
to compare the runtimes in production mode against equivalent runtimes with
sharrow disabled.  This is facilitated by the `activitysim workflow` command
line tool, which permits the use of pre-made batches of activitysim runs, as
well as automatic report generation from the results.  For more details on the
use of this tool, see [workflows](workflows).


## Sharrow Compatability and Limitations

In general, utility specification files contain a sequence of :ref:`expressions` that
are evaluated for each row in a main DataFrame.  In legacy ActivitySim,
there are two fundamental evaluation modes:

- `pandas.DataFrame.eval`, which is the default, and
- plain Python `eval`, which is used when the expression is prefixed with an `@` symbol.

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
uses a completely different evaluation system routed through numba. The `@` prefix
does not need to be stripped or added anywhere, it is simply ignored. The expression
can reference other columns of the main dataframe directly or indirectly, so that
either "income" or "df.income" is a valid reference to that column in the main
dataframe.  You can also reference other variables, including skims as usual for each component
and constants defined in in the model settings file.  Within scheduling models,
references can also be made to the timetable as "tt", accessing that interfaces
methods to compute presence and length of adjacent and available time windows.
However, you *cannot* write arbitrary Python code.  External functions are not allowed unless they have
already been compiled with numba's `njit` wrapper.  Typically, unless specifically
constructed to be allowed for a model component, cross-referencing, merging or
reindexing against tables other than the main dataframe is not allowed. Such
customizations will generally require a custom extension.

Sharrow only runs for utility specification files.  ActivitySim also features the
ability to apply pre-processors and post-processors to annotate tables, which use
specification files that look very similar to utility specifications.  These
functions do *not* run through sharrow, and are a great place to relocate expressions
that need to run arbitrary code or join data from other tables.

### Temporary Variables

Temporary variables can be created from `@` mode expressions by adding a variable
name beginning with an underscore before the `@`, e.g. `_stops_on_leg@df.trip_count-1`.

In legacy mode, temporary variables are useful but they can consume substantial
memory, as the variable is computed and stored for every row in the entire dataframe
before it can be used in other expressions.  In sharrow, temporary variables are
allocated, used, and freed for each row separately, so no extra memory is required.

### Pandas-only Expressions

In legacy mode, expressions can be evaluated using expressions that tap into the
full pandas library, including the ability to call pandas functions and methods
directly.  This is not possible in sharrow, as the expressions are compiled into
numba code, which does not have access to the pandas library.  If a pandas function
is needed, it must be called in a pre-processor.  However, many pandas functions
can be replaced with numpy functions, which are available in numba.  For example,
`df.income.fillna(0)` can be replaced with `np.nan_to_num(df.income)`.

### Switchable Expressions

As a general rule, it is best to write each utility expression in a manner that
is cross-compatible between sharrow and legacy evaluation modes, even if that
means transitioning a few expressions or parts of expressions into preprocessors.

However, sometimes this is not possible or makes writing the expressions excessively
complex.  In this case, it is possible to write a toggling expression, where the
individual expression evaluated is different for sharrow and legacy modes.  The
special comment string `# sharrow:` splits the expression, with everything before this
comment evaluated under the legacy process, and everything after evaluated only
when sharrow is enabled.

An example of a switching expression is found in the trip destination utilities found
in several examples:

    `@np.log1p(size_terms.get(df.alt_dest, df.purpose)) # sharrow: np.log1p(size_terms['sizearray'])`

Here, `size_terms` is a DataFrameMatrix class instance, a special class written into
ActivitySim to facilitate reading from a DataFrame as if it were a 2-d array.  As it
is a special purpose class written in Python, the numba compiler cannot handle it directly.
Fortunately, sharrow provides an alternative: passing the size terms as a xarray `DataArray`.
This has a slightly different interface, so the sharrow and legacy evaluation modes require
different expressions. The switching expression is used to handle the DataFrameMatrix
on the left (before "# sharrow:") and the DataArray on the right.

### Optional Variables

In some cases, a variable may be used where it is available, but is not strictly
necessary for the model to run.  For example, a model may have a reference to
mode choice logsums, but the model can still run without them, if it is being used
prior to when logsums are calculated.  In this case, the variable can be accessed
using the `get` method, which allows for a default value if the variable is not found.

    `@df.get('mode_choice_logsum', 0)`

### Performance Considerations

Sharrow is usually expected to bring significant performance gains to ActivitySim.
However, the numba engine beneath sharrow has a few known performance limitations,
most notably when using [strings](https://numba.readthedocs.io/en/stable/reference/pysupported.html#str).
To enhance performance, it is best to limit the number of string operations,
including simple equality checks (e.g. `df.tour_purpose == 'work'`).  Ideally,
such string operations won't appear in utility specifications at all, or if they
do appear, they are executed only once and stored in a temporary value for re-use
as needed.

A good approach to reduce string operations in model spec files is to convert
string columns to integer or categorical columns in preprocessors.  This can
be done using the `map` method, which can be used to convert strings to integers,
for example:

    `df['fuel_type'].map({'Gas': 1, 'Diesel': 2, 'Hybrid': 3}).fillna(-1).astype(int)`

Alternatively, data columns can be converted to categorical columns with well-defined
structures. Recent versions of sharrow have made significant improvements in
handling of unordered categorical values, allowing for the use of possibly
more intuitive categorical columns.  For example, the fuel type column above
could instead be redefined as a categorical column with the following code:

    `df['fuel_type'].astype(pd.CategoricalDtype(categories=['Gas', 'Diesel', 'Hybrid'], ordered=False))`

It is important that the categories are defined with the same set of values
in the same order, as any deviation will from this will void the compiler cache
and cause the model specification to be recompiled.  This means that using
`x.astype('category')` is not recommended, as the categories will be inferred
from the data and may not be consistent across multiple calls to the model
specification evaluator.

```{note}
Beginning with ActivitySim version 1.3, string-valued
columns created in preprocessors are converted to categorical columns automatically,
which means that ignoring encoding for string-valued outputs is equivalent to
using the `astype('category')` method, and is not recommended.
```

For models with utility expressions that include a lot of string comparisons,
(e.g. because they are built for the legacy `pandas.eval` interpreter and have not
been updated) sharrow can be disabled by setting

```yaml
compute_settings:
  sharrow_skip: true
```

in the component's configuration yaml file.

In addition, by default sharrow also tries to optimize performance by setting the
`fastmath` flag to True in the numba compiler.  This makes the compiler generate
faster code, by assuming that all variables have finite values (not NaN or Inf).
If the model has expressions that use variables that can contains NaN or Inf
values, the `fastmath` flag can be disabled by setting

```yaml
compute_settings:
  fastmath: false
```

### Multiprocessing Performance

Sharrow leverages a number of performance enhancing techniques, including
parallelization of various computations. This multi-threading can provide significant
benefits within a single-process, but if enabled alongside ActivitySim's multiprocessing
paradigm, the multi-threading does more harm than good, as too many threads will
compete for limited computational resources. To avoid this, the user should completely
disable multi-threading and rely exclusively on multiprocessing to generate parallelism.
This can be done by setting a number of thread-limiting environment variables before
running Python, or immediately at the start of a Python script before ActivitySim
is loaded:

```python
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

### Limited Tracing and Estimation Mode Capabilities

When running sharrow-optimized code, large parts of certain calculations are routed
through numba to enhance speed and reduce memory footprint.  Many intermediate arrays
are not created, or are allocated and released by numba in a manner not visible to the
general Python interpreter.  As a result, in many places the "trace" capabilities of
ActivitySim are limited.  These capabilities output intermediate calculations in
extreme detail and are typically only used in debugging and model development, not
in production model runs that need to be optimized for speed.  Tracing features can
be re-enabled by switching sharrow off or by using "test" mode, which runs both sharrow
and legacy evaluations together to confirm they are substantially identical.  In this
fashion, detailed tracing (albeit slow) remains available for users on all models when
needed.

Similar constraints apply to estimation mode, as complete estimation mode output
capabilities are not yet integrated with the sharrow engine.  Estimation mode remains
fully available when running with sharrow disabled.


### Arithmetic on Logical Values

In expressions written in specification files, boolean values must be treated with
care.  When an expression is evaluated in the legacy implementation, the addition
of two boolean values will be processed according to numpy logic, such that:

```python
np.array([True]) + np.array([True]) == np.array([True])
np.array([True]) + np.array([False]) == np.array([True])
np.array([False]) + np.array([True]) == np.array([True])
np.array([False]) + np.array([False]) == np.array([False])
```

When the same expression is evaluated using sharrow, the expression is evaluated
using Pythonesque rules, such that logical values are implicitly upcast to integers,
giving:

```python
True + True == 2
True + False == 1
False + True == 1
False + False == 0
```

If this value is later upcast to a number and used in a mathematical calculation
(e.g. multiplied by a float-valued coefficient), obviously the results will vary,
as in the first case the result is never other than 1 or 0, but in the latter case
the result can also be 2.  This mismatch can be readily avoided by wrapping the
term in an extra logic gate, which will evaluate the same in both environments:

```python
(True + True)>0 == True
(True + False)>0 == True
(False + True)>0 == True
(False + False)>0 == False
```

(digital-encoding)=
## Digital Encoding

Sharrow is compatible with and able to efficiently use
[digital encoding](https://activitysim.github.io/sharrow/walkthrough/encoding.html).
These encodings are applied to data either prospectively (i.e. before ActivitySim
ever sees the skim data), or dynamically within a run using the
`taz_skims.digital-encoding` or `taz_skims.zarr-digital-encoding` settings in
the `network_los.yaml` settings file.  The only difference between these two
settings is that the former applies this digital encoding internally every
time you run the model, while the latter applies it prior to caching encoded
skims to disk in Zarr format (and then reuses those values without re-encoding
on subsequent runs with the same data).  Dictionary encoding (especially joint
dictionary encodings) can take a long time to prepare, so caching the values can
be useful. As read/write speeds for zarr files are fast, you usually won't
notice a meaningful performance degradation when caching, so the default is
generally to use `zarr-digital-encoding`.

Very often, data can be expressed adequately with far less memory than is
needed to store a standard 32-bit floating point representation.  There are
two simple ways to reduce the memory footprint for data: fixed point
encoding, or dictionary encoding.

### Fixed Point Encoding

In fixed point encoding, which is also sometimes called scaled integers,
data is multiplied by some factor, then rounded to the nearest integer.
The integer is stored in memory instead of a floating point value, but the
original value can be (approximately) restored by reversing the process.
An offset factor can also be applied, so that the domain of values does not
need to start at zero.

For example, instead of storing matrix table values as 32-bit floating point values,
they could be multiplied by a scale factor (e.g., 100)
and then converted to 16-bit integers. This uses half the
RAM and can still express any value (to two decimal point
precision) up to positive or negative 327.68.  If the lowest
values in that range are never needed, it can also be shifted,
moving both the bottom and top limits by a fixed amount. Then,
for a particular scale $\mu$ and shift $\xi$ (stored in metadata),
from any array element $i$ the implied (original) value $x$
can quickly be recovered by evaluating $(i / \mu) - \xi$.

Fixed point digital encoding can be applied to matrix tables in the skims
using options in the `network_los.yaml` settings file.  Making transformations
currently also requires shifting the data from OMX to ZARR file formats;
future versions of ActivitySim may accept digitally encoded data directly
from external sources.

To apply the default 16-bit encoding to individual named skim variables in the
TAZ skims, just give their names under the `zarr-digital-encoding` setting
like this:

```yaml
taz_skims:
    omx: skims.omx
    zarr: skims.zarr
    zarr-digital-encoding:
        - name: SOV_TIME
        - name: HOV2_TIME
```

If some variables can use less RAM and still be represented adequately with only
8-bit integers, you can specify the bitwidth as well:

```yaml
taz_skims:
    omx: skims.omx
    zarr: skims.zarr
    zarr-digital-encoding:
        - name: SOV_TIME
        - name: HOV2_TIME
        - name: SOV_TOLL
          bitwidth: 8
        - name: HOV2_TOLL
          bitwidth: 8
```

If groups of similarly named variables should have the same encoding applied,
they can be identified by regular expressions ("regex") instead of explicitly
giving each name.  For example:

```yaml
taz_skims:
    omx: skims.omx
    zarr: skims.zarr
    zarr-digital-encoding:
        - regex: .*_TIME
        - regex: .*_TOLL
          bitwidth: 8
```


### Dictionary Encoding

For dictionary encoding, a limited number of unique values are stored in a
lookup array, and then each encoded value is stored as the position of the
value (or its closest approximation) in the lookup array.  If there are fewer
than 256 unique values, this can allow the storage of those values to any level
of precision (even float64 if needed) while using only a single byte per array
element, plus a small fixed amount of overhead for the dictionary itself.  The
overhead memory doesn't scale with the dimensions of the array, so this works
particularly well for models with thousands of zones.

Dictionary encoding can be applied to a single variable in a similar fashion as
fixed point encoding, giving the dictionary bit width in the `by_dict` setting,
or as an additional setting value.

```yaml
taz_skims:
    omx: skims.omx
    zarr: skims.zarr
    zarr-digital-encoding:
        - name: TRANSIT_FARE
          by_dict: 8
        - name: TRANSIT_XFERS
          by_dict: true
          bitwidth: 8
```

The most dramatic memory savings can be found when the categorical correlation
(also known as [Cramér's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V))
between multiple variables is high.  In this case, we can encode more than one
matrix table using the same dictionary lookup indexes.  There may be some
duplication in the lookup table, (e.g. if FARE and XFER are joint encoded,
and if a FARE of 2.25 can be matched with either 0 or 1 XFER, the 2.25 would
appear twice in the lookup array for FARE, once for each value of XFER.)

Since it is the lookup *indexes* that scale with the number of zones and consume most
of the memory for large zone systems, putting multiple variables together into one
set of indexes can save a ton of memory, so long as the overhead of the lookup array
does not combinatorially explode (hence the need for categorical correlation).

Practical testing for large zone systems suggest this method of encoding can
reduce the footprint of some low variance data tables (especially transit data)
by 95% or more.

Applying joint dictionary encoding requires more than one variable name, so only
the `regex` form works here. Use wildcards to match on name patterns, or select a
few specific names by joining them with the pipe operator (|).

```yaml
taz_skims:
    omx: skims.omx
    zarr: skims.zarr
    zarr-digital-encoding:
        - regex: .*_FARE|.*_WAIT|.*_XFERS
          joint_dict: true
        - regex: FERRYTIME|FERRYFARE|FERRYWAIT
          joint_dict: true
```

For more details on all the settings available for digital encoding, see
[DigitalEncoding](activitysim.core.configuration.network.DigitalEncoding).

## Troubleshooting

If you encounter errors when running the model with sharrow enabled, it is
important to address them before using the model for analysis.  This is
especially important when errors are found running in "test" mode (activated
by `sharrow: test` in the top level settings.yaml).  Errors may
indicate that either sharrow or the legacy evaluator is not correctly processing
the mathematical expressions in the utility specifications.

### "utility not aligned" Error

One common error that can occur when running the model with sharrow in "test"
mode is the "utility not aligned" error.  This error occurs when a sharrow
compiled utility calculation does not sufficiently match the legacy utility
calculation.  We say "sufficiently" here because the two calculations may have
slight differences due to numerical precision optimizations applied by sharrow.
These optimizations can result in minor differences in the final utility values,
which are typically inconsequential for model results.  However, if the differences
are too large, the "utility not aligned" error will be raised.  This error does
not indicate whether the incorrect result is from the sharrow or legacy calculation
(or both), and it is up to the user to determine how to align the calculations
so they are reflective of the model developer's intent.

To troubleshoot the "utility not aligned" error, the user can use a Python debugger
to compare the utility values calculated by sharrow and the legacy evaluator.
ActivitySim also includes error handler code that will attempt to find the
problematic utility expression and print it to the console or log file, under the
heading "possible problematic expressions".  This can be helpful in quickly narrowing
down which lines of a specification file are causing the error.

Common causes of the "utility not aligned" error include:

- model data includes `NaN` values but the component settings do not
  disable `fastmath` (see [Performance Considerations](#performance-considerations))
- incorrect use of arithmatic on logical values (see
  [Arithmetic on Logical Values](#arithmetic-on-logical-values))

### Insufficient system resources

For large models run on large servers, it is possible to overwhelm the system
with too many processes and threads, which can result in the following error:

```
OSError: Insufficient system resources exist to complete the requested service
```

This error can be resolved by reducing the number of processes and/or threads per
process.  See [Multiprocessing](../users-guide/performance/multiprocessing.md) and
[Multithreading](../users-guide/performance/multithreading.md) in the User's Guide
for more information on how to adjust these settings.

### Permission Error

If running a model using multiprocessing with sharrow enabled, it is necessary
to have pre-compiled all the utility specifications to prevent the multiple
processes from competing to write to the same cache location on disk.  Failure
to do this can result in a permission error, as some processes may be unable to
write to the cache location.

```
PermissionError: The process cannot access the file because it is being used by another process
```

To resolve this error, run the model with sharrow enabled in single-process mode
to pre-compile all the utility specifications.  If that does not resolve the error,
it is possible that some compiling is being triggered in multiprocess steps that
is not being handled in the single process mode.  This is likely due to the presence
of string or categorical columns created in a preprocessor that are not being
stored in a stable data format.  To resolve this error, ensure that all expressions
in pre-processors are written in a manner that results in stable data types (e.g.
integers, floats, or categorical columns with a fixed set of categories).  See
see [Performance Considerations](#performance-considerations)) for examples.
