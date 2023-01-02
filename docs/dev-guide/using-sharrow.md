
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
ActivitySim to facilitate reading from a DataFrame as it it were a 2-d array.  As it
is a special purpose class written in Python, the numba compiler cannot handle it directly.
Fortunately, sharrow provides an alternative: passing the size terms as a xarray `DataArray`.
This has a slightly different interface, so the sharrow and legacy evaluation modes require
different expressions. The switching expression is used to handle the DataFrameMatrix
on the left (before "# sharrow:") and the DataArray on the right.

### Performance Considerations

Sharrow is usually expected to bring significant performance gains to ActivitySim.
However, the numba engine beneath sharrow has a few known performance limitations,
most notably when using [strings](https://numba.readthedocs.io/en/stable/reference/pysupported.html#str).
To enhance performance, it is best to limit the number of string operations,
including simple equality checks (e.g. `df.tour_purpose == 'work'`).  Ideally,
such string operations won't appear in utility specifications at all, or if they
do appear, they are executed only once and stored in a temporary value for re-use
as needed.

For models with utility expressions that include a lot of string comparisons,
(e.g. because they are built for the legacy `pandas.eval` interpreter and have not
been updated) sharrow can be disabled by setting `sharrow_skip: true` in the
component's configuration yaml file.

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
they can be identifed by regular expressions ("regex") instead of explicitly
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
