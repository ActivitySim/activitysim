# Runtime Performance

Achieving "good" runtime performance in ActivitySim depends on several factors,
including the model size (number of zones, number of households), model complexity,
(eg. number of components, length and complexity of utility specifications), and
hardware capabilities (amount of RAM, number of CPU cores, amount of on-chip cache,
speed of storage devices).


## Multiprocessing

ActivitySim implements an extensive set of multiprocessing capabilities to take
advantage of modern multicore CPUs.  The number of processes used is controlled
by the `multiprocessing` configuration setting in the `settings.yaml` file.

There are a number of settings that need to be configured to use multiprocessing
in ActivitySim.

- multiprocess

    A boolean setting that enables or disables multiprocessing.

- num_processes

  This controls the number of processes used by ActivitySim in multiprocessing,
  unless the value is overloaded in a particular multiprocess step.

- multiprocess_steps

  This setting controls which model components are handled by multiprocessing,
  and when the model flow needs to begin and end each multiprocessing step. An
  individual multiprocessing step can include one or more model components, and
  can be configured to use a different number of processes than other
  multiprocessing step

These settings are described in more detail in the [Configuration](configuration)
section.

### How Many Processes

The number of processes to use in multiprocessing is a complex question.  Using
more processes might typically be expected to reduce overall model runtime, and
it usually does for the first few processes added.  However, as the number of
processes grows, the marginal benefit of adding more processes decreases.  There
is also additional overhead associated with each process, so using too many
processes can actually slow down the model.

Experiments by the ActivitySim consortium have generally found that the optimal
number of processes is usually around 10, even if the machine has many more
cores available.  Of course, this is a general rule of thumb, and the optimal
number of processes will vary depending on the specific model being run, the
hardware being used, and the specific configuration of the model.


## Multithreading

Some parts of ActivitySim are also multithreaded, using various different
multithreading technologies. This includes large matrix multiplication operations
used in certain utility calculations (multithreaded by the MKL library where
available) as well as computations accelerated using the sharrow library (which
uses numba to parallelize certain operations), as well as potentially other
parallelization subroutines in other libraries upon which ActivitySim depends.

In general, it is desirable to *disable* multithreading in ActivitySim when using
multiprocessing, as the two technologies can interfere with each other.

Disabling multithreading can be done two ways, either by setting environment
variables before starting a Python instance that will run ActivitySim, or by
setting certain environment variables within Python before loading ActivitySim
or it's dependencies (especially numba).


### Disabling Multithreading via Environment Variables

Setting environment variables before starting a Python instance is the most
reliable way to disable multithreading in ActivitySim, as it ensures that the
settings are made before any Python libraries are loaded.  The downside of this
is that the exact commands needed are platform and shell-specific (i.e. the
commands for Powershell, Windows `cmd.exe`, and Linux's bash are all different).

The environment variables that need to be set are:

- `MKL_NUM_THREADS` - This environment variable controls the number of threads
  used by the Intel Math Kernel Library (MKL), which is used by numpy and pandas
  for certain operations.  Setting this to 1 will disable multithreading in MKL.
- `NUMBA_NUM_THREADS` - This environment variable controls the number of threads
  used by the numba library, which is used by ActivitySim for certain operations.
  Setting this to 1 will disable multithreading in numba.
- `OMP_NUM_THREADS` - This environment variable controls the number of threads
  used by the OpenMP library, which is used by some libraries that ActivitySim
  depends on.  Setting this to 1 will disable multithreading in OpenMP.
- `NUMEXPR_NUM_THREADS` - This environment variable controls the number of threads
  used by the numexpr library, which is used by pandas for certain operations.
  Setting this to 1 will disable multithreading in numexpr.


### Disabling Multithreading via Python

Disabling multithreading via Python is also possible, and doing so can be done
in a more platform-independent way.  However, this method requires some care, as
it must be completed before many of the other libraries that ActivitySim depends
on are loaded.  This can be accomplished by setting the environment variables at
the very top of the Python script used to run ActivitySim, before any other imports:

    ```python
    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import activitysim
    ...
    ```

If you accidentally import `activitysim` or its dependencies before setting the
environment variables, the environment variables may not be respected.  Note that
certain auto-formatting tools may automatically reorder imports and group them all
at the top of a Python script; when using these tools it is important to make
sure that the `os.environ` setting commands stay in front of the `import` statments
other than `import os`.

### How Many Threads

Sometimes, using more than one thread per process can be beneficial.

## Sharrow

Significant performance improvements can be achieved by using the `sharrow` library,
although doing so requires certain limitations on model design, particularly
relating to what expressions are allowed in utility specifications.  Generally
any model specification can be accommodated in `sharrow` by re-writing problematic
expressions in a more `sharrow`-friendly way, or by moving them to a pre-processor.

For more details on setting up ActivitySim models to use `sharrow`, see the
[Using Sharrow](../dev-guide/using-sharrow.md) section in the Developer's Guide.


## Chunking

The default operation of ActivitySim is to attempt to run simulations in each
component for that component's entire pool of choosers in a single operation.
This allows for efficient use of vectorization to speed up computations, but can
also lead to memory issues if the pool of choosers is too large.  This is particularly
a problem in interaction-type models, where a large pool of choosers is faced
with a large set of alternatives.

ActivitySim includes the ability to "chunk" these model components into more
manageable sized groups of choosers, which can be processed one chunk at a time.
There is a small overhead associated with chunking, but if the total number of
chunks is relatively small, the overhead is usually outweighed by the benefits
in reduced memory usage.

Individual model components can be configured to use chunking explicitly by
setting the `explicit_chunk` configuration setting in the model component's
settings, when available. (Refer to each model component's documentation for
details on whether explicit chunking is available with that component.)  The
chunk setting can be set to an integer number of choosers to process in each
chunk, or to a fractional value to make chunks approximately that fraction of
the overall number of chooser (e.g. set to 0.25 to get four chunks).
