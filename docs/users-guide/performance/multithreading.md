# Multithreading

Some parts of ActivitySim are also multithreaded, using various different
multithreading technologies. This includes large matrix multiplication operations
used in certain utility calculations (multithreaded by the MKL library where
available) as well as computations accelerated using the sharrow library (which
uses numba to parallelize certain operations), as well as potentially other
parallelization subroutines in other libraries upon which ActivitySim depends.

In general, it is desirable to fully or partially *disable* multithreading in
ActivitySim when using multiprocessing, as the two technologies can interfere
with each other.

When ActivitySim is run from the command line, using either `activitysim run` or
`python -m activitysim run`, by default multithreading is disabled.  This can
be overridden by using the `--fast` command line flag, which will not disable
multithreading, leaving whatever existing environment settings for threading
unchanged.  When running ActivitySim from within a Python script,
or by calling it from a Jupyter notebook, it is up to the user to enable or
disable multithreading if desired, as these settings need to be made before
libraries that use multithreading are loaded.

Disabling multithreading can be done two ways, either by setting environment
variables before starting a Python instance that will run ActivitySim, or by
setting certain environment variables within Python before loading ActivitySim
or its dependencies (especially numba).  Once certain libraries (e.g., numba)
are loaded, it is too late to set global process limits on multithreading.


## Disabling Multithreading via Environment Variables

Setting environment variables before starting a Python instance is the most
reliable way to disable multithreading in ActivitySim, as it ensures that the
settings are made before any Python libraries are loaded. The environment
variables that need to be set are:

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

The downside of this is that the exact commands needed are platform and
shell-specific (i.e. the commands for Powershell, Windows `cmd.exe`, and Linux's
bash are all different).

### Windows Command Line

```{cmd}
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set NUMBA_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
```

### Linux / MacOS (bash / zsh)

```{bash}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Disabling Multithreading via Python

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
certain auto-formatting tools (e.g. isort) may automatically reorder imports and group
them all at the top of a Python script; when using these tools it is important to make
sure that the `os.environ` setting commands stay in front of the `import` statments
other than `import os`.

## How Many Threads

Sometimes, using more than one thread per process can be beneficial.  In the discussion
above, threading is effectively disabled by setting the number of threads to 1.  However,
if you have a machine with many cores, you may want to experiment with using more than
one thread per process.  Depending on your particular model and hardware, you may find
that using multiple threads per process can improve performance.  However, it is likely
that the optimal number of threads per process will be less than the number of CPU cores
per process, so it is recommended to start with a small number of threads per process
and work up from there to see if gains are possible.
