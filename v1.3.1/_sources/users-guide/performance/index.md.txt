# Runtime Performance

Achieving "good" runtime performance in ActivitySim depends on several factors,
including the model size (number of zones, number of households), model complexity,
(eg. number of components, length and complexity of utility specifications), and
hardware capabilities (amount of RAM, number of CPU cores, amount of on-chip cache,
speed of storage devices).

Much of the work in making ActivitySim run efficiently sits on the shoulders of
model developers, who design models and determine the level of detail and complexity
of the model components. However, for any given fully developed ActivitySim model,
there are several techniques that users can employ to improve runtime performance,
and to tune that performance to the specific hardware on which the model is being run.
These techniques are the focus of this section.

## Performance Tuning Topics
```{eval-rst}
.. toctree::
   :maxdepth: 2

   Multiprocessing <multiprocessing>
   Multithreading <multithreading>
   Chunking to Reduce Peak Memory Usage <chunking>
   Compiling with Sharrow <sharrow>
   Skim Data Format <skim-data-format>
```

## Checklist for Performance Tuning

Here is a checklist of common steps to take when tuning the performance of an
ActivitySim model:

1. Precompile (if using sharrow)

    Run the entire model with a modest number of households, to let sharrow
    pre-compile relevant utility specifications.  This works fine if the "sharrow"
    setting is set to any of `true`, `require`, or `test`, but to give confidence that
    the model specification calculations are correct it is convenient to use `test` mode.

    ```{important}
    If errors are encountered when using `test` mode, they should be addressed before
    the model is used for analysis, as they may indicate that either sharrow or the
    legacy evaluator is not correctly processing the mathematical expressions in the
    utility specifications.  See [Using Sharrow](../../dev-guide/using-sharrow.md#troubleshooting)
    in the Developer's Guide for more information on how to troubleshoot errors.
    ```

    The pre-compile run needs be single-process, to avoid compiler race conditions between
    various subprocesses. The exact number of households to use is not particularly important,
    but it should be large enough to trigger all relevant model components (i.e., we need
    to be sure that there are worker, students, students getting escorted, all various
    household sizes, etc.)  A few thousand households is usually sufficient.

    Recommended settings for the precompile run include:

    ```yaml
    sharrow: test
    multiprocess: False
    households_sample_size: 10000
    ```

2. Memory Profiling

    Run the model single-process with a small sample size (10% or so) and profile
    the memory usage. This will give you an idea of how much memory the model will
    need to run the 100% household sample.  If the model is projected to run out of
    memory when run on a 100% sample, you may need to configure chunking (see below)
    or reduce the sample size if feasible for the analysis.

    Recommended settings for the memory profile run include:

    ```yaml
    sharrow: require
    multiprocess: False
    households_sample_size: 0.1
    memory_profile: True
    chunk_training_mode: disabled
    ```

3. Configure Chunking (if needed)

    If the model is projected to run out of memory when run on a 100% sample, you
    may need to configure chunking.  See [Explicit Chunking](chunking.md#explicit-chunking)
    for recommendations on how to configure chunking for reliable model operation.

    ```{note}
    The use of the explicit chunking algorithms is activated by a top level
    setting of `chunk_training_mode: explicit`, but actually using explicit chunking
    also requires configuration settings on each component where explicit chunking is
    used.
    ```

4. Experiment with Multiprocessing

    Run the model with multiprocessing enabled, and experiment with the number of
    cores to find the best performance.  The optimal number of cores will depend on
    the model size and complexity, and the hardware on which the model is being run.
    Typically, ActivitySim models seem to perform best with about 10 processes, which
    is usually a good starting point for experimentation. See [Multiprocessing](multiprocessing.md)
    for more information on how to experiment with multiprocessing.
