
.. _benchmarking :

Benchmarking
------------

ActivitySim includes the ability to run performance benchmarks using a tool
called `airspeed velocity <https://asv.readthedocs.io/en/stable/>`__.

The benchmarking process is closely tied to ActivitySim's *git* repository,
so it is recommended that you use Git to clone the repository from GitHub.


Benchmarking Setup
~~~~~~~~~~~~~~~~~~

The first step in running benchmarks is to have a conda environment for
benchmarking, as well as a local clone of the main ActivitySim repository,
plus one of the `activitysim_benchmarks` repository.

If this isn't already set up on your performance benchmarking machine, you can
do so by following these steps::

    conda create -n ASIM-BENCH git gh -c conda-forge --override-channels
    conda activate ASIM-BENCH
    gh auth login  # <--- (only needed if gh is not logged in)
    gh repo clone jpn--/activitysim  # FUTURE: use main repo
    cd activitysim
    git switch performance2  # FUTURE: use develop branch
    conda env update --file=conda-environments/activitysim-dev.yml
    cd ..
    gh repo clone jpn--/activitysim_benchmarks  # FUTURE: use org repo
    cd activitysim_benchmarks

If you want to submit your benchmarking results to the common database of
community results, you should also fork the `activitysim_benchmarks` repository::

    gh repo fork --remote=true

Next, we'll want to declare the specs of our benchmarking machine.  Some of
these can be determined quasi-automatically, but we want to confirm the specs
we'll use as they are written with our benchmark results into the database.
Define machine specs by running this command::

    activitysim benchmark machine

This will start an interactive questions and answer session to describe your
computer.  Don't be afraid, just answer the questions.  The tool may make
suggestions, but they are not always correct, so check them first and don't just
accept all.  For example, under "arch" it may suggest "AMD64", but for consistency
you can change that to "x86_64", which is the same thing by a different name.

Running Benchmarks
~~~~~~~~~~~~~~~~~~

ActivitySim automates the process of running many benchmarks. It can also easily
aggregate benchmark results across many different machines, as long as the
benchmarks are all run in the same (relative) place. So before running benchmarks,
change your working directory (at the command prompt) into the top directory of
the `activitysim_benchmarks` repository, if you're not already there.

To run all of the benchmarks on the most recent commit in the main ActivitySim repo::

    activitysim benchmark run "HEAD^!" --verbose --append-samples

In this command, `"HEAD^!"` instructs the benchmarking to run only on the "HEAD"
(most recent) commit in the repository, `--verbose` will generate a stream of output
to the console (otherwise, the process may appear stalled as it can run a very long
time without writing anything visible) and `--append-samples` will record the runtime
of each attempt on each model component, instead of merely recording some statistical
measurements.

To run only benchmarks from a certain example, we can
use the `--bench` argument, which allows us to write a "regular expression" that
filters the benchmarks actually executed.  This is handy if you are interested in
benchmarking a particular model or component, as running *all* the benchmarks can
take a very long time.  For example, to run only the SANDAG 1-Zone benchmarks,
run::

    activitysim benchmark run HEAD^! --verbose --append-samples --bench sandag1



Threading Limits
~~~~~~~~~~~~~~~~

When you run benchmarking using the `activitysim benchmark` command, the
following environment variable are set automatically before benchmarking begins::

    MKL_NUM_THREADS = 1
    OMP_NUM_THREADS = 1
    OPENBLAS_NUM_THREADS = 1
    NUMBA_NUM_THREADS = 1
    VECLIB_MAXIMUM_THREADS = 1
    NUMEXPR_NUM_THREADS = 1

This ensures that all benchmarking operations run processes in single-threaded
mode.  This still allows ActivitySim itself to spin up multiple processes if the
item being timed is a multiprocess benchmark.

Submitting Benchmarks
~~~~~~~~~~~~~~~~~~~~~

One of the useful features of the airspeed velocity benchmarking engine is the
opportunity to compare performance benchmarks across different machines. The
ActivitySim community is interested in aggregating such results from a number
of participants, so once you have successfully run a set of benchmarks, you
should submit those results to our repository.

To do so, assuming you have run the benchmark tool inside the `activitysim_benchmarks`
repository as noted above, you simply need to commit any new or changed files
in the `activitysim_benchmarks/results` directory.  You can then open a pull request
against the community `activitysim_benchmarks` to submit those results.
