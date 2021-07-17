
.. _benchmarking :

Benchmarking
------------

ActivitySim includes the ability to run performance benchmarks using a tool
called `airspeed velocity <https://asv.readthedocs.io/en/stable/>`__.

The benchmarking process is closely tied to ActivitySim's *git* repository,
so it is recommended that you use Git to clone the repository from GitHub.



Interim Directions
~~~~~~~~~~~~~~~~~~

Clone the repo, and setup a conda environment for benchmarking.

::

    conda create -n ASIM-BENCH git gh -c conda-forge --override-channels
    conda activate ASIM-BENCH
    gh auth login  # <--- (only needed if gh is not logged in)
    gh repo clone jpn--/activitysim  # FUTURE: use main repo
    cd activitysim
    git switch performance1  # FUTURE: use master branch
    conda env update --file=conda-environments/activitysim-dev.yml
    cd ..
    gh repo clone jpn--/activitysim_benchmarks  # FUTURE: use org repo
    cd activitysim_benchmarks

Define machine specs by running this command::

    activitysim benchmark machine

This will start an interactive questions and answer session to describe your
computer.  Don't be afraid, just answer the questions.  The tool may make
suggestions, but they are not always correct, so check them first and don't just
accept all.  For example, under "arch" it may suggest "AMD64", but for consistency
yo can change that to "x86_64", which is the same thing by a different name.

To run benchmarks on only the most recent commit in the main ActivitySim repo::

    activitysim benchmark run "HEAD^!" --verbose


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