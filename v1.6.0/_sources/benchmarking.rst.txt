
.. _benchmarking :

Benchmarking
------------

A key focus of the ActivitySim project is *performance*.  It's not enough
to build a new modeling platform that's mathematically sound and simulates
travel behavior as expected.  It's also required that it do so quickly.
It's not too hard to run performance tests manually on individual models, and
doing so after making changes that are expected to *improve* performance is
typical. But monitoring performance regularly and automatically can help ensure
that new features do not introduce unexpected performance regressions (i.e.
models run slower than before). Developing an extensive set of automatic
performance benchmarks can streamline the former problem and solve the latter.

ActivitySim includes the ability to run performance benchmarks using a tool
called `airspeed velocity <https://asv.readthedocs.io/en/stable/>`__.

The benchmarking process is closely tied to ActivitySim's *git* repository,
so it is recommended that you use Git to clone the repository from GitHub.


Benchmarking Setup
~~~~~~~~~~~~~~~~~~

The first step in running benchmarks is to have a conda environment for
benchmarking, as well as a local clone of the main ActivitySim repository,
plus one of the ``asim-benchmarks`` repository. If you plan to submit your
benchmarking results to the common repository of results, you'll want to
also make sure that your ``asim-benchmarks`` repository is using a fork of the
common repository to which you have write-access.

If this isn't already set up on your performance benchmarking machine, you can
do all of this setup by following these steps::

    conda create -n ASIM-BENCH mamba git gh -c conda-forge --override-channels
    conda activate ASIM-BENCH
    gh auth login  # <--- (only needed if gh is not logged in)
    gh repo clone ActivitySim/activitysim          # TEMPORARY: use jpn--/activitysim
    cd activitysim
    git switch develop                             # TEMPORARY: use performance1 branch
    mamba env update --file=conda-environments/activitysim-dev.yml
    cd ..
    gh repo fork ActivitySim/asim-benchmarks --remote      
    cd asim-benchmarks
    python initialize-hooks.py

For non-Windows users, you can then actually activate the pre-commit hooks like
this::

    pre-commit install     # macOS/Linux only, do not run this line on Windows

Windows users should not attempt to use installed pre-commit hooks with conda
(see note below).  Instead, you must manually ``pre-commit run`` inside the correct
conda environment before committing.

If this environment is set up but it's been a while since you last used it,
consider updating the environment like this::

    conda activate ASIM-BENCH
    cd activitysim
    git switch develop                             # TEMPORARY: use performance1 branch
    mamba env update --file=conda-environments/activitysim-dev.yml
    cd ..
    cd asim-benchmarks
    git pull

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
accumulate and analyze benchmark results across many different machines, as long as the
benchmarks are all run in the same (relative) place. So before running benchmarks,
change your working directory (at the command prompt) into the top directory of
the `asim-benchmarks` repository, if you're not already there.

To run all of the benchmarks on the most recent commit in the main ActivitySim repo::

    activitysim benchmark latest

.. important::

    The benchmarks do not currently use ActivitySim's dynamic chunking features,
    as these require manual configuration and training on a per-machine basis
    to ensure good performance.

    Running the complete suite of benchmarks currently includes downloading and
    running full-region model data for several different SANDAG zone systems.
    Ideally you should have at least 50 GB of free disk space and 120 GB of RAM
    to attempt this process on any given machine.  For a smaller machine, consider
    benchmarking only the "test" sized examples, by adding `--bench sandag.example`
    to this command, as discussed below.

This will run the benchmarks only on the "HEAD" commit of the main activitysim git
repository.  To run on some other historical commit[s] from the git history, you can
specify an individual commit or a range, in the same way you would do so for the
`git log` command. For example, to run benchmarks on the commits to develop since
it was branched off main, run::

    activitysim benchmark run main..develop

or to run only on the latest commit in develop, run::

    activitysim benchmark run "develop^!"

Note that the literal quotation marks are necessary on Windows, as the carat character
preceding the exclamation mark is otherwise interpreted as an escape character.
In most other shells (e.g. on Linux or macOS) the literal quotation marks are unnecessary.

To run only benchmarks from a certain example, we can
use the `--bench` argument, which allows us to write a "regular expression" that
filters the benchmarks actually executed.  This is handy if you are interested in
benchmarking a particular model or component, as running *all* the benchmarks can
take a very long time, and the larger benchmarks (e.g. on the full SANDAG model)
will need a lot of disk space and RAM.  For example, to run only the mandatory
tour frequency benchmark for the SANDAG 1-Zone example-sized system, run::

    activitysim benchmark latest --bench sandag1example.time_mandatory_tour_frequency

The "." character here means a literal dot, but since this is a regex expression,
it is also a single-character wildcard.  Thus, you can run all the example-sized
SANDAG benchmarks with::

    activitysim benchmark latest --bench sandag.example

You can also repeat the `--bench` argument to give multiple different expressions.
So, you can run just the 1- and 2-zone examples, without the 3-zone example::

    activitysim benchmark latest --bench sandag1example --bench sandag2example

If you want to run several different benchmarking commmands together, for example
to run a custom curated subset of interesting benchmarks, the benchmark tool also
includes a `batch` mode.  You can assemble the various commands you would run
(i.e. everything you would type on the command line after "activitysim benchmark")
into a text file, and then point to that file using the `batch` command::

    activitysim benchmark batch my_interesting_benchmarks.txt


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

To do so, assuming you have run the benchmark tool inside the ``asim-benchmarks``
repository as noted above, you simply need to commit any new or changed files
in the ``asim-benchmarks/results`` directory.  You can then open a pull request
against the community ``asim-benchmarks`` to submit those results.

Assuming you are in (or first ``cd`` into) the ``asim-benchmarks`` directory, You can
do this from the command line using the following steps::

    git add results
    pre-commit run    # required on Windows only, see note
    git commit -m "adding benchmark results"
    git push
    gh pr create

.. note::

    On Windows, the process for automatically running pre-commit hooks when
    making a Git a commit is not compatible with conda, see
    `here <https://github.com/pre-commit/pre-commit/issues/1329>`. This will
    probably never be fixed, as the developers of pre-commit and conda each
    feel that the "bug" is in the other library.  So, manually running the
    pre-commit step is required.

Users may find it simpler to skip the last step on the command line, and simply
visit their fork on GitHub.com to use the web interface to open a pull request.

Publishing to Github Pages
~~~~~~~~~~~~~~~~~~~~~~~~~~

Publishing the standard airspeed velocity content to GitHub pages is a built-in
feature of the command line tool, available to users who have write-access to the
asim-benchmarks GitHub repository.  Be sure you have all the relevant branches
tracked locally (especially main and develop) and then run::

    activitysim benchmark gh-pages


Profiling
~~~~~~~~~

The benchmarking tool can also be used for profiling, which allows a developer to
inspect the timings for various commands *inside* a particular benchmark. This is
most conveniently accomplished using the ``snakeviz`` tool, which should be installed
in the developer tools environment (``conda install snakeviz -c conda-forge``).
Then, the developer needs to run two commands to compute and view the component
profile.

To create a profile record when benchmarking, add the ``--profile`` option when
running the benchmarks.  For example, to create profile records for the SANDAG
example-sized model's non-mandatory tour scheduling component across all three
zone systems, run::

    activitysim benchmark latest --bench sandag.example.non_mandatory_tour_scheduling --profile

This command will save the profiling data directly into the json file that stores
the benchmark timings.  This is a lot of extra data, so it's not advised to
save profiling data for every benchmark, but only for benchmarks of particular
interest.

Once this data has been saved, you can access it using the ``snakeviz`` tool.  This
visualization requires pointing to a specific profiled benchmark in a specific
json result file.  For example::

    activitysim benchmark snakeviz results/LUMBERJACK/241ddb64-env-c87ac846ee78e51351a06682de5adcb5.json sandag3example.non_mandatory_tour_scheduling.time_component

On running this command, a web browser should pop open to display the snakeviz
interface.

Writing New Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

New benchmarks for other model examples can be added to
``activitysim/benchmarking/benchmarks``. A basic template structure has been used,
so that it should be relatively straight-forward to implement component-level
single thread benchmarks for any model that is available using the
``activitysim create`` tool.

A basic framework for multi-processing benchmarks has been implemented and is
demonstrated in the ``mtc1mp4`` benchmark file. However, work remains to write
a stable process to execute chunking training for each machine prior to running
the production-version benchmarks that will be meaningful for users.

Running Benchmarks for Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete set of performance benchmarks is too large to include in ActivitySim's
automatic continuous integration (CI) testing, both by compute time and by memory usage.
However, it is valuable to run these tests once against the final version of
each PR before merging into the ``develop`` branch, to ensure there are no
unexpected performance regressions. The airspeed velocity tools include a special
CI mode, which runs the same benchmarks on the same machine with the same settings,
giving developers a fair shot at a strict apples-to-apples comparison of performance.

This mode can be activated to check the performance of code on a git branch called
``my-new-feature-branch``, and compare against the ``develop`` branch like this::

    activitysim benchmark continuous develop my-new-feature-branch

Unlike other tests for mathematical correctness, it is not always necessary that
new PR's must "pass" this testing, as new features or capabilities may justify a
performance degradation.  But developers should always run these tests on new PR's
so that the community is aware of the trade offs (if any) and can take steps to
mitigate problems promptly if desired.

