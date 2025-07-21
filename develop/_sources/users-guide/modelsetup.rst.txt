
Model Setup
===========

This page describes the system requirements and installation procedures to set up ActivitySim.

.. note::
   ActivitySim is under active development



Quick Reference
---------------
This section briefly describes the quickest way to install and start running ActivitySim. This section
assumes the user is more experienced in running travel demand models and proficient in Python, but has not
used ActivitySim or has not used recent versions of ActivitySim. More detailed instructions for installing
and running ActivitySim are also available in this Users Guide.

* Use the :ref:`pre-packaged installer`
* :ref:`Run the Primary Example`
* Placeholder (Edit model input files, configs, as needed)

System Requirements
-------------------

This section highlights the software requirements for any implementation, as well as hardware recommendations.

Hardware
________

The computing hardware required to run a model implemented in the ActivitySim framework generally depends on:

* The number of households to be simulated for disaggregate model steps
   * In addition to the total number of households in the model region, runtime and hardware requirements can be reduced by sampling a subset of the households. The user can adjust the sampling rate for a particular run (see Settings.yaml).
* The number of model zones (for each zone system) for aggregate model steps
* The number and size of network skims by mode and time-of-day
* The number of zone systems, see :ref:`Zone system`
* The desired runtimes

ActivitySim framework models use a significant amount of RAM since they store data in-memory to reduce
data access time in order to minimize runtime.

For example, the SEMCOG ABM, a model that follows a 2-Zone system runs on a windows machine, with the minimum and recommended system specification as follows:

* Minimum Specification:
   + Operating System: 64-bit Windows 7, 64-bit Windows 8 (8.1) or 64-bit Windows 10
   + Processor: 8-core CPU processor
   + Memory: 128 GB RAM
   + Disk space: 150 GB

* Recommended Specification:
   + Operating System: 64-bit Windows 7, 64-bit Windows 8 (8.1), 64-bit Windows 10, or 64-bit Windows 11
   + Processor: Intel CPU Xeon Gold / AMD CPU Threadripper Pro (12+ cores)
   + Memory: 256 GB RAM
   + Disk space: 150 GB

As another example, the prototype MTC example model - which has 2.7 million households, 7.5 million people, 1475 zones, 826 network skims - has a runtime between one hour and one day depending on the amount of RAM and number of processors allocated.

ActivitySim has features that makes it possible to customize model runs or improve model runtimes based on the available hardware resources and requirements. A few ways to do this are listed below:

* :ref:`Chunking <chunking_ways_to_run>` allows the user to run eligible steps in parallel. This can be turned on/off.
* :ref:`Multiprocessing <multi_proc_ways_to_run>` allows the user to segment processing into discrete sets of data. This will increase the runtime but allow for lower RAM requirements. This feature can also be turned on/off.
* :ref:`Sharrow <sharrow_ways_to_run>` is a Python library designed to decrease run-time for ActivitySim models by creating an optimized compiled version of the model. This can also be turned on/off.
* :ref:`Tracing <tracing_ways_to_run>` allows the user to access information throughout the model run for a specified number of households/persons/zones. Enabling this feature will increase run-time and memory usage. It is recommended that this feature be turned off for typical model application.
* Optimization of data types including:
   + Converting string variables to pandas categoricals. ActivitySim releases 1.3.0 and higher have this capability.
   + Converting higher byte integer variables to lower byte integer variables (such as reducing ‘num tours’ from int64 to int8). ActivitySim releases 1.3.0 and higher have this capability as a switch and defaults to turning this feature off.
   + Converting higher byte float variables to lower bytes. ActivitySim releases 1.3.0 and higher have this capability as a switch and defaults to turning this feature off.

Steps for enabling/disabling these options are included in the :ref:`Advanced Configuration` sub-section, under :ref:`Ways to Run the Model` page of this Users’ Guide.


.. note::
   In general, more CPU cores and RAM will result in faster run times.
   ActivitySim has also been run in the cloud, on both Windows and Linux using
   `Microsoft Azure <https://azure.microsoft.com/en-us/>`__.  Example configurations,
   scripts, and runtimes are in the <todo: cross-ref> ``other_resources\example_azure`` folder.


Software
________

Activitysim is implemented in the Python programming language. It uses several open source Python packages such as pandas, numpy, pytables, openmatrix etc.


Installing ActivitySim
----------------------

There are two recommended ways to install ActivitySim:

1. Using a :ref:`pre-packaged installer`

2. Using the :ref:`UV Package and Project Manager`

The first is recommended for users who do not need to change the Python code and are on Windows, 
and the second is recommended for users who need to change/customize the Python code.


Pre-packaged Installer
______________________

Beginning with version 1.2, ActivitySim is now available for Windows via a
pre-packaged installer.  This installer provides everything you need to run
ActivitySim, including Python, all the necessary supporting packages, and
ActivitySim itself.  You should only choose this installation process if you
plan to use ActivitySim but you don't need or want to do other Python
development.  Note this installer is provided as an "executable" which (of course)
installs a variety of things on your system, and it is quite likely to be flagged by
Windows, anti-virus, or institutional IT policies as "unusual" software, which
may require special treatment to actually install and use.

Download the installer from GitHub `here <https://github.com/ActivitySim/activitysim/releases/download/v1.3.1/Activitysim-1.3.1-Windows-x86_64.exe>`_.
It is strongly recommended to choose the option to install "for me only", as this
should not require administrator privileges on your machine.  Pay attention
to the *complete path* of the installation location. You will need to know
that path to run ActivitySim in the future, as the installer does not modify
your "PATH" and the location of the `ActivitySim.exe` command line tool will not
be available without knowing the path to where the install has happened.

Once the install is complete, ActivitySim can be run directly from any command
prompt by running `<install_location>/Scripts/ActivitySim.exe`.


UV Package and Project Manager
______________________________________

This method is recommended for ActivitySim users who are familiar with 
Python and optionally wish to customize the Python code to run their models.
UV is a free open source cross-platform package and project manager that runs 
on Windows, OS X, and Linux. It is 10-100x faster than conda, and pip itself, which is 
the standard Python package manager. The *uv* features include automatic 
environment management including installation and management of Python 
versions and dependency locking. 

.. note::
  There are two options for using *uv* to install ActivitySim. 

The first is to use *uv* to install an official ActivitySim release from the Python Package Index (PyPI). 
The second is to use *uv* to install ActivitySim from a local directory, which should be the cloned ActivitySim repository.

.. note::
  The first *uv* option is recommended for users who want to install ActivitySim from an official release and do not wish to change the Python code. 
  However, they may end up using different deep dependencies than those tested by the developers. 
  The second *uv* option is recommended for users who may want to customize the Python code, and who want to run ActivitySim 
  exactly as it was tested by the developers using the dependency lockfile which results in the exact same deep dependencies.

The steps involved are described as follows.

Option 1: Install ActivitySim from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Note: This step may fail at the moment because the ActivitySim version available on PyPI has dependency conflicts.
This step should work when ActivitySim release *<placeholder for version number>* which is built with *uv* is available on PyPI.
In the meantime, use Option 2 below to install ActivitySim from the lockfile.)

1. Install *uv*. Instructions can be found 
`here <https://docs.astral.sh/uv/getting-started/installation/>`_.

2. Create a new project and virtual environment to work from and add ActivitySim. (Warning: This approach is quickest
for getting started but does not rely on the lockfile to install dependencies so you may
end up with different versions. If you want to use the lockfile, see Option 2 below.)

Open a terminal, such as Command Prompt (note: not Anaconda Prompt), and run the following commands.

::

  mkdir asim_project
  cd asim_project
  echo 3.10 > .python-version # This sets the Python version to 3.10, which is currently used for ActivitySim development.
  uv init
  uv add activitysim

*uv* will create a new virtual environment within the `asim_project` project folder 
and install ActivitySim and its dependencies. The virtual environment is a hidden folder 
within the `asim_project` directory called `.venv` and operates the same way as Python's classic *venv*.

3. Run an ActivitySim command using the following.

::

  uv run ...

For example, run the ActivitySim commandline using the following. 
More information about the commandline interface is available in 
the :ref:`Ways to Run the Model` section.

::

  uv run activitysim run -c configs -o output -d data


Run ActivitySim from a Different Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to run ActivitySim from a directory different than where the code lives, 
use the `project` option to point *uv* to this project using relative paths:

::

  uv run --project relative/path/to/asim_project activitysim run -c configs -o output -d data


You could also activate the virtual environment created by *uv* and run ActivitySim from any directory. 

::

  .venv\Scripts\activate

With this command, you have activated the virtual environment created by *uv* and can run ActivitySim commands from any directory.

For more on *uv*, visit https://docs.astral.sh/uv/.

Option 2: Install ActivitySim from the lockfile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install dependencies from the lockfile and run ActivitySim exactly how
its developers tested it, after installing *uv* clone the code repository
and then run code.

1. Install *uv*. Instructions can be found 
`here <https://docs.astral.sh/uv/getting-started/installation/>`_. (Skip
if already installed above. It only needs to be installed once per machine.)

.. note::
  If you already have *uv* installed from an older project and you encounter errors
  such as 
  ::

    error: Failed to parse uv.lock... missing field version...
  
  later in the process, you may need to update *uv* to the latest version by reinstalling it via the official
  installation script: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer.
  You can check the version of *uv* you have installed by running
  ::

    uv --version

1. Clone the ActivitySim project using Git. (If Git is not installed, 
instructions can be found `here <https://git-scm.com/downloads>`_.)

::

  git clone https://github.com/ActivitySim/activitysim.git  
  cd activitysim

3. Optionally create the virtual environment. This is created automatically 
when running code in the next step, but manually syncing is an option too. 
This step creates a hidden folder within the current directory called 
`.venv` and operates the same way as Python's classic *venv*. (If you 
want to install the project in a non-editable mode so that users on
your machine cannot accidentally change the source code, use the 
`--no-editable` option.) 

::

  uv sync --no-editable

4. Run an ActivitySim command using the following. (This will automatically 
create a virtual environment from the lockfile, if it does not already 
exist.)

::

  uv run ...


It is worth pointing out that by default, *uv* installs projects in 
editable mode, such that changes to the source code are immediately reflected 
in the environment. `uv sync` and `uv run` both accept a `--no-editable` 
flag, which instructs *uv* to install the project in non-editable mode, 
removing any dependency on the source code.

Also, `uv run` automatically installs the dependencies listed in `pyproject.toml`
under `dependencies` under `[project]`, and it also installs those listed 
under `dev` under `[dependency-groups]` (not `github-action`). If you want to 
skip the dependency groups entirely with a *uv* install (and only install those
that would install via `pip` from 'pypi`), use the `--no-default-groups` flag 
with `uv sync`.

Run ActivitySim from a Different Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to run ActivitySim from a directory different than where the code lives, 
use the `project` option to point *uv* to this project using relative paths:

::

  uv run --project relative/path/to/asim_project activitysim run -c configs -o output -d data


You could also activate the virtual environment created by *uv* and run ActivitySim from any directory. 

::

  .venv\Scripts\activate

With this command, you have activated the virtual environment created by *uv* and can run ActivitySim commands from any directory.

For more on *uv*, visit https://docs.astral.sh/uv/.