
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

* Use the :ref:`Pre-packaged Installer`
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
   + Converting string variables to pandas categoricals. ActivitySim releases *<placeholder for version number>* and higher have this capability.
   + Converting higher byte integer variables to lower byte integer variables (such as reducing ‘num tours’ from int64 to int8).
   + Converting higher byte float variables to lower bytes. ActivitySim releases X.X.X and higher have this capability as a switch and defaults to turning this feature off.

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

1. Using a :ref:`Pre-packaged Installer` (recommended for users who do not need to change the Python code)

2. Using a :ref:`Python the uv package and project manager` (recommended for users who need to change/customize the Python code)


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


Using *uv* package and project manager
______________________________________

This method is recommended for ActivitySim users who are familiar with Python and optionally wish to customize the Python code to run their models.
UV is a free open source cross-platform package and project manager that runs on
Windows, OS X, and Linux. It is 10-100x faster than pip itself, which is the standard Python package manager. The uv features include automatic 
environment management including installation and management of Python versions and dependency locking. The steps involved are described as follows:

1. Install *uv*. Instructions can be found `here <https://docs.astral.sh/uv/getting-started/installation/>`.

2. Clone the ActivitySim project using Git. (If Git is not installed, instructions can be found `here <https://git-scm.com/downloads>`.)

  git clone https://github.com/ActivitySim/activitysim.git
  cd activitysim

3. Optionally create the virtual environment. This is created automatically when running code in the next step, but manually syncing is an option too. This step creates a hidden folder within the current directory called `.venv` and operates the same way as Python's classic *venv*.

  uv sync

4. Run an ActivitySim command using the following. (This will automatically create a virtual environment from the lockfile, if it does not already exist.)

  uv run ...

For example, run the ActivitySim commandline using the following, which makes sure the code is run within the correct (locked) Python environment.

  uv run activitysim run -c configs -o output -d data

If you want to run ActivitySim from a directory different than where the code lives, use the `project` option to point *uv* to this project using relative paths:

  uv run --project relative/path/to/activitysim/code activitysim run -c configs -o output -d data


For more on *uv*, visit https://docs.astral.sh/uv/.


