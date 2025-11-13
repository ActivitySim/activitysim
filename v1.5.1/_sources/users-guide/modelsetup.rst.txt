
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

The first is recommended for users who are new to Python and use Windows, do not actively create and manage Python virtual environments, 
and do not need to change the ActivitySim code. The second is recommended for users who actively create and manage Python virtual environments, 
and/or want to change/customize the ActivitySim code.


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
Python, create and manage ActivitySim Python virtual environments, and optionally wish to customize ActivitySim code to run their models.
UV is a free open source cross-platform package and project manager that runs 
on Windows, OS X, and Linux. It is 10-100x faster than conda, and pip itself, which is 
the standard Python package manager. The *uv* features include automatic 
environment management including installation and management of Python 
versions and dependency locking. 

Install UV
^^^^^^^^^^^^^^

We recommend installing UV as an independent tool on your machine, separate from any existing package managers you may have such as conda or pip.

For Windows users, run the following command in PowerShell to install *uv*. It does not require administrator privileges and installs *uv* for the current user only.
By default, uv is installed to ``~/.local/bin`` directory. Usually, this is ``C:/Users/<username>/.local/bin``.

::

  # Run the installer. Please review the printed message after installation.
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

  # Add uv to PATH
  $env:PATH = "$env:USERPROFILE\.local\bin;$env:Path"

If an agency wants to install *uv* globally for all users on Windows, run PowerShell as Administrator and run the following command.

::

  # Run the installer with a custom install directory (e.g., C:\shared\uv) that is accessible to all users
  powershell -ExecutionPolicy ByPass -c {$env:UV_INSTALL_DIR = "C:\shared\uv";irm https://astral.sh/uv/install.ps1 | iex}

  # Add uv to PATH for all users (requires administrator privileges)
  [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\shared\uv", [EnvironmentVariableTarget]::Machine)

For more instructions on installing *uv* on Windows, MacOS, or Linux, please visit https://docs.astral.sh/uv/getting-started/installation/.

To verify that *uv* is installed correctly, open a new Command Prompt (not Anaconda Prompt) and run the following command.

::

  uv --version

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



Install ActivitySim with UV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two options to install ActivitySim using *uv*.

The first is to use *uv* to install an official ActivitySim release from the Python Package Index (PyPI).
The second is to use *uv* to install ActivitySim from the source code repository and use the dependency lockfile.

.. note::
  The first option (:ref:`Option 1: From PyPI`) is the quickest way to install ActivitySim from an official release and is recommended for users who do not wish to change the Python code. 
  However, they may end up using different deep dependencies than those tested by the developers. 
  The second option (:ref:`Option 2: From Source with Lockfile`) is recommended for users who may want to customize the Python code, and/or who want to run ActivitySim 
  exactly as it was tested by the developers using the dependency lockfile which results in the exact same deep dependencies.

The steps involved are described as follows.

Option 1: From PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will use *uv* to create a project and virtual environment to work from and add ActivitySim.

Open Command Prompt (not Anaconda Prompt), and run the following commands.

::

  # create a new project directory and cd into it
  mkdir asim_project
  cd asim_project

  # initialize a virtual environment
  # This sets the Python version to 3.10, which is currently fully tested for ActivitySim development
  uv init --python 3.10 

  # add ActivitySim package from the latest release on PyPI
  uv add activitysim

*uv* will create a new virtual environment within the ``asim_project`` project folder 
and install ActivitySim and its dependencies. The virtual environment is a hidden folder 
within the ``asim_project`` directory called ``.venv`` and operates the same way as Python's classic *venv*. You will notice 
two new files created in the ``asim_project`` directory: ``pyproject.toml`` and ``uv.lock``. These files
are automatically created, updated, and used by *uv* to manage your ``asim_project`` project and its dependencies. 
You can share these files with others to recreate the same environment for your ``asim_project`` project. For more guidance on sharing your working environment, 
see the Common Q&A :ref:`How to share my working environment with others?` section below.

By running the command ``uv add activitysim``, you install the official release of ActivitySim from PyPI and its direct dependencies 
listed in ActivitySim's ``pyproject.toml`` file. This approach is the quickest
for getting started but it does not rely on ActivitySim's own lockfile to install deep dependencies so you may
end up with different versions of deep dependencies than those tested by ActivitySim developers. 
If you want to ensure exact versions of ActivitySim's deep dependencies, you should install ActivitySim using Option 2: From Source with Lockfile.

Option 2: From Source with Lockfile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install dependencies from the lockfile and run ActivitySim exactly how
its developers tested it, after installing *uv*, open Command Prompt, clone the ActivitySim project using Git. (If Git is not installed, 
instructions can be found `here <https://git-scm.com/downloads>`_.)

::

  git clone https://github.com/ActivitySim/activitysim.git  
  cd activitysim

Run the ``uv sync --locked`` command to create a virtual environment using the lockfile. It will initialize a virtual environment within the ``activitysim`` directory
and install ActivitySim and all its dependencies exactly as specified in the ``uv.lock`` file. 
The virtual environment is a hidden folder within the current directory called 
``.venv`` and operates the same way as Python's classic *venv*.

::

  uv sync --locked
  # or uv sync --locked --no-editable

It is worth pointing out that by default, *uv* installs projects in 
editable mode, such that changes to the source code are immediately reflected 
in the environment. ``uv sync`` accepts a ``--no-editable`` 
flag, which instructs *uv* to install the project in non-editable mode, 
removing any dependency on the source code.

Also, ``uv sync`` automatically installs the dependencies listed in ``pyproject.toml``
under ``dependencies`` under ``[project]``, and it also installs those listed 
under ``dev`` under ``[dependency-groups]``. If you want to
skip the dependency groups entirely with a *uv* install (and only install those
that would install via ``pip`` from ``pypi``), use the ``--no-default-groups`` flag 
with ``uv sync``.


Which Option Should I Use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------------------------------------------------+-----------+---------------------------+
| If I want to ...                                                               | From PyPI | From Source with Lockfile |
+================================================================================+===========+===========================+
| Install an official release of ActivitySim.                                    | Yes       |                           |
+--------------------------------------------------------------------------------+-----------+---------------------------+
| Install a development version of ActivitySim.                                  |           | Yes                       |
+--------------------------------------------------------------------------------+-----------+---------------------------+
| Install ActivitySim quickly to run models without changing the code.           | Yes       |                           |
+--------------------------------------------------------------------------------+-----------+---------------------------+
| Do ActivitySim code development.                                               |           | Yes                       |
+--------------------------------------------------------------------------------+-----------+---------------------------+
| Run ActivitySim with deep dependencies exactly as tested by the developers.    |           | Yes                       |
+--------------------------------------------------------------------------------+-----------+---------------------------+


Run ActivitySim with UV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the virtual environment created by *uv*. This option is similar to using Python's classic venv or Conda env.

::

  # cd into the project directory if not already there
  ## if you used the From PyPI option
  cd asim_project
  ## if you used the From Source with Lockfile option
  cd activitysim

  # Activate the virtual environment
  .venv\Scripts\activate

Once the virtual environment is activated, you can run ActivitySim commands directly using the ``activitysim`` command.
For example, run the ActivitySim commandline using the following. More information about the commandline interface is available in 
the :ref:`Ways to Run the Model` section.

::

  activitysim run -c configs -o output -d data

Alternatively, you can run ActivitySim commands directly using *uv* without activating the virtual environment.

::

  uv run activitysim run -c configs -o output -d data

Common Q&A
^^^^^^^^^^^^^^^^
My travel demand model requires additional Python packages not included with ActivitySim. How do I add them?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can add additional packages to your *uv* project by using the ``uv add`` command. For example, to add the ``geopandas`` package, 
run the following command within your existing *uv* project directory.

::

  # cd into your project directory
  cd asim_project

  # Add geopandas package
  uv add geopandas

This will add the package to your virtual environment and update the ``pyproject.toml`` and the ``uv.lock`` file to include the new package and its dependencies.

If you envision having a version of Python packages that is different from the one used by ActivitySim, e.g., you need pandas 1.x for visualization (for some reason), 
we recommend creating a separate *uv* project for your custom packages and managing them independently from ActivitySim.

::

  # Open Command Prompt
  mkdir viz_project
  cd viz_project
  uv init
  uv add pandas==1.5.3

Many agencies use commercial software that have Python APIs and dependencies that may conflict with ActivitySim dependencies. 
In such cases, we also recommend creating a separate *uv* project for the commercial software and managing them independently from ActivitySim.

::

  # Open Command Prompt
  mkdir emme_project
  cd emme_project
  uv init --python 2.7
  # Then copy the emme.pth file (provides EMME API handshakes) from the Emme installation directory to emme_project/.venv/Lib/site-packages/

When having multiple *uv* projects, you can switch between them by activating the respective virtual environments.

::

  # Activate visualization project
  # Open Command Prompt
  cd path\to\viz_project
  .venv\Scripts\activate

  # Deactivate visualization project
  deactivate

How to share my working environment with others?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can share your working environment with others by sharing the ``uv.lock`` file and the ``pyproject.toml`` file located in your project directory (and ``.python-version`` file if it exists). 
The ``uv.lock`` file contains the exact versions of all packages and dependencies used in your project. 
Others can recreate the same environment by running the ``uv sync --locked`` command in a new project directory containing the shared files.

::

  # Initialize a new project directory
  mkdir new_asim_project
  cd new_asim_project

  # Copy .python-version file to new project directory (if exists)
  copy path\to\shared\.python-version .
  # Copy pyproject.toml file to new project directory
  copy path\to\shared\pyproject.toml .
  # Copy uv.lock file to new project directory
  copy path\to\shared\uv.lock .

  # Recreate the same environment
  uv sync --locked

Can other users on the same server or machine use my already created virtual environment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is doable but it requires additional setup and admin rights - therefore we do not generally recommend it. We recommend following the practice in :ref:`How to share my working environment with others?`. 

If you'd still like to proceed, here are the recommended steps to follow (proceed with caution!):

1. Ensure that you installed UV globally (requires admin rights) on the server/machine for all users. :ref:`Install UV` section above provides instructions on how to do this.

2. Assuming you installed UV globally in ``C:\shared\uv\``, ensure that all users have read and execute permissions to this directory.

3. Create a directory under ``C:\shared\uv\`` to install Python globally for UV. For example, open Command Prompt, create a directory named ``uv_python`` under ``C:\shared\uv\``.

::

  cd C:\shared\uv\
  mkdir uv_python

4. Under Environment Variables > System variables (requires Admin), create a new system environment variable named ``UV_PYTHON_INSTALL_DIR`` and set its value to the Python directory created in step 3 ``C:\shared\uv\uv_python\``.

5. Run the following command to install Python globally for UV. This should install Python executables globally in the ``UV_PYTHON_INSTALL_DIR`` directory.

::

  uv python install 3.10

6. Under Environment Variables > System variables (requires Admin), create a new system environment variable named ``UV_PYTHON`` and set its value to the ``python.exe`` created in step 5.

7. Create a directory to host UV projects under ``C:\shared\uv\``

::

  cd C:\shared\uv\
  mkdir uv_projects
  cd uv_projects

8. Create a new *uv* project and install ActivitySim using either :ref:`Option 1: From PyPI` or :ref:`Option 2: From Source with Lockfile` as described above.

9. Ensure that all users have read and execute permissions to the shared uv directory.

::

  icacls C:\shared\uv /reset /T

If I use the From PyPI option to install ActivitySim, would I run into dependency issues?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the :ref:`Option 1: From PyPI` option to install ActivitySim may result in different versions of deep dependencies than those tested by ActivitySim developers.
This is because the :ref:`Option 1: From PyPI` option installs only the direct dependencies listed in ActivitySim's ``pyproject.toml`` file,
and relies on *uv* to resolve and install the deep dependencies. It is likely that a newer version of ActivitySim deep dependencies
may cause compatibility issues. For example, see this recent update with ``numexpr``: https://github.com/pydata/numexpr/issues/540

When that happens, we recommend using the :ref:`Option 2: From Source with Lockfile` option to install ActivitySim, which ensures that
you are using the exact same deep dependencies as those tested by ActivitySim developers. In the meantime, you can also
report the compatibility issues to the ActivitySim development team via GitHub Issues, so that they can address them in future releases.

If I want to use ``uv run`` to run ActivitySim commands, do I still need to activate the virtual environment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
No, if you use ``uv run`` to run ActivitySim commands, you do not need to activate the virtual environment first.
However, you will need to call ``uv run`` in the project directory where the virtual environment is located. Also, like ``uv sync``, 
``uv run`` automatically updates the lockfile and installs any missing dependencies before running the command.
