
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

Activitysim is implemented in the Python programming language. It also uses several open source Python packages such as pandas, numpy, pytables, openmatrix etc. Hence it is recommended that you install and use a *conda* package manager for your system.
One easy way to do so is by using
`Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`__.
Mamba is a free open source cross-platform package manager that runs on
Windows, OS X and Linux and is fully compatible with conda packages.  It is
also usually substantially faster than conda itself. Instructions to install mambaforge can be found `here <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`__. Installers for different Operating Systems can be found `here <https://github.com/conda-forge/miniforge#miniforge3>`__.

Alternatively, if you prefer a package installer backed by corporate tech
support available (for a fee) as necessary, you can install
`Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__,
although you should consult the `terms of service <https://www.anaconda.com/terms-of-service>`__
for this product and ensure you qualify since businesses and
governments with over 200 employees do not qualify for free usage.
If you're using `conda` instead of `mamba`, just replace every call to
`mamba` below with `conda`, as they share the same user interface and most
command formats.

If you access the internet from behind a firewall, then you may need to
configure your proxy server. To do so, create a `.condarc` file in your
home installation folder, such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false



Installing ActivitySim
----------------------

There are multiple ways to install the ActivitySim codebase:

1. Using a :ref:`Pre-packaged Installer` (recommended for users who do not need to change the Python code)

2. Using a :ref:`Python package manager like mamba <Using *mamba* package manager>` (recommended for users who need to change/customize the Python code)

3. Using :ref:`pip - Python's standard package manager <Using *pip* - Python's standard package manager>`

Pre-packaged Installer
______________________

Begining with version 1.2, ActivitySim is now available for Windows via a
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


Using *mamba* package manager
_____________________________

This method is recommended for ActivitySim users who also wish to customize the Python code to run their models. The steps involved are described as follows:

1. Install the *mamba* package manager as described in the :ref:`Software Requirements <Software>` subsection.

2. Create a conda environment (basically a Python install just for this project)
using mambaforge prompt or conda prompt depending on the package manager you use (on Windows) or the terminal (macOS or Linux)::

  mamba create -n asim python=3.10 activitysim -c conda-forge --override-channels

This command will create the environment and install all the dependencies
required for running ActivitySim.  It is only necessary to create the environment
once per machine, you do not need to (re)create the environment for each session.
If you would also like to install other tools or optional dependencies, it is
possible to do so by adding additional libraries to this command.  For example::

  mamba create -n asim python=3.10 activitysim jupyterlab larch -c conda-forge --override-channels

This example installs a specific version of Python, version 3.9.  A similar
approach can be used to install specific versions of other libraries as well,
including ActivitySim, itself. For example::

  mamba create -n asim python=3.9 activitysim=1.0.2 -c conda-forge --override-channels

Additional libraries can also be installed later.  You may want to consider these
tools for certain development tasks::

  # packages for testing
  mamba install pytest pytest-cov coveralls black flake8 pytest-regressions -c conda-forge --override-channels -n asim

  # packages for building documentation
  mamba install sphinx numpydoc sphinx_rtd_theme==0.5.2 -c conda-forge --override-channels -n asim

  # packages for estimation integration
  mamba install larch -c conda-forge --override-channels -n asim

  # packages for example notebooks
  mamba install jupyterlab matplotlib geopandas descartes -c conda-forge --override-channels -n asim

To create an environment containing all these optional dependencies at once, you
can run the shortcut command

::

  mamba env create activitysim/ASIM -n asim

3. To use the **asim** environment, you need to activate it

::
  conda activate asim

The activation of the correct environment needs to be done every time you
start a new session (e.g. opening a new conda Prompt window).

.. note::

  The *activate* and *deactivate* commands to start and stop using environments
  are called as `conda` even if you are otherwise using `mamba`. mamba is a drop-in replacement and uses the same commands and configuration options as conda.
  You can swap almost all commands between conda & mamba. For more details, refer to `the mamba user guide <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`__.

Using *pip* - Python's standard package manager
_______________________________________________

If you prefer to install ActivitySim without a package manager like *mamba* or *conda*, it is possible to
do so with pip, although you may find it more difficult to get all of the
required dependencies installed correctly.  If you can use conda for
the dependencies, you can get most of the libraries you need from there::

  # required packages for running ActivitySim
  mamba install cytoolz numpy pandas psutil pyarrow numba pytables pyyaml openmatrix requests -c conda-forge

  # required for ActivitySim version 1.0.1 and earlier
  pip install zbox

And then simply install activitysim with pip.

::

  python -m pip install activitysim

If you are using a firewall you may need to add ``--trusted-host pypi.python.org --proxy=myproxy.org:8080`` to this command.

For development work, can also install ActivitySim directly from source. Clone
the ActivitySim repository, and then from within that directory run::

  python -m pip install . -e

The "-e" will install in editable mode, so any changes you make to the ActivitySim
code will also be reflected in your installation.

Installing from source is easier if you have all the necessary dependencies already
installed in a development conda environment.  Developers can create an
environment that has all the optional dependencies preinstalled by running::

  mamba env create activitysim/ASIM-DEV

If you prefer to use a different environment name than `ASIM-DEV`, just
append `--name OTHERNAME` to the command. Then all that's left to do is install
ActivitySim itself in editable mode as described above.

.. note::

  ActivitySim is a 64bit Python 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__
  and `numpy <http://numpy.org>`__.

  As mentioned above, the recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.

  Anaconda includes the ``conda`` command line tool, which does a number of useful
  things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__
  (i.e. stand-alone Python installations/instances/sandboxes) that are the recommended
  way to work with multiple versions of Python on one machine.  Using conda
  environments keeps multiple Python setups from conflicting with one another.

  You need to activate the activitysim environment each time you start a new command
  session.  You can remove an environment with ``conda remove -n asim --all`` and
  check the current active environment with ``conda info -e``.

  For more information on Anaconda, see Anaconda's `getting started
  <https://docs.anaconda.com/anaconda/user-guide/getting-started>`__ guide.
