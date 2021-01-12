
Getting Started
===============

This page describes how to get started with ActivitySim.

.. note::
   ActivitySim is under development


.. index:: installation


Installation
------------

1. Install `Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__.  It is best to use Anaconda as noted below with ActivitySim.
2. If you access the internet from behind a firewall, then you need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder, such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

3. Create and activate an Anaconda environment (basically a Python install just for this project) using Anaconda Prompt or the terminal.

::

  conda create -n asimtest python=3.7

  #Windows
  activate asimtest

  #Mac
  source activate asimtest

4. Get and install other required libraries on the activated conda Python environment using `pip <https://pypi.org/project/pip>`__:

::

  # required packages for running ActivitySim
  conda install cytoolz numpy pandas psutil pyarrow numba
  conda install -c anaconda pytables pyyaml
  pip install openmatrix zbox requests

  # optional required packages for testing and building documentation
  conda install pytest pytest-cov coveralls pycodestyle
  conda install sphinx numpydoc sphinx_rtd_theme
  
  # optional required packages for example notebooks and estimation integration
  conda install jupyterlab          # for notebooks
  conda install matplotlib          # for charts
  conda install geopandas descartes # for maps
  conda install larch               # for estimation

5. If you access the internet from behind a firewall, then you need to configure your proxy server when downloading packages.

For `conda` for example, create a `.condarc` file in your Anaconda installation folder with the following:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

For `pip` for example:

::

  pip install --trusted-host pypi.python.org --proxy=myproxy.org:8080  openmatrix

6. Get and install the ActivitySim package on the activated conda Python environment:

::

  #new install
  pip install activitysim

  #update to a new release
  pip install -U activitysim

.. note::

  ActivitySim is a 64bit Python 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__
  and `numpy <http://numpy.org>`__.

  The recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.

  Anaconda includes the ``conda`` command line tool, which does a number of useful
  things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__
  (i.e. stand-alone Python installations/instances/sandboxes) that are the recommended
  way to work with multiple versions of Python on one machine.  Using conda
  environments keeps multiple Python setups from conflicting with one another.

  You need to activate the activitysim environment each time you start a new command
  session.  You can remove an environment with ``conda remove -n asimtest --all`` and
  check the current active environment with ``conda info -e``.

  For more information on Anaconda, see Anaconda's `getting started
  <https://docs.anaconda.com/anaconda/user-guide/getting-started>`__ guide.

Run the Example
---------------

ActivitySim includes a :ref:`cli` for creating examples and running the model.

To setup and run the primary :ref:`example`, do the following:

* Open a command prompt
* Activate the Anaconda environment with ActivitySim installed (i.e. asimtest)
* Type ``activitysim create -e example_mtc -d test_example_mtc`` to copy the very small MTC example to a new test_example_mtc directory
* Change to the test_example_mtc directory
* Type ``activitysim run -c configs -o output -d data`` to run the example
* Review the outputs in the output directory

.. note::
   Common configuration settings can be overridden at runtime.  See ``activitysim -h``, ``activitysim create -h`` and ``activitysim run -h``.

More complete examples, including the full scale MTC regional demand model, estimation integration examples, and multiple zone system examples, 
are available for creation by typing ``activitysim create -l``.  To create these examples, ActivitySim downloads the (large) input files from 
the `ActivitySim resources <https://github.com/rsginc/activitysim_resources>`__ repository.

Try the Notebooks
-----------------

ActivitySim includes a `Jupyter Notebook <https://jupyter.org>`__ recipe book with interactive examples.  To run a Jupyter notebook, do the following:

* Open an Anaconda prompt and activate the Anaconda environment with ActivitySim installed
* If needed, ``conda install jupyterlab`` so you can run jupyter notebooks
* Type ``jupyter notebook`` to launch the web-based notebook manager
* Navigate to the examples notebooks folder and select a notebook to learn more:

  * `Getting started <https://github.com/ActivitySim/activitysim/blob/master/activitysim/examples/example_mtc/notebooks/getting_started.ipynb/>`__
  * `Summarizing results <https://github.com/ActivitySim/activitysim/blob/master/activitysim/examples/example_mtc/notebooks/summarizing_results.ipynb/>`__
  * `Testing a change in auto ownership <https://github.com/ActivitySim/activitysim/blob/master/activitysim/examples/example_mtc/notebooks/change_in_auto_ownership.ipynb/>`__
  * `Adding TNCs <https://github.com/ActivitySim/activitysim/blob/master/activitysim/examples/example_mtc/notebooks/adding_tncs.ipynb/>`__

Hardware
--------

The computing hardware required to run a model implemented in the ActivitySim framework generally depends on:

* The number of households to be simulated for disaggregate model steps
* The number of model zones (for each zone system) for aggregate model steps
* The number and size of network skims by mode and time-of-day
* The number of zone systems, see :ref:`multiple_zone_systems`
* The desired runtimes

ActivitySim framework models use a significant amount of RAM since they store data in-memory to reduce
access time in order to minimize runtime.  For example, the example MTC Travel Model One model has 2.7 million
households, 7.5 million people, 1475 zones, 826 network skims and has been run between one hour and one day depending
on the amount of RAM and number of processors allocated.

.. note::
   ActivitySim has been run in the cloud, on both Windows and Linux using
   `Microsoft Azure <https://azure.microsoft.com/en-us/>`__.  Example configurations, 
   scripts, and runtimes are in the ``other_resources\example_azure`` folder.
