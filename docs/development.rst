
Software Development
====================

This page documents the ActivitySim software design and how to contribute to the project.

Software Design
---------------

The core software components of ActivitySim are described below.  ActivitySim is
implemented in Python, and makes heavy use of the vectorized backend C/C++ libraries in 
`pandas <http://pandas.pydata.org>`__  and `numpy <http://numpy.org>`__ in order to be quite performant.  
The core design principle of the system is vectorization of for loops, and this principle 
is woven into the system wherever reasonable.  As a result, the Python portions of the software 
can be thought of as more of an orchestrator, data processor, etc. that integrates a series of 
C/C++ vectorized data table and matrix operations.  The model system formulates 
each simulation as a series of vectorized table operations and the Python layer 
is responsible for setting up and providing expressions to operate on these large data tables.

In developing this software platform, we strive to adhere to a best practices approach to scientific computing, 
as summarized in `this article. <http://www.plosbiology.org/article/info%3Adoi%2F10.1371%2Fjournal.pbio.1001745>`__

Model Orchestrator
~~~~~~~~~~~~~~~~~~

An ActivitySim model is a sequence of model / data processing steps, commonly known as a data pipeline.
A well defined data pipeline has the ability to resume jobs at a known point, which facilitates 
debugging of problems with data and/or calculations.  It also allows for checkpointing model
resources, such as the state of each person at a point in the model simulation.  Checkpointing also
allows for regression testing of results at specified points in overall model run.

Earlier versions of ActivitySim depended on `ORCA <https://github.com/udst/orca>`__, an orchestration/pipeline tool 
that defines model steps, dynamic data sources, and connects them to processing functions. ORCA defined dynamic data tables 
based on pandas DataFrames, columns based on pandas Series, and injectables (functions).  Model steps 
were executed as steps registered with the ORCA engine.  Over time ActivitySim has extended ORCA's functionality by
adding a :ref:`pipeline_in_detail` that runs a series of model steps, manages the state of the data 
tables throughout the model run, allows for restarting at any model step, and integrates with the 
random number generation procedures (see :ref:`random_in_detail`).  As a result, ORCA is no longer a dependency of
the system.  See :mod:`activitysim.core.inject` for more information.

Data Handling
~~~~~~~~~~~~~

ActivitySim works with three open data formats, `HDF5 <https://www.hdfgroup.org/HDF5/>`__ 
, `Open Matrix (OMX) <https://github.com/osPlanning/omx>`__, and `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__ . 
The HDF5 binary data container is used for the :ref:`pipeline_in_detail` data store.
OMX, which is based on HDF5, is used for input and output matrices (skims and demand matrices).  CSV files 
are used for various inputs and outputs as well.

Three key data structures in ActivitySim are:

* `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__ - A data table with rows and columns, similar to an R data frame, Excel worksheet, or database table
* `pandas.Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`__ - a vector of data, a column in a DataFrame table or a 1D array
* `numpy.array <http://docs.scipy.org/doc/numpy/reference/arrays.html>`__ - an N-dimensional array of items of the same type, and is often a network skim matrix or collection of skim matrices by time-period or mode for example

Expressions
~~~~~~~~~~~

ActivitySim exposes all model expressions in CSV files.  These model expression CSV files
contain Python expressions, mainly pandas/numpy expression, and reference input data tables 
and network skim matrices.  With this design, the Python code, which can be thought of as a generic expression 
engine, and the specific model calculations, such as the utilities, are separate.  This helps to avoid 
modifying the actual Python code when making changes to the models, such as during model calibration. An 
example of model expressions is found in the example auto ownership model specification file - 
`auto_ownership.csv <https://github.com/activitysim/activitysim/blob/master/example/configs/auto_ownership.csv>`__. 
Refer to the :ref:`expressions` section for more detail.

Many of the models have pre- and post-processor table annotators, which read a CSV file of expression, calculate 
required additional table fields, and join the fields to the target tables.  An example table annotation expressions 
file is found in the example configuration files for households for the CDAP model - 
`annotate_households_cdap.csv <https://github.com/activitysim/activitysim/blob/master/example/configs/annotate_households_cdap.csv>`__. 
Refer to :ref:`table_annotation` for more information and the :func:`activitysim.abm.models.util.expressions.assign_columns` function.

Choice Models
~~~~~~~~~~~~~

ActivitySim currently supports multinomial (MNL) and nested logit (NL) choice models. Refer to :ref:`logit_in_detail` 
for more information.  It also supports custom expressions as noted above, which can often be used to 
code additional types of choice models.  In addition, developers can write their own choice models 
in Python and expose these through the framework.  

Person Time Windows
~~~~~~~~~~~~~~~~~~~

The departure time and duration models require person time windows. Time windows are adjacent time 
periods that are available for travel. ActivitySim maintains time windows in a pandas table where each row is 
a person and each time period is a column.  As travel is scheduled throughout the simulation, the relevant 
columns for the tour, trip, etc. are updated as needed. Refer to :ref:`time_windows` for more information.

Models
~~~~~~

An activitysim travel model is made up of a series of models, or steps in the data pipeline.  A model
typically does the following:

  * registers an ORCA step that is called by the model runner
  * sets up logging and tracing
  * gets the relevant input data tables from ORCA
  * gets all required settings, config files, etc.
  * runs a data preprocessor on each input table that needs additional fields for the calculation
  * solves the model in chunks of data table rows
  * runs a data postprocessor on the output table data that needs additional fields for later models
  * writes the resulting table data to the pipeline

See :ref:`models` for more information. 


Development Install
-------------------

The development version of ActivitySim can be installed as follows:

* Clone or fork the source from the `GitHub repository <https://github.com/activitysim/activitysim>`__
* Activate the correct conda environment if needed
* Navigate to your local activitysim git directory
* Run the command ``python setup.py develop``

The ``develop`` command is required in order to make changes to the 
source and see the results without reinstalling.  You may need to first uninstall the
the pip installed version before installing the development version from source.  This is 
done with ``pip uninstall activitysim``.


Development Guidelines
----------------------

ActivitySim development adheres to the following standards.

Style
~~~~~

* Python code should follow the `pycodestyle style guide <https://pypi.python.org/pypi/pycodestyle>`__
* Python docstrings should follow the `numpydoc documentation format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__

Imports
~~~~~~~

* Imports should be one per line.
* Imports should be grouped into standard library, third-party, and intra-library imports. 
* ``from`` import should follow regular ``imports``.
* Within each group the imports should be alphabetized.
* Imports of scientific Python libraries should follow these conventions:

::

    import numpy as np
    import pandas as pd


Working Together in the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `GitHub Flow <https://guides.github.com/introduction/flow>`__.  The key points to 
our GitHub workflow are:

* The master branch contains the latest working/release version of the ActivitySim resources
* The master branch is protected and therefore can only be written to by the `Travis <https://travis-ci.org/>`__ CI system
* Work is done in an issue/feature branch (or a fork) and then pushed to a new brach
* The test system automatically runs the tests for a subset of the model (a subset of zones, households, and models) on the new branch
* If the tests pass, then a manual pull request can be approved to merge into master
* The repository administrator handles the pull request and makes sure that related resources such as the wiki, documentation, issues, etc. are updated.  The repository administrator handles the pull request and makes sure that related resources such as the wiki, documentation, issues, etc. are updated.  The repository manager also runs the full scale model (all zones, households, and models) to ensure the example model runs successfully.  
* Every time a merge is made to master, the version is incremented and a new package posted to `Python Package Index <https://pypi.org/>`__

Versioning
~~~~~~~~~~

ActivitySim uses the following `versioning convention <http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`__

::

  MAJOR.MINOR[.MICRO]

* where MAJOR designates a major revision number for the software, like 2 or 3 for Python. Usually, raising a major revision number means that you are adding a lot of features, breaking backward-compatibility or drastically changing the APIs or ABIs.
* MINOR usually groups moderate changes to the software like bug fixes or minor improvements. Most of the time, end users can upgrade with no risks their software to a new minor release. In case an API changes, the end users will be notified with deprecation warnings. In other words, API and ABI stability is usually a promise between two minor releases.
* Some softwares use a third level: MICRO. This level is used when the release cycle of minor release is quite long. In that case, micro releases are dedicated to bug fixes.

Testing
~~~~~~~

ActivitySim testing is done with three tools:

* `pycodestyle <https://pypi.python.org/pypi/pycodestyle>`__, a tool to check Python code against the pycodestyle style conventions
* `pytest <http://pytest.org/latest/>`__, a Python testing tool
* `coveralls <https://github.com/coagulant/coveralls-python>`__, a tool for measuring code coverage and publishing code coverage stats online

To run the tests locally, first make sure the required packages are installed:

::

    pip install pytest pytest-cov coveralls pycodestyle
    

Next, run the tests with the following commands:

::

    pycodestyle activitysim
    py.test --cov activitysim --cov-report term-missing

These same tests are run by Travis with each push to the repository.  These tests need to pass in order
to merge the revisions into master.

In some cases, test targets need to be updated to match the new results produced by the code since these 
are now the correct results.  In order to update the test targets, first determine which tests are 
failing and then review the failing lines in the source files.  These are easy to identify since each 
test ultimately comes down to one of Python's various types of `assert` statements.  Once you identify 
which `assert` is failing, you can work your way back through the code that creates the test targets in 
order to update it.  After updating the test targets, re-run the tests to confirm the new code passes all 
the tests.

Profiling
~~~~~~~~~

A good way to profile ActivitySim model runs is to use `snakeviz <https://jiffyclub.github.io/snakeviz/>`__.  
To do so, first install snakeviz and then run ActivitySim with the Python profiler (cProfile) to create 
a profiler file.  Then run snakeviz on the profiler file to interactively explore the component runtimes.

::

    pip install snakeviz
    python -m cProfile -o asim.prof simulation.py
    snakeviz asim.prof

Documentation
~~~~~~~~~~~~~

The documentation is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ markup 
and built with `Sphinx <http://www.sphinx-doc.org/en/stable/>`__.  In addition to converting rst files
to html and other document formats, these tools also read the inline Python docstrings and convert
them into html as well.  ActivitySim's docstrings are written in `numpydoc format
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__ since it is easier to use 
than standard rst format.

To build the documentation, first make sure the required packages are installed:

::

    pip install sphinx numpydoc sphinx_rtd_theme

Next, build the documentation in html format with the following command run from the ``docs`` folder:

::

    make html

If the activitysim package is installed, then the documentation will be built from that version of 
the source code instead of the git repo version.  Make sure to ``pip uninstall activitysim`` before 
bulding the documentation if needed.  

When pushing revisions to the repo, the documentation is automatically built by Travis after 
successfully passing the tests.  The documents are built with the ``bin/build_docs.sh`` script.  
The script does the following:

* installs the required python packages
* runs ``make html``
* copies the ``master`` branch ``../activitysim/docs/_build/html/*`` pages to the ``gh-pages`` branch

GitHub automagically publishes the gh-pages branch at https://activitysim.github.io/activitysim.  

Releases
~~~~~~~~

ActivitySim releases are manually uploaded to the `Python Package Index <https://pypi.python.org/pypi/activitysim>`__  (pypi). 
