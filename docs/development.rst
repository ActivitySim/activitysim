
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
* Work is done in an issue/feature branch (or a fork) and then pushed to a new branch
* The test system automatically runs the tests on the new branch
* If the tests pass, then a manual pull request can be approved to merge into master
* The repository administrator handles the pull request and makes sure that related resources such as the wiki, documentation, issues, etc. are updated.  See :ref:`release_steps` for more information.


Versioning
~~~~~~~~~~

ActivitySim uses the following `versioning convention <http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`__

::

  MAJOR.MINOR

* where MAJOR designates a major revision number for the software, like 2 or 3 for Python. Usually, raising a major revision number means that you are adding a lot of features, breaking backward-compatibility or drastically changing the APIs (Application Program Interface) or ABIs (Application Binary Interface).
* MINOR usually groups moderate changes to the software like bug fixes or minor improvements. Most of the time, end users can upgrade with no risks their software to a new minor release. In case an API changes, the end users will be notified with deprecation warnings. In other words, API and ABI stability is usually a promise between two minor releases.

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

A handy way to profile ActivitySim model runs is to use `snakeviz <https://jiffyclub.github.io/snakeviz/>`__.  
To do so, first install snakeviz and then run ActivitySim with the Python profiler (cProfile) to create 
a profiler file.  Then run snakeviz on the profiler file to interactively explore the component runtimes.

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
building the documentation if needed.  

When pushing revisions to the repo, the documentation is automatically built by Travis after 
successfully passing the tests.  The documents are built with the ``bin/build_docs.sh`` script.  
The script does the following:

* installs the required python packages
* runs ``make html``
* copies the ``master`` branch ``../activitysim/docs/_build/html/*`` pages to the ``gh-pages`` branch

GitHub automatically publishes the gh-pages branch at https://activitysim.github.io/activitysim.  

.. _release_steps :

Releases
~~~~~~~~

Before releasing a new version of ActivitySim, the following release checklist should be consulted:

* Create the required Anaconda environment
* Run all the examples, including the full scale example
* Build the package
* Install and run the package in a new Anaconda environment
* Build the documentation
* Run the tests
* Run pycodestyle
* Increment the package version number
* Update any necessary web links, such as switching from the develop branch to the master branch

ActivitySim releases are manually uploaded to the `Python Package Index <https://pypi.python.org/pypi/activitysim>`__  
(pypi) and also tagged as GitHub `releases <https://github.com/ActivitySim/activitysim/releases>`__.

Issues and Support
~~~~~~~~~~~~~~~~~~

Issue tracking and support is done through GitHub `issues <https://github.com/ActivitySim/activitysim/issues>`__.  

License
~~~~~~~

ActivitySim is provided "as is."  See the 
`License <https://github.com/ActivitySim/activitysim/blob/master/LICENSE.txt>`__ for more information.

Contribution Review Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When contributing to ActivitySim, the set of questions below will be asked of the contribution.  Make sure to also 
review the documentation above before making a submittal.  The automated test system also provides some helpful 
information where identified.

To submit a contribution for review, issue a pull request with a comment introducing your contribution.  The comment 
should include a brief overview, responses to the questions, and pointers to related information.  The entire submittal 
should ideally be self contained so any additional documentation should be in the pull request as well.  
The `PMC <https://github.com/ActivitySim/activitysim/wiki/Governance#project-management-committee-pmc>`__ and/or its Contractor will handle the review request, comment on each 
question, complete the feedback form, and reply to the pull request.  If accepted, the commit(s) will 
be `squashed and merged <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits>`__.
Its a good idea to setup a pre-submittal meeting to discuss questions and better understand expectations.

**Review Criteria**

  1. Does it contain all the required elements, including a runnable example, documentation, and tests?
  2. Does it implement good methods (i.e. is it consistent with good practices in travel modeling)?
  3. Are the runtimes reasonable and does it provide documentation justifying this claim?
  4. Does it include non-Python code, such as C/C++?  If so, does it compile on any OS and are compilation instructions included?
  5. Is it licensed with the ActivitySim license that allows the code to be freely distributed and modified and includes attribution so that the provenance of the code can be tracked? Does it include an official release of ownership from the funding agency if applicable?
  6. Does it appropriately interact with the data pipeline (i.e. it doesn't create new ways of managing data)?  
  7. Does it include regression tests to enable checking that consistent results will be returned when updates are made to the framework?
  8. Does it include sufficient test coverage and test data for existing and proposed features? 
  9. Any other comments or suggestions for improving the developer experience? 

**Feedback**

The PMC and/or its Contractor will provide feedback for each review criteria above and tag each submittal category as follows:

+-----------------------------------+-------------+-------------------+-------------------+
| Status                            | Code        | Documentation     | Tests/Examples    |
+===================================+=============+===================+===================+ 
| Accept                            |             |                   |                   |
+-----------------------------------+-------------+-------------------+-------------------+
| Accept but recommend revisions    |             |                   |                   |
+-----------------------------------+-------------+-------------------+-------------------+
| Do not accept                     |             |                   |                   |
+-----------------------------------+-------------+-------------------+-------------------+
