
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

ActivitySim's model orchestrator makes use of depedency injection, which is where one object (or method)
supplies the dependencies of another object.  Dependency injection is done by the :mod:`activitysim.core.inject`
module, which wraps `ORCA <https://github.com/udst/orca>`__, an orchestration/pipeline tool.  Inject defines model
steps, dynamic data sources, and connects them to processing functions. It also defines dynamic data tables
based on pandas DataFrames, columns based on pandas Series, and injectables (functions).  Model steps are executed
as steps registered with the model orchestration engine.  Over time Inject has extended ORCA's functionality by
adding a :ref:`pipeline_in_detail` that runs a series of model steps, manages the state of the data
tables throughout the model run, allows for restarting at any model step, and integrates with the
random number generation procedures (see :ref:`random_in_detail`).

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
`auto_ownership.csv <https://github.com/activitysim/activitysim/blob/main/example/configs/auto_ownership.csv>`__.
Refer to the :ref:`expressions` section for more detail.

Many of the models have pre- and post-processor table annotators, which read a CSV file of expression, calculate
required additional table fields, and join the fields to the target tables.  An example table annotation expressions
file is found in the example configuration files for households for the CDAP model -
`annotate_households_cdap.csv <https://github.com/activitysim/activitysim/blob/main/example/configs/annotate_households_cdap.csv>`__.
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

  * registers an Inject step that is called by the model runner
  * sets up logging and tracing
  * gets the relevant input data tables from Inject
  * gets all required settings, config files, etc.
  * runs a data preprocessor on each input table that needs additional fields for the calculation
  * solves the model in chunks of data table rows if needed
  * runs a data postprocessor on the output table data that needs additional fields for later models
  * writes the resulting table data to the pipeline

See :ref:`models` for more information.


Development Install
-------------------

The development version of ActivitySim can be installed as follows:

* Clone or fork the source from the `GitHub repository <https://github.com/activitysim/activitysim>`__
* Navigate to your local activitysim git directory
* Create a development environment by running
  ``conda env create --file=conda-environments/activitysim-dev.yml --name ASIM_DEV``.
  This will create a new conda environment named "ASIM_DEV" (change the name in
  the command if desired). ActivitySim will be installed in "editable" mode, so
  any changes you make in the code in your git directory will be reflected.
* Activate the new conda environment ``conda activate ASIM_DEV``


Development Guidelines
----------------------

ActivitySim development adheres to the following standards.

Style
~~~~~

* Python code should follow the `black code style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`__
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

* The ``main`` branch contains the latest release version of the ActivitySim resources
* The ``develop`` branch contains new features or revisions planned for the next release.
  Generally, developers should not work directly in the ``develop`` branch.
* Work to implement new features or other revisions is done in an issue/feature branch
  (or a fork) and developers can open a pull request (PR) to merge their work into ``develop``.
* The test system automatically runs the tests on PR's.  PR's do not necessarily need to pass
  all tests to be merged into ``develop``, but any failures should be cause by known existing
  problems -- PR's should strive to not break anything beyond what was broken previously.
* Upon review and agreement by a consortium member or committer other than the author,
  and barring any objection raised by a consortium member, PR's can be merged into the
  ``develop`` branch.
* If tests pass for the ``develop`` branch, new features are suitably documented, and on approval of
  `a lazy majority of the PMC <https://github.com/ActivitySim/activitysim/wiki/Governance#actions>`__,
  a repository administrator can approve a manual pull request to merge ``develop`` into ``main``,
  and otherwise make a `product release <https://github.com/ActivitySim/activitysim/blob/main/HOW_TO_RELEASE.md>`__.


Versioning
~~~~~~~~~~

ActivitySim uses the following `versioning convention <http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`__:

::

  MAJOR.MINOR.PATCH[.devN]

* where MAJOR designates a major revision number for the software, like 2 or 3 for Python.
  Usually, raising a major revision number means that you are adding a lot of features,
  breaking backward-compatibility or drastically changing the APIs (Application Program
  Interface) or ABIs (Application Binary Interface).
* MINOR usually groups moderate changes to the software like bug fixes or minor improvements.
  Most of the time, end users can upgrade with no risks their software to a new minor release.
  In case an API changes, the end users will be notified with deprecation warnings. In other
  words, API and ABI stability is usually a promise between two minor releases.
* PATCH releases are made principally to address bugs or update non-core parts of the
  ActivitySim codebase (e.g. dependency requirements, distribution channels). End users
  should expect no changes at all in how the software works between two patch releases.
* DEVELOPMENT pre-releases are used to test and prepare integration with other external
  services that require a "release". End users should not typically install or use a development
  release other than for a specific well-defined purpose.

Testing
~~~~~~~

ActivitySim testing is done with several tools:

* `black <https://black.readthedocs.io>`__, a tool to check and enforce black
  code style on Python code
* `isort <https://pycqa.github.io/isort/>` to organize imports
* `pytest <http://pytest.org/latest/>`__, a Python testing tool
* `coveralls <https://github.com/coagulant/coveralls-python>`__, a tool for measuring code coverage and publishing code coverage stats online

To run the tests locally, first make sure the required packages are installed.  Next, run the tests with the following commands:

::

    black --check
    py.test

These same tests are run by Travis with each push to the repository.  These tests need to pass in order
to merge the revisions into main.

In some cases, test targets need to be updated to match the new results produced by the code since these
are now the correct results.  In order to update the test targets, first determine which tests are
failing and then review the failing lines in the source files.  These are easy to identify since each
test ultimately comes down to one of Python's various types of `assert` statements.  Once you identify
which `assert` is failing, you can work your way back through the code that creates the test targets in
order to update it.  After updating the test targets, re-run the tests to confirm the new code passes all
the tests.

See :ref:`adding_agency_examples` for more information on testing, most notably, agency example models.

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

To build the documentation, first make sure the required packages are installed.  Next, build the
documentation in html format with the ``make html`` command run from the ``docs`` folder.

If the activitysim package is installed, then the documentation will be built from that version of
the source code instead of the git repo version.  When pushing revisions to the repo, the documentation
is automatically built by Travis after successfully passing the tests.

GitHub automatically publishes the gh-pages branch at https://activitysim.github.io/activitysim.

.. _release_steps :

Releases
~~~~~~~~

With the agreement of the PMC, a project administrator will handle making releases, following the detailed
steps outlined in the `HOW_TO_RELEASE <https://github.com/ActivitySim/activitysim/blob/main/HOW_TO_RELEASE.md>`__
document.


Issues and Support
~~~~~~~~~~~~~~~~~~

Issue tracking and support is done through GitHub `issues <https://github.com/ActivitySim/activitysim/issues>`__.

License
~~~~~~~

ActivitySim is provided "as is."  See the
`License <https://github.com/ActivitySim/activitysim/blob/main/LICENSE.txt>`__ for more information.

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

.. _adding_agency_examples :

Adding Agency Examples
----------------------

ActivitySim includes several mature or in-development full scale agency :ref:`examples`.  Adding an agency example to
ActivitySim adds additional assurances that future updates to ActivitySim will more easily work for users.  At the same
time, with each additional implementation, the need for additional test coverage increases.  This increased need for
test coverage relates to when setting up a new model, with differences in inputs and configurations, when adding new
model components (and/or revisions to the core) in order to implement new features, and when implementing model
components at a scale previously untested.  The following section describes the process to add an agency example model
to ActivitySim.

Examples
~~~~~~~~

Generally speaking, there are two types of ActivitySim examples: test examples and agency examples.

* Test examples - these are the core ActivitySim maintained and tested examples developed to date.  The current test
  examples are :ref:`prototype_mtc`, :ref:`example_estimation`, :ref:`placeholder_multiple_zone`, and :ref:`prototype_marin`.
  These examples are owned and maintained by the project.
* Agency examples - these are agency partner model implementations currently being setup.  The current agency examples
  are :ref:`prototype_arc`, :ref:`prototype_semcog`, :ref:`placeholder_psrc`, :ref:`placeholder_sandag`, and :ref:`prototype_sandag_xborder`.  These examples can be
  configured in ways different from the test examples, include new inputs and expressions, and may include new planned
  software components for contribution to ActivitySim.  These examples are owned by the agency.

Furthermore, multiple versions of these examples can exist, and be used for various testing purposes:

* Full scale - a full scale data setup, including all households, zones, skims, time periods, etc.  This is a "typical"
  model setup used for application.  This setup can be used to test the model results and performance since model
  results can be compared to observed/known answers and runtimes can be compared to industry experience.  It can also
  be used to test core software functionality such as tracing and repeatability.
* Cropped - a subset of households and zones for efficient / portable running for testing.  This setup can really only
  be used to test the software since model results are difficult to compare to observed/known answers.  This version of
  an example is not recommended for testing overall runtime since it's a convenience sample and may not represent the
  true regional model travel demand patterns.  However, depending on the question, this setup may be able to answer
  questions related to runtime, such as improvements to methods indifferent to the size of the population and number of
  zones.
* Other - a specific route/path through the code for testing.  For example, the estimation example tests the estimation
  mode functionality.  The estimation example is a version of the example prototype MTC example - it inherits most settings from
  prototype_mtc and includes additional settings for reading in survey files and producing estimation data bundles.

Regardless of the type or version, all functioning examples are described in a common list stored in
`example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_.
Each item included in this file represents one example, and each includes the following tags:

* *name*: A unique name for the example, used to identify the example when using the `activitysim create` command. The
  naming convention used is to give each example a name that is all lower case, and which uses underscores to separate
  words.
* *description*: A short sentence describing the example.
* *include*: A list of files or directories to include in the example.  For smaller input files (e.g. configuration
  files, or data files used on "test" sized examples), each file or directory to include can be given as a simple
  string, which specifies the subdirectory of the embedded ActivitySim examples to be copied into a working directory.
  For larger files that are not embedded into the main ActivitySim GitHub repository, items are given as a 3-tuple:
  (url, target_path, sha256). The `url` points to a publicly available address where the file can be downloaded, the
  `target_path` gives the relative filepath where the file should be installed in the working directory, and the
  `sha256` is a checksum used to verify the file was downloaded correctly (and to prevent re-downloading when the file
  is already available).  For defining new examples, use the `sha256_checksum` function to get a file's checksum that
  should be included in the example manifest.

Testing
~~~~~~~

The test plan for test examples versus agency examples is different:

* Test examples test software features such as stability, tracing, expression solving, etc.  This set of tests is run
  by the TravisCI system and is a central feature of the software development process.
* Agency examples test a complete run of the cropped version to ensure it runs and the results are as expected.  This
  is done via a simple run model test that runs the cropped version and compares the output trip list to the expected
  trip list.  This is what is known as a regression test.  This test is also run by TravisCI.

Both types of examples are stored in the ActivitySim repositories for version control and collaborative maintenance.
There are two storage locations:

* The `activitysim package example folder <https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples>`_,
  which stores the test and agency example setup files, cropped data and cropping script, regression test script,
  expected results, and a change log to track any revisions to the example to get it working for testing.  These
  resources are the resources automatically tested by the TravisCI test system with each revision to the software.
* The `activitysim_resources repository <https://github.com/activitysim/activitysim_resources>`_, which stores just the
  full scale example data inputs using `Git LFS <https://git-lfs.github.com>`_.  This repository has a monthly cost and
  takes time to upload/download and so the contents of it are separate from the main software repository.  These
  resources are the resources periodically and manually tested (for now).

This two-part solution allows for the main activitysim repo to remain relatively lightweight, while providing an
organized and accessible storage solution for the full scale example data.  The ActivitySim command line interface for
creating and running examples makes uses the
`example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
to maintain the dictionary of the examples and how to get and run them.

Running the Test System
~~~~~~~~~~~~~~~~~~~~~~~

The automatic TravisCI test system runs the test examples and the cropped agency examples.  Examples of the testing
resources for each agency example that need to be up-to-date are:

* `scripts folder (including crop script) <https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_semcog/scripts>`_
* `test folder (including test script) <https://github.com/ActivitySim/activitysim/main/main/activitysim/examples/prototype_semcog/test>`_
* `regress folder (including expected outputs) <https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_semcog/test/regress>`_

For the time being, running the full scale examples is done manually since it involves getting and running several large examples that take many hours to run.  The entire system could be fully automated, and either run in the cloud or on a local server.

Update Use Cases
~~~~~~~~~~~~~~~~

To better illustrate the workflow for adding agency examples, a series of use cases is discussed.

When a new version of the code is pushed to develop:

* The automatic test system is run to ensure the tests associated with the test examples pass.  If any of the tests do not pass, then either the code or the expected test results are updated until the tests pass.
* The automatic test system also runs each cropped agency example regression test to ensure the model runs and produces the same results as before.  If any of the tests do not pass, then either the code or the expected test results are updated until the tests pass.  However, the process for resolving issues with agency example test failure has two parts:

  * If the agency example previous ran without error or future warnings (i.e. deprecation warnings and is therefore up-to-date), then the developer will be responsible for updating the agency example so it passes the tests
  * If the agency example previously threw errors or future warnings (i.e. is not up-to-date), then the developer will not update the example and the responsibility will fall to the agency to update it when they have time.  This will not preclude development from advancing since the agency specific test can fail while the other tests continue to pass.  If the agency example is not updated within an agreed upon time frame, then the example is removed from the test system.

To help understand this case, the addition of support for representative logsums to :ref:`prototype_mtc` is discussed.  prototype_mtc was selected as the test case for development of this feature because this feature could be implemented and tested against this example, which is the primary example to date.  With the new feature configured for this example, the automatic test system was run to ensure all the existing test examples pass their tests.  The automatic test system was also run to ensure all the cropped agency examples passed their tests, but since not of them include this new feature in their configuration, the test results were the same and therefore the tests passed.

When an agency wants to update their example:

* It is recommended that agencies keep their examples up-to-date to minimize the cost/effort of updating to new versions of ActivitySim.  However, the frequency with which to make that update is a key issue.  The recommended frequency of ensuring the agency example is up-to-date depends on the ActivitySim development roadmap/phasing and the current features being developed.  Based on past project experience, it probably makes sense to not let agency examples fall more than a few months behind schedule, or else updates can get onerous.

* When making an agency model update, agencies update their example through a pull request.  This pull request changes nothing outside their example folder.  The updated resources may include updated configs, inputs, revisions to the cropped data/cropping script, and expected test results.  The automatic cropped example test must run without warnings.  The results of the full scale version is shared with the development team in the PR comments.

To help understand this case, the inclusion of :ref:`placeholder_psrc` as an agency example is discussed.  This model is PSRC's experimentation of a two zone model and is useful for testing the two zone features, including runtime.  A snapshot of PSRC's efforts to setup an ActivitySim model with PSRC inputs was added to the test system as a new agency example, called placeholder_psrc.  After some back and forth between the development team and PSRC, a full scale version of placeholder_psrc was successfully run.  The revisions required to create a cropped version and full scale version were saved in a change log included with the example.  When PSRC wants to update placeholder_psrc, PSRC will pull the latest develop code branch and then update placeholder_psrc so the cropped and full scale example both run without errors.  PSRC also needs to update the expected test results.  Once everything is in good working order, then PSRC issues a pull request to develop to pull their updated example.  Once pulled, the automatic test system will run the cropped version of placeholder_psrc.

When an agency example includes new submodels and/or contributions to the core that need to be reviewed and then pulled/accepted:

* First, the agency example must comply with the steps outlined above under "When an agency wants to update their example".
* Second, the agency example must be up-to-date with the latest develop version of the code so the revisions to the code are only the exact revisions for the new submodels and/or contributions to the core.
* The new submodels and/or contributions to the core will then be reviewed by the repository manager and it's likely some revisions will be required for acceptance.  Key items in the review include python code, user documentation, and testable examples for all new components.  If the contribution is just new submodels, then the agency example that exercises the new submodel is sufficient for test coverage since TravisCI will automatically test the cropped version of the new submodel.  If the contribution includes revisions to the core that impact other test examples, then the developer is responsible for ensuring all the other tests that are up-to-date are updated/passing as well.  This includes other agency examples that are up-to-date.  This is required to ensure the contribution to the core is adequately complete.

To help understand this case, the addition of the parking location choice model for :ref:`prototype_arc` is discussed.  First, ARC gets their example in good working order - i.e. updates to develop, makes any required revisions to their model to get it working, creates a cropped and full scaled example, and creates the expected test results.  In addition, this use case includes additional submodel and/or core code so ARC also authors the new feature, including documentation and any other relevant requirements such as logging, tracing, support for estimation, etc.  With the new example and feature working offline, then ARC issues a pull request to add prototype_arc and the new submodel/core code and makes sure the automatic tests are passing.  Once accepted, the automatic test system will run the test example tests and the cropped agency examples.  Since the new feature - parking location choice model - is included in prototype_arc, then new feature is now tested.  Any testing of downstream impacts from the parking location choice model would also need to be implemented in the example.
