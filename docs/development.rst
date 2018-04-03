Development
===========

This page documents how to contribute to ActivitySim.

In developing this software platform, we strive to adhere to a best practices approach to scientific computing, 
as summarized in `this article. <http://www.plosbiology.org/article/info%3Adoi%2F10.1371%2Fjournal.pbio.1001745>`__

Software Design
---------------
ActivitySim is implemented in Python, and makes heavy use of the vectorized backend 
C/C++ libraries in pandas and numpy. The core design principle of the system is 
vectorization of for loops, and this principle is woven into the system wherever 
reasonable. As a result, the Python portions of the software can be thought of as 
more of an orchestrator, data processor, etc. that integrates a series of C/C++ 
vectorized data table and matrix operations. The model system formulates each 
simulation as a series of vectorized table operations and the Python layer is 
responsible for setting up and providing expressions to operate on these large 
data tables.

Style
-----

* Python code should follow the `pycodestyle style guide <https://pypi.python.org/pypi/pycodestyle>`__
* Python docstrings should follow the `numpydoc documentation format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__

Imports
-------

* Imports should be one per line.
* Imports should be grouped into standard library, third-party, and intra-library imports. 
* ``from`` import should follow regular ``imports``.
* Within each group the imports should be alphabetized.
* Imports of scientific Python libraries should follow these conventions:

::

    import numpy as np
    import pandas as pd
    import scipy as sp

Working Together in the Repository
----------------------------------

We use `GitHub Flow <https://guides.github.com/introduction/flow>`__.  The key points to 
our GitHub workflow are:

* The master branch contains the latest working/release version of the ActivitySim resources
* The master branch is protected and therefore can only be written to by the `Travis <https://travis-ci.org/>`__ CI system
* Work is done in an issue/feature branch (or a fork) and then pushed to a new brach in the repo
* The test system automatically runs the tests on the new branch
* If the tests pass, then a manual pull request can be approved to merge into master
* The repository administrator handles the pull request and makes sure that related resources such as the wiki, documentation, issues, etc. are updated
* Every time a merge is made to master, the version should be incremented and a new package posted to PyPI

Versioning
----------
ActivitySim uses the following `versioning convention <http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`__

::

  MAJOR.MINOR[.MICRO]

* where MAJOR designates a major revision number for the software, like 2 or 3 for Python. Usually, raising a major revision number means that you are adding a lot of features, breaking backward-compatibility or drastically changing the APIs or ABIs.
* MINOR usually groups moderate changes to the software like bug fixes or minor improvements. Most of the time, end users can upgrade with no risks their software to a new minor release. In case an API changes, the end users will be notified with deprecation warnings. In other words, API and ABI stability is usually a promise between two minor releases.
* Some softwares use a third level: MICRO. This level is used when the release cycle of minor release is quite long. In that case, micro releases are dedicated to bug fixes.

Testing
-------

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
---------

A good way to profile ActivitySim model runs is to use `snakeviz <https://jiffyclub.github.io/snakeviz/>`__.  
To do so, first install snakeviz and then run ActivitySim with the Python profiler (cProfile) to create 
a profiler file.  Then run snakeviz on the profiler file to interactively explore the component runtimes.

::

    pip install snakeviz
    python -m cProfile -o asim.prof simulation.py
    snakeviz asim.prof

Documentation
-------------

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
--------

ActivitySim releases are manually uploaded to the `Python Package Index <https://pypi.python.org/pypi/activitysim>`__  (pypi). 
