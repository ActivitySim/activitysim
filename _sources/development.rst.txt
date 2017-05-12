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

* Python code should follow the `pep8 style guide <http://legacy.python.org/dev/peps/pep-0008/>`__
* Python docstrings should follow the `numpydoc documentation format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__

Imports
-------

* Imports should be one per line.
* Imports should be grouped into standard library, third-party, and intra-library imports. 
* ``from`` import should follow regular ``imports``.
* Within each group the imports should be alphabetized.
* Imports of scientific Python libraries should follow these conventions:

::

    import matplotlib.pyplot as plt
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
* If the tests pass, then a pull request can be approved to merge into master
* The repository administrator handles the pull request and makes sure that related resources such as the wiki, documentation, issues, etc. are updated

Testing
-------

ActivitySim testing is done with three tools:

* `pep8 <http://pep8.readthedocs.org/en/latest/intro.html>`__, a tool to check Python code against the PEP8 style conventions
* `pytest <http://pytest.org/latest/>`__, a Python testing tool
* `coveralls <https://github.com/coagulant/coveralls-python>`__, a tool for measuring code coverage and publishing code coverage stats online

To run the tests locally, first make sure the required packages are installed:

::

    pip install pytest pytest-cov coveralls pep8
    

Next, run the tests with the following commands:

::

    pep8 activitysim
    py.test --cov activitysim --cov-report term-missing
    coveralls

These same tests are run by Travis with each push to the repository.  These tests need to pass in order
to merge the revisions into master.

In some cases, test targets need to be updated to match the new results produced by the code since these 
are now the correct results.  In order to update the test targets, first determine which tests are 
failing and then review the failing lines in the source files.  These are easy to identify since each 
test ultimately comes down to one of Python's various types of `assert` statements.  Once you identify 
which `assert` is failing, you can work your way back through the code that creates the test targets in 
order to update it.  After updating the test targets, re-run the tests to confirm the new code passes all 
the tests.

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

When pushing revisions to the repo, the documentation is automatically built by Travis after 
successfully passing the tests.  The documents are built with the ``bin/build_docs.sh`` script.  
The script does the following:

* installs the required python packages
* runs ``make html``
* copies the ``master`` branch ``../activitysim/docs/_build/html/*`` pages to the ``gh-pages`` branch

GitHub automagically publishes the gh-pages branch at https://udst.github.io/activitysim.  

Releases
--------

ActivitySim releases are manually uploaded to the `Python Package Index <https://pypi.python.org/pypi/activitysim>`__  (pypi). 
