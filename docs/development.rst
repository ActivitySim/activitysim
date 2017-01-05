Development
===========

The purpose of this page is to document how to contribute to ActivitySim.

In developing this software platform, we strive to adhere to a best practices approach to scientific computing, 
as summarized in `this article. <http://www.plosbiology.org/article/info%3Adoi%2F10.1371%2Fjournal.pbio.1001745>`__

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

Working with Git and GitHub
---------------------------

* `GitHub Help <https://help.github.com>`__
* `GitHub Guides <https://guides.github.com>`__
* `Workflows <https://guides.github.com/introduction/flow>`__

Testing
-------

ActivitySim testing is done with three tools:

* `pep8 <http://pep8.readthedocs.org/en/latest/intro.html>`__, a tool to check Python code against the PEP8 style conventions
* `pytest <http://pytest.org/latest/>`__, a Python testing tool
* `coveralls <https://github.com/coagulant/coveralls-python>`__, a tool for measuring code coverage and publishing code coverage stats online

To run the test, first make sure the required packages are installed:

::

    pip install pytest pytest-cov coveralls pep8
    

Next, run the tests with the following commands:

::

    pep8 activitysim
    py.test --cov activitysim --cov-report term-missing
    coveralls
    
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
