Development
===========

The purpose of this page is to document how to test and document ActivitySim.  

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

Creating Tests
~~~~~~~~~~~~~~

To be completed later

    
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

    pip install sphinx numpydoc

Next, build the documentation in html format with the following command:

::

    make html

When pushing revisions to the repo, the documentation is automatically built by Travis after 
successfully passing the tests.  The documents are built with the ``bin/build_docs.sh`` script.  
The script does the following:

* installs the required python packages
* runs ``make html``
* copies the ``master`` branch ``../activitysim/docs/_build/html/*`` pages to the ``gh-pages`` branch

GitHub automagically publishes the gh-pages branch at https://udst.github.io/activitysim
