
Getting Started
===============

This page describes how to get started with ActivitySim.

.. note::
   ActivitySim is under development

.. index:: installation

Installation
------------

The following installation instructions are for Windows or Mac users.  ActivitySim is built
to run on one machine with sufficient available RAM to fit all required data and calculations
in memory.

Anaconda
~~~~~~~~

ActivitySim is a 64bit Python 2.7 library that uses a number of packages from the
scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__ 
and `numpy <http://numpy.org>`__. ActivitySim does not currently support Python 3.
   
The recommended way to get your own scientific Python installation is to
install Anaconda_, which contains many of the libraries upon which
ActivitySim depends + some handy Python installation management tools.  

Anaconda includes the ``conda`` command line tool, which does a number of useful 
things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__ 
(i.e. stand-alone Python installations/instances/sandboxes) that are the recommended 
way to work with multiple versions of Python on one machine.  Using conda 
environments keeps multiple Python setups from conflicting with one another.

After installing Anaconda, create an ActivitySim test environment 
with the following commands:

::
    
    #Windows
    conda create -n asimtest python=2.7
    activate asimtest

    #Mac
    conda create -n asimtest python=2.7
    source activate asimtest

If you access the internet from behind a firewall, then you will need to configure your proxy 
server. To do so, create a ``.condarc`` file in your Anaconda installation folder, such as:
::
   proxy_servers:
     http: http://proxynew.odot.state.or.us:8080
     https: https://proxynew.odot.state.or.us:8080
   ssl_verify: false

This will create a new conda environment named ``asimtest`` and set it as the 
active conda environment.  You need to activate the environment each time you
start a new command session.  You can remove an environment with 
``conda remove -n asimtest --all`` and check the current active environment with
``conda info -e``.

Dependencies
~~~~~~~~~~~~

ActivitySim depends on the following libraries, some of which* are pre-installed
with Anaconda:

* `numpy <http://numpy.org>`__ >= 1.12.0 \*
* `pandas <http://pandas.pydata.org>`__ >= 0.20.3 \*
* `pyyaml <http://pyyaml.org/wiki/PyYAML>`__ >= 3.0 \*
* `tables <http://www.pytables.org/moin>`__ >= 3.3.0 \*
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ or
  `cytoolz <https://github.com/pytoolz/cytoolz>`__ >= 0.7 \*
* `psutil <https://pypi.python.org/pypi/psutil>`__ >= 4.1
* `zbox <https://pypi.python.org/pypi/zbox>`__ >= 1.2
* `orca <https://udst.github.io/orca>`__ >= 1.1
* `openmatrix <https://pypi.python.org/pypi/OpenMatrix>`__ >= 0.2.4

To install the dependencies with conda, first make sure to activate the correct
conda environment and then install each package using pip_.  Pip will 
attempt to install any dependencies that are not already installed.  

::    
    
    #required packages for running ActivitySim
    pip install cytoolz numpy pandas tables pyyaml psutil
    pip install orca openmatrix zbox
    
    #optional required packages for testing and building documentation
    pip install pytest pytest-cov coveralls pep8 pytest-xdist
    pip install sphinx numpydoc sphinx_rtd_theme

If numexpr (which numpy requires) fails to install, you may need 
the `Microsoft Visual C++ Compiler for Python <http://aka.ms/vcpython27>`__. 

If you access the internet from behind a firewall, then you will need to configure 
your proxy server when downloading packages.  For example:
::
   pip install --trusted-host pypi.python.org --proxy=proxynew.odot.state.or.us:8080  cytoolz

ActivitySim
~~~~~~~~~~~

The current ``release`` version of ActivitySim can be installed 
from `PyPI <https://pypi.python.org/pypi/activitysim>`__  as well using pip_.  
The development version can be installed directly from the source.

Release
^^^^^^^

::
    
    #new install
    pip install activitysim

    #update to a new release
    pip install -U activitysim

Development
^^^^^^^^^^^

The development version of ActivitySim can be installed as follows:

* Clone or fork the source from the `GitHub repository <https://github.com/udst/activitysim>`__
* Activate the correct conda environment if needed
* Navigate to your local activitysim git directory
* Run the command ``python setup.py develop``

The ``develop`` command is required in order to make changes to the 
source and see the results without reinstalling.  You may need to first uninstall the
the pip installed version before installing the development version from source.  This is 
done with ``pip uninstall activitysim``.

.. _Anaconda: http://docs.continuum.io/anaconda/index.html
.. _conda: http://conda.pydata.org/
.. _pip: https://pip.pypa.io/en/stable/

.. _expressions_in_detail :

Expressions
-----------

Much of the power of ActivitySim comes from being able to specify Python, pandas, and 
numpy expressions for calculations. Refer to the pandas help for a general 
introduction to expressions.  ActivitySim provides two ways to evaluate expressions:

* Simple table expressions are evaluated using ``DataFrame.eval()``.  `pandas' eval <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.eval.html>`__ operates on the current table.
* Python expressions, denoted by beginning with ``@``, are evaluated with `Python's eval() <https://docs.python.org/2/library/functions.html#eval>`__.

Simple table expressions can only refer to columns in the current DataFrame.  Python expressions can refer
to any Python objects currently in memory.  

Conventions
~~~~~~~~~~~

There are a few conventions for writing expressions in ActivitySim:

* each expression is applied to all rows in the table being operated on
* expressions must be vectorized expressions and can use most numpy and pandas expressions
* global constants are specified in the settings file
* comments are specified with ``#``
* you can refer to the current table being operated on as ``df``
* often an object called ``skims``, ``skims_od``, or similar is available and is used to lookup the relevant skim information.  See :ref:`skims_in_detail` for more information.
* when editing the CSV files in Excel, use single quote ' or space at the start of a cell to get Excel to accept the expression

Example Expressions File
~~~~~~~~~~~~~~~~~~~~~~~~

An expressions file has the following basic form:

+---------------------------------+-------------------------------+-----------+----------+
| Description                     |  Expression                   |     cars0 |    cars1 |
+=================================+===============================+===========+==========+
| 2 Adults (age 16+)              |  drivers==2                   |         0 |   3.0773 |
+---------------------------------+-------------------------------+-----------+----------+
| Persons age 35-34               |  num_young_adults             |         0 |  -0.4849 |
+---------------------------------+-------------------------------+-----------+----------+
| Number of workers, capped at 3  |  @df.workers.clip(upper=3)    |         0 |   0.2936 |
+---------------------------------+-------------------------------+-----------+----------+
| Distance, from 0 to 1 miles     |  @skims['DIST'].clip(1)       |   -3.2451 |  -0.9523 |
+---------------------------------+-------------------------------+-----------+----------+

* Rows are vectorized expressions that will be calculated for every record in the current table being operated on
* The Description column describes the expression
* The Expression column contains a valid vectorized Python/pandas/numpy expression.  In the example above, ``drivers`` is a column in the current table.  Use ``@`` to refer to data outside the current table
* There is a column for each alternative and its relevant coefficient

There are some variations on this setup, but the functionality is similar.  For example, 
in the example destination choice model, the size terms expressions file has market segments as rows and employment type 
coefficients as columns.  Broadly speaking, there are currently four types of model expression configurations:

* simple choice model - select from a fixed set of choices defined in the specification file, such as the example above
* destination choice model - combine the destination choice expressions with the destination choice alternatives files since the alternatives are not listed in the expressions file
* complex choice model - an expressions file, a coefficients file, and a YAML settings file with model structural definition.  The mode models are examples of this and are illustrated below
* combinatorial choice model - first generate a set of alternatives based on a combination of alternatives across choosers, and then make choices.  The CDAP model implements this approach as illustrated below

The :ref:`mode_choice` model is a complex choice model since the expressions file is structured a little bit differently, as shown below.  
Each row is an expression for one of the alternatives, and each column is the coefficient for a tour purpose.  The alternatives are specified in the YAML settings file for the model.  
In the example below, the ``@odt_skims['SOV_TIME'] + dot_skims['SOV_TIME']`` expression is travel time for the tour origin to desination at the tour start time plus the tour
destination to tour origin at the tour end time.  The ``odt_skims`` and ``dot_skims`` objects are setup ahead-of-time to refer to the relevant skims for this model.
The tour mode choice model is a nested logit (NL) model and the nesting structure (including nesting coefficients) is specified in the YAML settings file as well.

+----------------------------------------+-------------------------------------------------+----------------------+-----------+----------+
| Description                            |  Expression                                     |     Alternative      |   school  | shopping |
+========================================+=================================================+======================+===========+==========+ 
|DA - Unavailable                        | sov_available == False                          |  DRIVEALONEFREE      |         0 |   3.0773 | 
+----------------------------------------+-------------------------------------------------+----------------------+-----------+----------+ 
|DA - In-vehicle time                    | @odt_skims['SOV_TIME'] + dot_skims['SOV_TIME']  |  DRIVEALONEFREE      |         0 |  -0.4849 | 
+----------------------------------------+-------------------------------------------------+----------------------+-----------+----------+ 
|DAP - Unavailable for age less than 16  | age < 16                                        |  DRIVEALONEPAY       |         0 |   0.2936 | 
+----------------------------------------+-------------------------------------------------+----------------------+-----------+----------+ 
|DAP - Unavailable for joint tours       | is_joint                                        |  DRIVEALONEPAY       | -3.2451   |  -0.9523 | 
+----------------------------------------+-------------------------------------------------+----------------------+-----------+----------+ 

In ActivitySim, all models are implemented as a series of table operations.  The :ref:`cdap` model sequence of vectorized table operations is:

* create a person level table and rank each person in the household for inclusion in the CDAP model
* solve individual M/N/H utilities for each person
* take as input an interaction coefficients table and then programatically produce and write out the expression files for households size 1, 2, 3, 4, and 5 models independent of one another
* select households of size 1, join all required person attributes, and then read and solve the automatically generated expressions
* repeat for households size 2, 3, 4, and 5. Each model is independent of one another.

Example
-------

The next logical step in getting started is to run the :ref:`example`.
