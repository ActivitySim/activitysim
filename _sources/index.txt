.. ActivitySim documentation master file, created by
   sphinx-quickstart on Tue May 26 14:13:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ActivitySim
===========

ActivitySim is an open platform for activity-based travel modeling. It emerged
from a consortium of Metropolitan Planning Organizations (MPOs) and other
transportation planning agencies that wanted to build a shared, open, platform
that could be easily adapted to their individual needs, but would share a
robust, efficient, and well-maintained common core.

Additional information about the ActivitySim development effort is on the  
`GitHub project wiki <https://github.com/udst/activitysim/wiki>`__.

Software Design
---------------

The core software components of ActivitySim are described below.  ActivitySim is
implemented in Python, and makes heavy use of the vectorized backend C/C++ libraries in 
`pandas <http://pandas.pydata.org>`__  and `numpy <http://numpy.org>`__.  The core design 
principle of the system is vectorization of for loops, and this principle 
is woven into the system wherever reasonable.  As a result, the Python portions of the software 
can be thought of as more of an orchestrator, data processor, etc. that integrates a series of 
C/C++ vectorized data table and matrix operations.  The model system formulates 
each simulation as a series of vectorized table operations and the Python layer 
is responsible for setting up and providing expressions to operate on these large data tables.

Data Handling
~~~~~~~~~~~~~

ActivitySim operates on two data formats, `HDF5 <https://www.hdfgroup.org/HDF5/>`__ 
and `Open Matrix (OMX) <https://github.com/osPlanning/omx>`__. 
The HDF5 binary data container is used for managing flat files, including land use 
inputs, synthetic population files, and accessibility files. 
OMX, which is based on HDF5 as well, is used for managing network skims. 

Three key data structures in ActivitySim are:

* `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__ - A data table with rows and columns, similar to an R data frame, Excel worksheet, or database table
* `pandas.Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`__ - a vector of data, a column in a DataFrame table or a 1D array
* `numpy.array <http://docs.scipy.org/doc/numpy/reference/arrays.html>`__ - an N-dimensional array of items of the same type, and is often a network skim matrix or collection of skim matrices by time-period or mode for example

Model Orchestrator
~~~~~~~~~~~~~~~~~~

`ORCA <https://github.com/udst/orca>`__ is an orchestration/pipeline tool that defines 
dynamic data sources and connects them to processing functions.  ORCA is used for running 
the overall model system and for defining dynamic data tables (based on pandas DataFrames), 
columns ((based on pandas Series), and injectables (functions).  Model steps are executed 
as steps registered with ORCA.

Expressions
~~~~~~~~~~~

ActivitySim exposes most of its model expressions in CSV files that contain Python 
expressions, mainly pandas/numpy expression that operate on the input data tables and skims. 
This helps to avoid having to modify Python code when making changes to the model calculations. 
An example of model expressions is found in the example auto ownership model specification file - 
`auto_ownership.csv <https://github.com/UDST/activitysim/blob/master/example/configs/auto_ownership.csv>`__. 
Refer to the :ref:`expressions_in_detail` section for more detail.

Contents
--------

.. toctree::
   :maxdepth: 2

   gettingstarted
   dataschema
   core
   models
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
