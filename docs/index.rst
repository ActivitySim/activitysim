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

ActivitySim works with three open data formats, `HDF5 <https://www.hdfgroup.org/HDF5/>`__ 
, `Open Matrix (OMX) <https://github.com/osPlanning/omx>`__, and `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__ . 
The HDF5 binary data container is used for input tables such as land use inputs, synthetic 
population files, and accessibility files. HDF5 is also used for the :ref:`pipeline_in_detail` data store.
OMX, which is based on HDF5, is used for input and output matrices (skims and demand matrices).  CSV files 
are used for various inputs and outputs as well.

Three key data structures in ActivitySim are:

* `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__ - A data table with rows and columns, similar to an R data frame, Excel worksheet, or database table
* `pandas.Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`__ - a vector of data, a column in a DataFrame table or a 1D array
* `numpy.array <http://docs.scipy.org/doc/numpy/reference/arrays.html>`__ - an N-dimensional array of items of the same type, and is often a network skim matrix or collection of skim matrices by time-period or mode for example

Model Orchestrator
~~~~~~~~~~~~~~~~~~

An ActivitySim model is a sequence of model / data processing steps, commonly known as a data pipeline.
A well defined data pipeline has the ability to resume jobs at a known point, which facilitates 
debugging of problems with data and/or calculations.  It also allows for checkpointing model
resources, such as the state of each person at a point in the model simulation.  Checkpointing also
allows for regression testing of results at specified points in overall model run.

`ORCA <https://github.com/udst/orca>`__ is an orchestration/pipeline tool that defines model steps, 
dynamic data sources, and connects them to processing functions. ORCA defines dynamic data tables 
based on pandas DataFrames, columns based on pandas Series, and injectables (functions).  Model steps 
are executed as steps registered with the ORCA engine.  ActivitySim extends ORCA's functionality by
adding a :ref:`pipeline_in_detail`, that runs a series of ORCA model steps, manages the state of the data 
tables throughout the model run, allows for restarting at any model step, and integrates with the 
random number generation procedures (see :ref:`random_in_detail`).

Expressions
~~~~~~~~~~~

ActivitySim exposes most of its model expressions in CSV files.  These model expression CSV files
contain Python expressions, mainly pandas/numpy expression, and reference various input data tables 
and network skim matrices.  With this design, the Python code, which can be thought of as a generic expression 
engine, and the specific model calculations, such as the utilities, are separate.  This helps to avoid 
modifying the actual Python code when making changes to the models, such as during model calibration. An 
example of model expressions is found in the example auto ownership model specification file - 
`auto_ownership.csv <https://github.com/UDST/activitysim/blob/master/example/configs/auto_ownership.csv>`__. 
Refer to the :ref:`expressions_in_detail` section for more detail.

Choice Models
~~~~~~~~~~~~~

ActivitySim currently supports multinomial (MNL) and nested logit (NL) choice models. Refer to :ref:`logit_in_detail` 
for more information.  It also supports custom expressions as noted above, which can often be used to 
code additional types of choice models.  In addition, developers can write their own choice models 
in Python and expose these through the framework.  

Contents
--------

.. toctree::
   :maxdepth: 2

   gettingstarted
   abmexample
   howitworks
   dataschema
   core
   models
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
