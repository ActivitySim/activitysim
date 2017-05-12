
Getting Started
===============

This page describes how to install ActivitySim and setup and run the included example AB model.

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
and `numpy <http://numpy.org>`__.  

.. note::
   ActivitySim does not currently support Python 3
   
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
    
This will create a new conda environment named ``asimtest`` and set it as the 
active conda environment.  You need to activate the environment each time you
start a new command session.  You can remove an environment with 
``conda remove -n asimtest --all`` and check the current active environment with
``conda info -e``.

Dependencies
~~~~~~~~~~~~

ActivitySim depends on the following libraries, some of which* are pre-installed
with Anaconda:

* `numpy <http://numpy.org>`__ >= 1.8.0 \*
* `pandas <http://pandas.pydata.org>`__ >= 0.18.0 \*
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

.. index:: tutorial
.. index:: example

Example
-------

This section describes how to setup and run the example AB model, as well as how it works.  The example
is limited to a small sample of households and zones so that it can be run quickly and require less than 1 GB of RAM.

Folder/File Setup
~~~~~~~~~~~~~~~~~

The example has the following root folder/file setup:

  * configs - settings, expressions files, etc.
  * data - input data such as land use, synthetic population files, and skims
  * simulation.py - main script to run the model
    
Inputs
~~~~~~

In order to run the example, you first need two input files in the ``data`` folder as identified in the ``configs\settings.yaml`` file:

* store: mtc_asim.h5 - an HDF5 file containing the following MTC travel model one tables as pandas DataFrames for a subset of zones:

    * skims/accessibility - Zone-based accessibility measures
    * land_use/taz_data - Zone-based land use data (population and employment for example)
    * persons - Synthetic population person records
    * households - Synthetic population household records
    
* skims_file: skims.omx - an OMX matrix file containing the MTC travel model one skim matrices for a subset of zones.

Both of these files are used in the tests as well and are available here ``activitysim\defaults\test\data``.  Alternatively, 
these files can be downloaded from the SF 25 zone example example data folder on 
MTC's `box account <https://mtcdrive.app.box.com/v/activitysim>`__.  Both files can 
be viewed with the `OMX Viewer <https://github.com/osPlanning/omx/wiki/OMX-Viewer>`__.
The pandas DataFrames are stored in an efficient pandas format within the HDF5 file so they are a 
bit cumbersome to inspect. 

The ``scripts\data_mover.ipynb`` was used to create the mtc_asim.h5 file from the raw CSV files.  
This script reads the CSV files, creates DataFrame indexes, and writes the pandas objects to the HDF5 
file.

The full set of MTC travel model one OMX skims are also on the box account. The ``scripts\build_omx.py`` script 
will build one OMX file containing all the skims. The original MTC travel model one skims were converted from 
Cube to OMX using the `Cube to OMX converter <https://github.com/osPlanning/omx/wiki/Cube-OMX-Converter>`__.

Finally, the example inputs were created by the ``scripts\create_sf_example.py`` script,
which creates the land use, synthetic population, and skim inputs for a subset of user-defined zones.

Configuration
~~~~~~~~~~~~~

The ``configs`` folder contains settings, expressions files, and other files required for specifying 
model utilities and form.  The first place to start in the ``configs`` folder is ``settings.yaml``, which 
is the main settings file for the model run.  This file includes:

* ``store`` - HDF5 input file and also output file
* ``skims_file`` - skim matrices in one OMX file
* ``households_sample_size`` - number of households to sample and simulate; comment out to simulate all households
* ``trace_hh_id`` - trace household id; comment out for no trace
* ``trace_od`` - trace origin, destination pair in accessibility calculation; comment out for no trace
* ``chunk_size`` - batch size for processing choosers
* ``check_for_variability`` - disable check for variability in an expression result debugging feature in order to speed-up runtime
* global variables that can be used in expressions tables and Python code such as:

    * ``urban_threshold`` - urban threshold area type max value
    * ``county_map`` - mapping of county codes to county names
    * ``time_periods`` - time period upper bound values and labels

Logging Files
^^^^^^^^^^^^^

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging 
library and defines two key log files: 

* ``asim.log`` - overall system log file
* ``hhtrace.log`` - household trace log file if tracing is on

Refer to the :ref:`tracing` section for more detail on tracing.

Model Specification Files
^^^^^^^^^^^^^^^^^^^^^^^^^

Included in the ``configs`` folder are the model specification files that store the 
Python/pandas/numpy expressions, alternatives, and other settings used by each model.  Some models includes an 
alternatives file since the alternatives are not easily described as columns in the expressions file.  An example
of this is the non_mandatory_tour_frequency_alternatives.csv file, which lists each alternative as a row and each 
columns indicates the number of non-mandatory tours by purpose.

The current set of files are:

* ``accessibility.csv, , accessibility.yaml`` - accessibility model
* ``auto_ownership.csv, auto_ownership.yaml`` - auto ownership model
* ``cdap_indiv_and_hhsize1.csv, cdap_interaction_coefficients.csv, cdap_fixed_relative_proportions.csv`` - CDAP model
* ``destination_choice.csv, destination_choice_size_terms.csv`` - destination choice model
* ``mandatory_tour_frequency.csv`` - mandatory tour frequency model
* ``non_mandatory_tour_frequency.csv, non_mandatory_tour_frequency_alternatives.csv`` - non mandatory tour frequency model
* ``school_location.csv`` - school location model
* ``tour_departure_and_duration_alternatives.csv, tour_departure_and_duration_nonmandatory.csv, tour_departure_and_duration_school.csv, tour_departure_and_duration_work.csv`` - tour departure and duration model
* ``tour_mode_choice.csv, tour_mode_choice.yaml, tour_mode_choice_coeffs.csv`` - tour mode choice model
* ``trip_mode_choice.csv, trip_mode_choice.yaml, trip_mode_choice_coeffs.csv`` - trip mode choice model
* ``workplace_location.csv`` - work location model

Running the Example Model
~~~~~~~~~~~~~~~~~~~~~~~~~

To run the example, do the following:

* Open a command line window in the ``example`` folder
* Activate the correct conda environment if needed
* Run ``python simulation.py`` to the run data pipeline (i.e. model steps)
* ActivitySim should log some information and write outputs to the ``outputs`` folder.  

The example should complete within a couple minutes since it is running a small sample of households.

Pipeline
~~~~~~~~

The ``simulation.py`` script contains the specification of the data pipeline model steps, as shown below:

::

  _MODELS = [
    'compute_accessibility',
    'school_location_simulate',
    'workplace_location_simulate',
    'auto_ownership_simulate',
    'cdap_simulate',
    'mandatory_tour_frequency',
    'mandatory_scheduling',
    'non_mandatory_tour_frequency',
    'destination_choice',
    'non_mandatory_scheduling',
    'tour_mode_choice_simulate',
    'create_simple_trips',
    'trip_mode_choice_simulate'
  ]

These model steps must be registered orca steps, as noted below.  If you provide a ``resume_after`` 
argument to :func:`activitysim.core.pipeline.run` the pipeliner will load checkpointed tables from the checkpoint store 
and resume pipeline processing on the next model step after the specified checkpoint.  

::

  resume_after = None
  #resume_after = 'mandatory_scheduling'

The model is run by calling the :func:`activitysim.core.pipeline.run` method.

::

  pipeline.run(models=_MODELS, resume_after=resume_after)

Outputs
~~~~~~~

The key output of ActivitySim is the HDF5 data pipeline file ``outputs\pipeline.h5``.  This file contains the 
state of the key data tables after each model step in which the table was modified.  The 
``pd.io.pytables.HDFStore('output\pipeline.h5')`` command returns the following information about 
the datastore.  You can see that the number of columns changes as each model step is run.  The checkpoints
table stores the crosswalk between model steps and table states in order to reload tables for restarting
the pipeline at any step.

+---------------------------------------------------+-------+-------------------+
| Table                                             | Type  | [Rows, Columns]   |
+===================================================+=======+===================+ 
| /checkpoints                                      | frame | (shape->[14,11])  |
+---------------------------------------------------+-------+-------------------+
| /accessibility/compute_accessibility              | frame | (shape->[25,21])  |
+---------------------------------------------------+-------+-------------------+
| /households/compute_accessibility                 | frame | (shape->[100,64]) |
+---------------------------------------------------+-------+-------------------+
| /households/auto_ownership_simulate               | frame | (shape->[100,67]) |
+---------------------------------------------------+-------+-------------------+
| /households/cdap_simulate                         | frame | (shape->[100,68]) |
+---------------------------------------------------+-------+-------------------+
| /land_use/compute_accessibility                   | frame | (shape->[25,49])  |
+---------------------------------------------------+-------+-------------------+
| /mandatory_tours/mandatory_tour_frequency         | frame | (shape->[77,4])   |
+---------------------------------------------------+-------+-------------------+
| /mandatory_tours/mandatory_scheduling             | frame | (shape->[77,5])   |
+---------------------------------------------------+-------+-------------------+
| /non_mandatory_tours/non_mandatory_tour_frequency | frame | (shape->[83,5])   |
+---------------------------------------------------+-------+-------------------+
| /non_mandatory_tours/destination_choice           | frame | (shape->[83,6])   |
+---------------------------------------------------+-------+-------------------+
| /non_mandatory_tours/non_mandatory_scheduling     | frame | (shape->[83,7])   |
+---------------------------------------------------+-------+-------------------+
| /persons/compute_accessibility                    | frame | (shape->[156,50]) |
+---------------------------------------------------+-------+-------------------+
| /persons/school_location_simulate                 | frame | (shape->[156,54]) |
+---------------------------------------------------+-------+-------------------+
| /persons/workplace_location_simulate              | frame | (shape->[156,59]) |
+---------------------------------------------------+-------+-------------------+
| /persons/cdap_simulate                            | frame | (shape->[156,64]) |
+---------------------------------------------------+-------+-------------------+
| /persons/mandatory_tour_frequency                 | frame | (shape->[156,69]) |
+---------------------------------------------------+-------+-------------------+
| /persons/non_mandatory_tour_frequency             | frame | (shape->[156,72]) |
+---------------------------------------------------+-------+-------------------+
| /tours/tour_mode_choice_simulate                  | frame | (shape->[160,38]) |
+---------------------------------------------------+-------+-------------------+
| /trips/create_simple_trips                        | frame | (shape->[320,8])  |
+---------------------------------------------------+-------+-------------------+
| /trips/trip_mode_choice_simulate                  | frame | (shape->[320,9])  |
+---------------------------------------------------+-------+-------------------+

The example ``simulation.py`` run model script also writes the final table to a CSV file
for illustrative purposes by using the :func:`activitysim.core.pipeline.get_table` method.  This method
returns a pandas DataFrame, which can then be written to a CSV with the ``to_csv(file_path)`` method.

ActivitySim also writes log and trace files to the ``outputs`` folder.  The asim.log file, which
is the overall log file is always produced.  If tracing is specified, then trace files are output
as well.

.. _tracing :

Tracing
~~~~~~~

There are two types of tracing in ActivtiySim: household and origin-destination (OD) pair.  If a household trace ID 
is specified, then ActivitySim will output a comprehensive set of trace files for all 
calculations for all household members:

* ``hhtrace.log`` - household trace log file, which specifies the CSV files traced. The order of output files is consistent with the model sequence.
* ``various CSV files`` - every input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for every sub-model

If an OD pair trace is specified, then ActivitySim will output the acessibility calculations trace 
file:

* ``accessibility.result.csv`` - accessibility expression results for the OD pair

With the set of output CSV files, the user can trace ActivitySim's calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.


