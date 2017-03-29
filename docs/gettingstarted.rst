
Getting Started
===============

This page describes how to install ActivitySim and setup and run the included example.

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

ActivitySim is a Python 2.7 library that uses a number of packages from the
scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__ 
and pandas and `numpy <http://numpy.org>`__.  

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
* `tables <http://www.pytables.org/moin>`__ >= 3.1.0, <3.3.0 \*
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ or
  `cytoolz <https://github.com/pytoolz/cytoolz>`__ >= 0.7 \*
* `psutil <https://pypi.python.org/pypi/psutil>`__ >= 4.1
* `zbox <https://pypi.python.org/pypi/zbox>`__ >= 1.2
* `orca <https://udst.github.io/orca>`__ >= 1.1
* `openmatrix <https://pypi.python.org/pypi/OpenMatrix/0.2.3>`__ >= 0.2.2

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
    
If ``numexpr`` (which ``numpy`` requires) fails to install, you may need 
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
------------

Much of the power of ActivitySim comes from being able to specify Python, pandas, and 
numpy expressions for calculations. Refer to the pandas help for a general 
introduction to expressions.  ActivitySim provides two ways to evaluate expressions:

* Simple table expressions are evaluated using ``DataFrame.eval()``.  `pandas' eval <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.eval.html>`__ operates on the current table.
* Python expressions, denoted by beginning with the ``@``, are evaluated with `Python's eval() <https://docs.python.org/2/library/functions.html#eval>`__.

Conventions
~~~~~~~~~~~

Here are a few conventions for writing expressions in ActivitySim.

* each expression is applied to all rows in the table being operated on
* expressions must be vectorized expressions and can use most numpy and pandas expressions
* global constants are specified in the settings file
* comments are specified with ``#``
* you can refer to the current table as ``df``
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
| Distance, from 0 to 1 miles     |  @skims['DIST'].clip(1)       | -3.2451   |  -0.9523 |
+---------------------------------+-------------------------------+-----------+----------+

* Rows are vectorized expressions that will be calculated for every record in the current table
* A Description column to describe the expression
* An Expression column with a valid vectorized Python/pandas/numpy expression.  In the example above, ``drivers`` is a column in the current table.  Use ``@`` to refer to data outside the current table
* A column for each alternative and its relevant coefficient

There are some variations on this setup, but the functionality is similar.  For example, 
in the destination choice model, the size terms expressions file has market segments as rows and employment type 
coefficients as columns.  Broadly speaking, there are currently four types of model expression configurations:

* simple choice model - select from a fixed set of choices defined in the specification file, such as the example above
* destination choice model - combine the destination choice expressions with the destination choice alternatives files since the alternatives are not listed in the expressions file
* complex choice model - an expressions file, a coefficients file, and a YAML settings file with model structural definition.  The mode models are examples of this and are illustrated below
* combinatorial choice model - first generate a set of alternatives based on a combination of alternatives across choosers, and then make choices.  The CDAP model implements this approach as illustrated below

The :ref:`mode_choice` model is a complex choice model since the expressions file is structured a little bit differently, as shown below.  
Each row is an expression for one alternative and columns are for tour purposes.  The alternatives, as well as template expressions such as 
``$IN_N_OUT_EXPR.format(sk='SOV_TIME')`` are specified in the YAML settings file for the model.  The tour mode choice model is a nested logit (NL) model
and the nesting structure (including nesting coefficients) is specified in the YAML settings file as well.

+----------------------------------------+------------------------------------------+----------------------+-----------+----------+
| Description                            |  Expression                              |     Alternative      |   school  | shopping |
+========================================+==========================================+======================+===========+==========+ 
|DA - Unavailable                        | sov_available == False                   |  DRIVEALONEFREE      |         0 |   3.0773 | 
+----------------------------------------+------------------------------------------+----------------------+-----------+----------+ 
|DA - In-vehicle time                    | $IN_N_OUT_EXPR.format(sk='SOV_TIME')     |  DRIVEALONEFREE      |         0 |  -0.4849 | 
+----------------------------------------+------------------------------------------+----------------------+-----------+----------+ 
|DAP - Unavailable for age less than 16  | age < 16                                 |  DRIVEALONEPAY       |         0 |   0.2936 | 
+----------------------------------------+------------------------------------------+----------------------+-----------+----------+ 
|DAP - Unavailable for joint tours       | is_joint                                 |  DRIVEALONEPAY       | -3.2451   |  -0.9523 | 
+----------------------------------------+------------------------------------------+----------------------+-----------+----------+ 

The :ref:`cdap` model operates as a series of vectorized table operations:

* create a person level table and rank each person in the household for inclusion in the CDAP model
* solve individual M/N/H utilities for each person
* take as input an interaction coefficients table and then programatically produce and write out the expression files for households size 1, 2, 3, 4, and 5 models independent of one another
* select households of size 1, join all required person attributes, and then read and solve the automatically generated expressions
* repeat for households size 2, 3, 4, and 5. Each model is independent of one another.

.. index:: tutorial
.. index:: example

Example
-------

This section describes how to setup and run the example, as well as how the example works.  The example
is a small subset of households and zones and so it requires less than 1 GB of RAM to run.

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
these files can be downloaded from the ``SF 25 zone example`` example data folder on 
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
* Run ``python simulation.py``
* ActivitySim will print some logging information and write some outputs to the ``outputs`` folder.  

The example should complete within a couple minutes since it is running a small sample of households.

Outputs
~~~~~~~

ActivitySim writes log and trace files to the ``outputs`` folder.  The asim.log file, which
is the overall log file is always produced.  If tracing is specified, then trace files are output
as well.

.. note::
   Currently the example produces no standard outputs, such as trip lists.  The next 
   phase of development will address creating and writing of outputs.  In the 
   meantime, the example writes the in-memory households table to a CSV file 
   for illustrative purposes.

.. _tracing :

Tracing
~~~~~~~

There are two types of tracing in ActivtiySim: household and OD pair.  If a household trace ID 
is specified, then ActivitySim will output a comprehensive set of trace files for all 
calculations for all household members:

* ``hhtrace.log`` - household trace log file, which specifies the CSV files traced. The order of output files is consistent with the model sequence.
* ``various CSV files`` - every input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for every sub-model

If an OD pair trace is specified, then ActivitySim will output the acessibility calculations trace 
file:

* ``accessibility.result.csv`` - accessibility expression results for the OD pair

With the set of output CSV files, the user can trace ActivitySim's calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.

.. _how_the_system_works:

How the System Works
--------------------

This section describes ActivitySim's flow of execution.

The Basic Flow of Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example model run starts by running ``simulation.py``, which calls:

::

  import orca
  from activitysim import defaults 
  
which starts orca, which will now take over running the system and defines the orca/pandas tables and their data sources 
but does not load the data.  The second statement loads ``defaults.__init__``, which calls:

::

   import misc 
   import tables
   import models

which then loads the misc, tables, and models class definitions.  Loading ``misc`` defines orca injectables (functions) 
for the ``settings`` object based on the setting.yaml file and the ``store`` based on the HDF5 input file.  The
Python decorator ``@orca.injectable`` overrides the function definition ``store`` to execute this function 
whenever ``store`` is called by orca.

:: 

  @orca.injectable(cache=True)
  def store(data_dir, settings):
    return pd.HDFStore(os.path.join(data_dir, settings["store"]),mode='r')

Next, the following import statement define the dynamic orca tables households, persons, skims, etc., but does not load them.
It also defines the dynamic orca table columns (calculated fields) and injectables (functions) defined in the classes.  The
Python decorator ``@orca.table`` and ``@orca.column("households")`` override the function definitions so the function name
becomes the table name in the first case, whereas the function name becomes the column in the second case.  The argument to 
``households`` in ``@orca.column("households")`` is table (either real or virtual) that the column is added to.  

::

  import households
  import persons
  import skims
  #etc...
  
  @orca.table(cache=True)
    def households(store, settings):
    
  @orca.column("households")
  def income_in_thousands(households):
    return households.income / 1000
  
The first microsimulation model run is school location, which is called via the following command.  The ``@orca.step()`` decorator registers
the function as runnable by orca.

::

  orca.run(["school_location_simulate"])

  @orca.step()
  def school_location_simulate(
    persons_merged,
    school_location_spec, school_location_settings, 
    skims,
    destination_size_terms, 
    chunk_size, trace_hh_id):
                             
The ``school_location_simulate`` step requires the objects defined in the function definition above.  Since they are not yet loaded, 
orca goes looking for them.  This is called lazy loading (or on-demand loading).  The steps to get the persons data loaded is illustrated below.

::

  #persons_merged is in the step function signature

  @orca.table()
  def persons_merged(persons, households, land_use, accessibility):
    return orca.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
        
  #it required persons, households, land_use, accessibility
  @orca.table(cache=True)
  def persons(persons_internal):
      return persons_internal.to_frame()
      
  #persons requires persons_internal
  @orca.table(cache=True)
  def persons_internal(store, settings, households):
    df = store["persons"]
    if "households_sample_size" in settings:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]
    return df
  
  #persons_internal requires store, settings, households
  @orca.table(cache=True)
  def households(store, households_sample_size, trace_hh_id):

    df_full = store["households"]

    # if we are tracing hh exclusively
    if trace_hh_id and households_sample_size == 1:
      ...
    # if we need sample a subset of full store
    elif households_sample_size > 0 and len(df_full.index) > households_sample_size:
      ...
    else:
        df = df_full

    if trace_hh_id:
        tracing.register_households(df, trace_hh_id)
        tracing.trace_df(df, "households")

    return df
  
  #households calls asim.random_rows to read a sample of households records 
  #households calls tracing.register_households to setup tracing

``school_location_simulate`` also reads the expressions specification file, settings yaml file,
destination_size_terms file, sets the persons merged table as choosers, and sets the chunk size, trace id, and random seed. 

::

  def school_location_simulate(
    persons_merged,
    school_location_spec, school_location_settings, 
    skims,
    destination_size_terms, 
    chunk_size, trace_hh_id):
    
Next the method sets up the skims required for this model.
The following code set the keys for looking up the skim values for this model. In this case there is a ``TAZ`` column in the choosers,
which was in the ``households`` table that was joined with ``persons`` to make ``persons_merged`` and a ``TAZ`` in the alternatives 
generation code which get merged during interaction as renamed ``TAZ_r``.  The skims are lazy loaded under the name 
"skims" and are available in the expressions using ``@skims``.

::

    skims.set_keys("TAZ", "TAZ_r")
    locals_d = {"skims": skims}

The next step is to call ``asim.interaction_simulate`` function which run a MNL choice model simulation in which alternatives 
must be merged with choosers because there are interaction terms or because alternatives are sampled.  The choosers table, the
alternatives table, the model specification expressions file, the skims, and the sample size are all passed in.  

:: 
      
  asim.interaction_simulate(choosers_segment, alternatives, spec[[school_type]],
    skims=skims, locals_d=locals_d, sample_size=50, chunk_size=0, trace_label=None, trace_choice_name=None)

This function solves the utilities, calculates probabilities, draws random numbers, selects choices, and returns a column of choices. 
This is done in a for loop of chunks of choosers in order to avoid running out of RAM when building the often large data tables.
The ``eval_variables`` loops through each expression and solves it at once for all records in the chunked chooser table using 
either pandas' eval() or Python's eval().

The ``asim.interaction_simulate`` method is currently only a multinomial logit choice model.  The ``asim.simple_simulate`` method 
supports both MNL and NL as specified by the ``LOGIT_TYPE`` setting in the model settings YAML file.   The ``auto_ownership.yaml`` 
file for example specifies the ``LOGIT_TYPE`` as ``MNL.``

If the expression is a skim matrix, then the entire column of chooser OD pairs is retrieved from the matrix (i.e. numpy array) 
in one vectorized step.  The ``orig`` and ``dest`` objects in ``self.data[orig, dest]`` in ``activitysim.skim.py`` are vectors
and selecting numpy array items with vector indexes returns a vector.  Trace data is also written out if configured.

:: 

    # evaluate variables from the spec
    model_design = eval_variables(spec.index, choosers, locals_d)
    
    # multiply by coefficients and reshape into choosers by alts
    utilities = model_design.dot(spec)

    # convert to probabilities and make choices
    probs = utils_to_probs(utilities)
    choices = make_choices(probs)

    #write trace information
    if trace_label:
        #write trace information
    
    #return choices
    return choices

Finally, the model adds the choices as a column to the applicable table - ``persons`` - and adds 
additional dependent columns.  The dependent columns are those orca columns with the virtual table 
name ``persons_school``.

:: 

   orca.add_column("persons", "school_taz", choices)
   add_dependent_columns("persons", "persons_school")

   # columns to update after the school location choice model
   @orca.table()
   def persons_school(persons):
    return pd.DataFrame(index=persons.index)
    
   @orca.column("persons_school")
   def distance_to_school(persons, skim_dict):
       distance_skim = skim_dict.get('DIST')
       return pd.Series(distance_skim.get(persons.home_taz,
                                          persons.school_taz),
                        index=persons.index)
   
   @orca.column("persons_school")
   def roundtrip_auto_time_to_school(persons, skim_dict):
       sovmd_skim = skim_dict.get(('SOV_TIME', 'MD'))
       return pd.Series(sovmd_skim.get(persons.home_taz,
                                       persons.school_taz) +
                        sovmd_skim.get(persons.school_taz,
                                       persons.home_taz),
                        index=persons.index)

Any orca columns that are required are calculated-on-the-fly, such as ``roundtrip_auto_time_to_school``
which i turn uses skims from the skim_dict orca injectable.

The rest of the microsimulation models operate in a similar fashion with a few notable additions:

* creating new tables
* using 3D skims instead of skims (which is 2D)
* accessibilities

Creating New Tables
~~~~~~~~~~~~~~~~~~~

The mandatory tour frequency model sets the ``persons.mandatory_tour_frequency`` column.  Once the number of tours
is known, then the next step is to create tours records for subsequent models.  This is done with the following code,
which requires the ``persons`` table and returns a new pandas DataFrame which is registered as an 
orca table named ``mandatory_tours``.

::

  @orca.table(cache=True)
  def mandatory_tours(persons):
    persons = persons.to_frame(columns=["mandatory_tour_frequency","is_worker"])
    persons = persons[~persons.mandatory_tour_frequency.isnull()]
    return process_mandatory_tours(persons)
  
  #processes the mandatory_tour_frequency column that comes out of the model 
  #and turns into a DataFrame that represents the mandatory tours that were generated
  def process_mandatory_tours(persons):
    #...
    return pd.DataFrame(tours, columns=["person_id", "tour_type", "tour_num"])
  
.. _Skims_3D :

Skims3dWrapper
~~~~~~~~~~~~~~

The mode choice model uses the Skims3dWrapper class in addition to the skims (2D) class.  The Skims3dWrapper class represents
a collection of skims with a third dimension, which in this case in time period.  Setting up the 3D index for 
Skims3dWrapper is done as follows:

::

  #setup two indexes - tour inbound skims and tour outbound skims
  in_skims = askim.Skims3dWrapper(stack=stack, left_key=orig_key, right_key=dest_key, skim_key="in_period", offset=-1)
  out_skims = askim.Skims3dWrapper(stack=stack, left_key=dest_key, right_key=orig_key, skim_key="out_period", offset=-1)
    
  #where:
  stack = askim.SkimStack(skims)       #build 3D skim object from 2D skims table object
  orig_key = 'TAZ'                     #TAZ column
  dest_key = 'destination'             #destination column
  skim_key="in_period" or "out_period" #in_period or out_period column

When model expressions such as ``@in_skims['WLK_LOC_WLK_TOTIVT']`` are solved,
the ``WLK_LOC_WLK_TOTIVT`` skim matrix values for all chooser table origins, destinations, and 
in_periods can be retrieved in one request.

Depending on the settings, Skims3D can either get the requested OMX data from disk every time 
a vectorized request is made or preload (cache) all the skims at the beginning of a model run.  
Preload is faster and is the default.

See :ref:`skims_in_detail` for more information on skim handling.

Accessibilities
~~~~~~~~~~~~~~~~~~~

Unlike the microsimulation models, which operate on a table of choosers, the accessibilities model is 
an aggregate model that calculates accessibility measures by origin zone to all destination zones.  This 
model could be implemented with a matrix library such as ``numpy`` since it involves a series of matrix 
and vector operations.  However, all the other ActivitySim models - the 
microsimulation models - are implemented with ``pandas.DataFrame`` tables, and so this would be a 
different approach for just this model.  The benefits of keeping with the same table approach to 
data setup, expression management, and solving means ActivitySim has one expression syntax, is
easier to understand and document, and is more efficiently implemented.  

As illustrated below, in order to convert the 
accessibility calculation into a table operation, a table of OD pairs is first built using ``numpy``
``repeat`` and ``tile`` functions.  Once constructed, the additional data columns are added to the 
table in order to solve the accessibility calculations.  The ``skim`` data is also added in column form.
After solving the expressions for each OD pair row, the accessibility module aggregates the results
to origin zone and write them to the datastore.  

::

  # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(np.asanyarray(land_use_df.index), zone_count),
            'dest': np.tile(np.asanyarray(land_use_df.index), zone_count)
        }
    )
