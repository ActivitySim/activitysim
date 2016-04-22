
Getting Started
===============

This page describes how to install ActivitySim and setup and run the provided example.

.. index:: installation

Installation
------------

.. note::
   In the instructions below we will direct you to run various commands.
   On Mac and Linux these should go in your standard terminal.
   On Windows you may use the standard command prompt, the Anaconda
   command prompt, or Git Bash (if you have that installed).

Anaconda
~~~~~~~~

ActivitySim is a Python library that uses a number of packages from the
scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__ 
and pandas and `numpy <http://numpy.org>`__.  

The easiest way to get your own scientific Python installation is to
install Anaconda_, which contains many of the libraries upon which
ActivitySim depends.

Once you have a Python installation, install the dependencies listed below and
then install ActivitySim.

Dependencies
~~~~~~~~~~~~

ActivitySim depends on the following libraries, some of which* are pre-installed
with Anaconda:

* `numpy <http://numpy.org>`__ >= 1.8.0 \*
* `pandas <http://pandas.pydata.org>`__ >= 0.18.0 \*
* `pyyaml <http://pyyaml.org/wiki/PyYAML>`__ >= 3.0 \*
* `tables <http://www.pytables.org/moin>`__ >= 3.1.0 \*
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ or
  `cytoolz <https://github.com/pytoolz/cytoolz>`__ >= 0.7 \*
* `psutil <https://pypi.python.org/pypi/psutil>`__ >= 4.1
* `zbox <https://pypi.python.org/pypi/zbox>`__ >= 1.2
* `orca <https://udst.github.io/orca>`__ >= 1.1
* `openmatrix <https://pypi.python.org/pypi/OpenMatrix/0.2.3>`__ >= 0.2.2


ActivitySim
~~~~~~~~~~~

pip
^^^

ActivitySim can be installed from `PyPI <https://pypi.python.org/pypi/activitysim>`__ 
using pip_.  

::    

    pip install activitysim
  
Pip will attempt to install any dependencies that are not already installed.  If you
want to install the dependencies as well, the pip commands are:

::    
    
    pip install psutil orca openmatrix zbox
    
    #required packages for testing
    pip install pytest pytest-cov coveralls pep8
    
    #required packages for building documentation
    pip install sphinx numpydoc
    
To update to a new release of ActivitySim use the -U option with pip install:

::    

    pip install -U activitysim

Development
^^^^^^^^^^^

Development versions of ActivitySim can be installed as follows:

* Download the source from the `GitHub repo <https://github.com/udst/activitysim>`__
* ``cd`` into the ``activitysim`` directory 
* Run the command ``python setup.py develop``

The ``develop`` command is required in order to make changes to the 
source and see the results without reinstalling.

.. _Anaconda: http://docs.continuum.io/anaconda/index.html
.. _conda: http://conda.pydata.org/
.. _pip: https://pip.pypa.io/en/stable/

.. _expressions_in_detail :

Expressions
------------

Much of the power of ActivitySim comes from being able to specify Python, pandas, and 
numpy expressions for calculations. Refer to the pandas help for a general 
introduction to expressions.  ActivitySim provides two ways to evaluate expressions:

* Simple table expressions are evaluated using ``DataFrame.eval()``.  `pandas' eval <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.eval.html>`__ operates on the current table and is multi-threaded.
* Python expressions, denoted by beginning with the ``@``, are evaluated with `Python's eval() <https://docs.python.org/2/library/functions.html#eval>`__.

Conventions
~~~~~~~~~~~

Here are a few conventions for writing expressions in ActivitySim.

* each expression is applied to all rows in the table being operated on
* expressions must be vectorized expressions and can use most numpy and pandas expressions
* global constants are specified in the settings file
* comments are specified with ``#``
* you can refer to the current table as ``df``
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
| Distance, from 0 to 1 miles     |  @skims['DISTANCE'].clip(1)   | -3.2451   |  -0.9523 |
+---------------------------------+-------------------------------+-----------+----------+

* Rows are vectorized expressions that will be calculated for every record in the current table
* A Description column to describe the expression
* An Expression column with a valid vectorized Python/pandas/numpy expression.  In the example above, ``drivers`` is a column in the current table.  Use ``@`` to refer to data outside the current table
* A column for each alternative and its relevant coefficient

There are some variations on this setup, but the functionality is similar.  For example, 
in the destination choice model, the size terms expressions file has market segments as rows and employment type 
coefficients as columns.  Broadly speaking, there are currently three types of model expression configurations:

* simple choice model - selects from a fixed set of choices defined in the specification file, such as the example above
* destination choice model - which combines the destination choice expressions and destination choice alternatives files
* complex choice model - a CSV expressions file with description, a CSV coefficient file, and a YAML settings file, such as the example below

The tour mode choice model expressions file is structured a little differently, as shown below.  Each row is an expression for one
alternative and columns are for tour purposes.  The alternatives, as well as template expressions such as 
``$IN_N_OUT_EXPR.format(sk='SOV_TIME')`` are specified in the YAML settings file for the model.

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

.. index:: tutorial
.. index:: example

Example
-------

This section describes how to setup and run the example, as well as how the example works.

Folder/File Setup
~~~~~~~~~~~~~~~~~

The example has the following root folder/file setup:

  * configs - settings, expressions files, etc.
  * data - input data such as land use, synthetic population files, and skims
  * simulation.py - main script to run the model
    
Inputs
~~~~~~

In order to run the example, you first need two input data files in the ``data`` folder:

* mtc_asim.h5 - an HDF5 file containing the following MTC TM1 tables as pandas DataFrames for a subset of zones:

    * skims/accessibility - Zone-based accessibility measures
    * land_use/taz_data - Zone-based land use data (population and employment for example)
    * persons - Synthetic population person records
    * households - Synthetic population household records
    
* nonmotskm.omx - an OMX matrix file containing the MTC TM1 skim matrices for a subset of zones.

Both of these files can be downloaded from the `SF 25 zone example` example data folder on 
MTC's `box account <https://mtcdrive.app.box.com/v/activitysim>`__.  Both files can 
be viewed with the `OMX Viewer <https://github.com/osPlanning/omx/wiki/OMX-Viewer>`__.
The pandas DataFrames are stored in an efficient pandas format within the HDF5 file so they are a 
bit cumbersome to inspect. 

The ``scripts\data_mover.ipynb`` was used to create the mtc_asim.h5 file from the raw CSV files.  
This script reads the CSV files, creates DataFrame indexes, and writes the pandas objects to the HDF5 
file.

The full set of MTC TM1 OMX skims are also on the box account. The ``scripts\build_omx.py`` script 
will build one OMX file containing all the skims. The original MTC TM1 skims were converted from 
Cube to OMX using the `Cube to OMX converter <https://github.com/osPlanning/omx/wiki/Cube-OMX-Converter>`__.

Finally, the example inputs were created by the ``scripts\create_sf_example.py`` script,
which creates the land use, synthetic population, and skim inputs for a subset of user-defined zones.

Configuration
~~~~~~~~~~~~~

The ``configs`` folder contains settings, expressions files, and other files required for specifying 
model utilities and form.  The first place to start in the ``configs`` folder is ``settings.yaml``, which 
is the main settings file for the model run.  This file includes:

* store - HDF5 input file and also output file
* global variables that can be used in expressions tables and Python code such as:

    * urban_threshold - urban threshold area type max value
    * county_map - mapping of county codes to county names
    
* households_sample_size: 1000 - household sample size 
* time_periods - time period upper bound values and labels

Model Specification Files
^^^^^^^^^^^^^^^^^^^^^^^^^

Also stored in the configuration folder are the model specification files that store the 
Python/pandas/numpy expressions, alternatives, and other settings used by each model.  Some models includes an 
alternatives file since the alternatives are not easily described as columns in the expressions file.  An example
of this is the non_mandatory_tour_frequency_alternatives.csv file, which lists each alternative as a row and each 
columns indicates the number of non-mandatory tours by purpose.

The current set of files are:

* ``auto_ownership.csv`` - auto ownership model
* ``cdap_*.csv`` - CDAP model
* ``destination_choice.csv, destination_choice_size_terms.csv`` - destination choice model
* ``mandatory_tour_frequency.csv`` - mandatory tour frequency model
* ``non_mandatory_tour_frequency.csv, non_mandatory_tour_frequency_alternatives.csv`` - non mandatory tour frequency model
* ``school_location.csv`` - school location model
* ``tour_departure_and_duration_alternatives.csv, tour_departure_and_duration_nonmandatory.csv, tour_departure_and_duration_school.csv, tour_departure_and_duration_work.csv`` - tour departure and duration model
* ``tour_mode_choice.csv, tour_mode_choice.yaml, tour_mode_choice_coeffs.csv`` - tour mode choice model
* ``trip_mode_choice.csv, trip_mode_choice.yaml, trip_mode_choice_coeffs.csv`` - trip mode choice model
* ``workplace_location.csv`` - work location model

Running the Model
~~~~~~~~~~~~~~~~~

To run the example, do the following:

* Open a command line window in the ``example`` folder
* Ensure running ``python`` will call the Anaconda Python install on your machine
* Run ``python simulation.py``
* ActivitySim will print some logging information.  The example should complete within a couple minutes since it is running a small sample of households.

Outputs
~~~~~~~

There are currently no outputs produced by the example other than logging. 


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
    return pd.HDFStore(os.path.join(data_dir, "data", settings["store"]),mode='r')

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
    def households(set_random_seed, store, settings):
    
  @orca.column("households")
  def income_in_thousands(households):
    return households.income / 1000
  
The first model run is school location, which is called via the following command.  The ``@orca.step()`` decorator registers
the function as runnable by orca.

::

  orca.run(["school_location_simulate"])

  @orca.step()
  def school_location_simulate(set_random_seed, persons_merged,
    school_location_spec, skims, destination_size_terms):

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
  def households(set_random_seed, store, settings):
    if "households_sample_size" in settings:
        return asim.random_rows(store["households"],
                                settings["households_sample_size"])
    return store["households"]
  
  #households calls asim.random_rows to read a sample of households records 
  #from the households table in the HDF5 data store

``school_location_simulate`` then sets the persons merged table as choosers, reads the destination_size_terms 
alternatives file, and reads the expressions specification file. 

Next the method sets up the skims required for this model.
The following code set the keys for looking up the skim values for this model. In this case there is a ``TAZ`` column in the choosers,
which was in the ``households`` table that was joined with ``persons`` to make ``persons_merged`` and a ``TAZ`` in the alternatives 
generation code which get merged during interaction as renamed ``TAZ_r``.  The skims are lazy loaded under the name 
"skims" and are available in the expressions using ``@skims``.

::

    skims.set_keys("TAZ", "TAZ_r")
    locals_d = {"skims": skims}

The next step is to call ``asim.interaction_simulate`` function which run a simulation in which alternatives 
must be merged with choosers because there are interaction terms or because alternatives are being sampled.  The choosers table, the
alternatives table, the model specification expressions file, the skims, and the sample size are all passed in.  

:: 
  
  asim.interaction_simulate(choosers_segment, alternatives, spec[[school_type]],
    skims=skims, locals_d=locals_d, sample_size=50)


This function solves the utilities, calculates probabilities, draws random numbers, selects choices, and returns a column of choices. 
The ``eval_variables`` loops through each expression and solves it at once for all records in the chooser table using 
either pandas' multi-threaded eval() or Python's eval().

If the expression is a skim matrix, then the entire column of chooser OD pairs is retrieved from the matrix (i.e. numpy array) 
in one vectorized step.  The ``orig`` and ``dest`` objects in ``self.data[orig, dest]`` in ``activitysim.skim.py`` are vectors
and selecting numpy array items with vector indexes returns a vector.

:: 

    # evaluate variables from the spec
    model_design = eval_variables(spec.index, df, locals_d)
    
    # multiply by coefficients and reshape into choosers by alts
    utilities = model_design.dot(spec).astype('float')

    # convert to probabilities and make choices
    probs = utils_to_probs(utilities)
    positions = make_choices(probs)

    # positions come back between zero and num alternatives in the sample - need to get back to the indexes
    offsets = np.arange(len(positions)) * sample_size
    choices = model_design.index.take(positions + offsets)
    
    return pd.Series(choices, index=choosers.index), model_design

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
   def distance_to_school(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.school_taz),
                     index=persons.index)
   
   @orca.column("persons_school")
   def roundtrip_auto_time_to_school(persons, sovam_skim, sovmd_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.school_taz) +
                     sovmd_skim.get(persons.school_taz,
                                    persons.home_taz),
                     index=persons.index)

Any orca columns that are required are calculated-on-the-fly, such as ``roundtrip_auto_time_to_school`` as a 
function of the ``sovam_skim`` orca injectable.

The rest of the models operate in a similar fashion with two notable additions:

* creating new tables
* using 3D skims instead of skims (which is 2D)

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
  

Skims3D
~~~~~~~

The mode choice model uses the Skims3D class in addition to the skims (2D) class.  The Skims3D class represents 
a collection of skims with a third dimension, which in this case in time period.  Indexing Skims3D is done as follows:

::

    in_skims = askim.Skims3D(skims.set_keys("TAZ", "workplace_taz"),"in_period", -1)
    out_skims = askim.Skims3D(skims.set_keys("workplace_taz", "TAZ"),"out_period", -1)

Then when model expressions such as ``@in_skims['WLK_LOC_WLK_TOTIVT']`` are solved,
the ``WLK_LOC_WLK_TOTIVT`` skim matrix values for all chooser table origins, destinations, and 
in_periods can be looked-up in one request.

Dpending on the settings, Skims3D can either get the requested OMX data from disk every time 
a vectorized request is made or pre-load (cache) all the skims at the beginning of a model run.

See :ref:`skims_in_detail` for more information on skim handling.

