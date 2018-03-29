
.. _example :

Example
=======

This page describes how to setup and run the example AB model, as well as how it works.  The example
is limited to a small sample of households and zones so that it can be run quickly and require 
less than 1 GB of RAM.

.. index:: tutorial
.. index:: example

Folder/File Setup
-----------------

The example has the following root folder/file setup:

  * configs - settings, expressions files, etc.
  * data - input data such as land use, synthetic population files, and skims
  * simulation.py - main script to run the model
    
Inputs
------

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
-------------

The ``configs`` folder contains settings, expressions files, and other files required for specifying 
model utilities and form.  The first place to start in the ``configs`` folder is ``settings.yaml``, which 
is the main settings file for the model run.  This file includes:

* ``models`` - list of model steps to run - auto ownership, tour frequency, etc. - see :ref:`model_steps`
* ``store`` - HDF5 inputs file
* ``skims_file`` - skim matrices in one OMX file
* ``households_sample_size`` - number of households to sample and simulate; comment out to simulate all households
* ``trace_hh_id`` - trace household id; comment out for no trace
* ``trace_od`` - trace origin, destination pair in accessibility calculation; comment out for no trace
* ``chunk_size`` - batch size for processing choosers, see :ref:`chunk_size`
* ``check_for_variability`` - disable check for variability in an expression result debugging feature in order to speed-up runtime
* global variables that can be used in expressions tables and Python code such as:

    * ``urban_threshold`` - urban threshold area type max value
    * ``county_map`` - mapping of county codes to county names
    * ``skim_time_periods`` - time period upper bound values and labels

.. index:: chunk_size
.. _chunk_size:

Chunk size
~~~~~~~~~~

The ``chunk_size`` is the number of doubles in a chunk of the choosers table.  It is the number of rows 
times the number of columns and it needs to be set to a value that efficiently processes the table with 
the available RAM.  For example, a chunk size of 1,000,000 could be 100,000 households with 10 attribute 
columns.  Setting the chunk size too high will run into memory errors such as ``OverflowError: Python int 
too large to convert to C long.`` Setting the chunk size too low may result in smaller than optimal vector
lengths, which may waste runtime.  The chunk size is dependent on the size of the population, the complexity 
of the utility expressions, the amount of RAM on the machine, and other problem specific dimensions.  Thus, 
it needs to be set via experimentation.  

Logging
~~~~~~~

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging 
library and defines two key log files: 

* ``asim.log`` - overall system log file
* ``hhtrace.log`` - household trace log file if tracing is on

Refer to the :ref:`tracing` section for more detail on tracing.

Model Specification Files
~~~~~~~~~~~~~~~~~~~~~~~~~

Included in the ``configs`` folder are the model specification files that store the 
Python/pandas/numpy expressions, alternatives, and other settings used by each model.  Some models includes an 
alternatives file since the alternatives are not easily described as columns in the expressions file.  An example
of this is the ``non_mandatory_tour_frequency_alternatives.csv`` file, which lists each alternative as a row and each 
columns indicates the number of non-mandatory tours by purpose.  The current set of files are below.

+------------------------------------------------+----------------------------------------------------+
|            Model                               |    Specification Files                             |
+================================================+====================================================+
|  :ref:`accessibility`                          |  - accessibility.yaml                              |
|                                                |  - accessibility.csv                               |
+------------------------------------------------+----------------------------------------------------+
|   :ref:`school_location`                       |  - school_location.yaml                            |
|                                                |  - school_location_sample.csv                      |
|                                                |  - logsums_spec_school.csv                         |
|                                                |  - logsums_spec_university.csv                     |
|                                                |  - school_location.csv                             |
+------------------------------------------------+----------------------------------------------------+
|    :ref:`work_location`                        |  - workplace_location.yaml                         |
|                                                |  - workplace_location_sample.csv                   |
|                                                |  - logsums_spec_work.csv                           |
|                                                |  - workplace_location.csv                          |
+------------------------------------------------+----------------------------------------------------+
| :ref:`auto_ownership`                          |  - auto_ownership.yaml                             |
|                                                |  - auto_ownership.csv                              |
+------------------------------------------------+----------------------------------------------------+
|        :ref:`cdap`                             |  - cdap.yaml                                       |
|                                                |  - cdap_indiv_and_hhsize1.csv                      |
|                                                |  - cdap_interaction_coefficients.csv               |
|                                                |  - cdap_fixed_relative_proportions.csv             |
+------------------------------------------------+----------------------------------------------------+
|  :ref:`mandatory_tour_frequency`               |  - mandatory_tour_frequency.csv                    |
|                                                |  - mandatory_tour_frequency_alternatives.csv       |
+------------------------------------------------+----------------------------------------------------+
| :ref:`mandatory_tour_scheduling`               |  - tour_departure_and_duration_alternatives.csv    |
|                                                |  - tour_departure_and_duration_school.csv          |
|                                                |  - tour_departure_and_duration_work.csv            |
+------------------------------------------------+----------------------------------------------------+
| :ref:`non_mandatory_tour_frequency`            |  - non_mandatory_tour_frequency.csv                |
|                                                |  - non_mandatory_tour_frequency_alternatives.csv   |
+------------------------------------------------+----------------------------------------------------+
| :ref:`non_mandatory_tour_destination_choice`   |  - non_mandatory_tour_destination_choice.yaml      |
|                                                |  - non_mandatory_tour_destination_choice.csv       |
+------------------------------------------------+----------------------------------------------------+
| :ref:`non_mandatory_tour_scheduling`           |  - tour_departure_and_duration_alternatives.csv    |
|                                                |  - tour_departure_and_duration_nonmandatory.csv    |
+------------------------------------------------+----------------------------------------------------+
| :ref:`annotate_table` model_name=annotate_tours|  - annotate_tours.csv                              |
|                                                |  - annotate_tours.yaml                             |
+------------------------------------------------+----------------------------------------------------+
|        :ref:`tour_mode_choice`                 |  - tour_mode_choice.yaml                           |
|                                                |  - tour_mode_choice.csv                            |
|                                                |  - tour_mode_choice_coeffs.csv                     |
+------------------------------------------------+----------------------------------------------------+
|  :ref:`atwork_subtour_frequency`               |  - atwork_subtour_frequency.csv                    |
|                                                |  - atwork_subtour_frequency_alternatives.csv       |
+------------------------------------------------+----------------------------------------------------+
|   :ref:`atwork_subtour_destination`            |  - atwork_subtour_destination.yaml                 |
|                                                |  - atwork_subtour_destination_sample.csv           |
|                                                |  - atwork_subtour_destination.csv                  |
|                                                |  - logsums_spec_work.csv                           |
+------------------------------------------------+----------------------------------------------------+
| :ref:`atwork_subtour_scheduling`               |  - tour_departure_and_duration_subtour.csv         |
|                                                |  - tour_departure_and_duration_alternatives.csv    |
+------------------------------------------------+----------------------------------------------------+
|        :ref:`atwork_subtour_mode_choice`       |  - tour_mode_choice.yaml                           |
|                                                |  - tour_mode_choice.csv                            |
|                                                |  - tour_mode_choice_coeffs.csv                     |
+------------------------------------------------+----------------------------------------------------+
|      :ref:`trip_mode_choice`                   |  - trip_mode_choice.yaml                           |
|                                                |  - trip_mode_choice.csv                            |
|                                                |  - trip_mode_choice_coeffs.csv                     |
+------------------------------------------------+----------------------------------------------------+

Running the Example Model
-------------------------

To run the example, do the following:

* Open a command line window in the ``example`` folder
* Activate the correct conda environment if needed
* Run ``python simulation.py`` to run the data pipeline (i.e. model steps)
* ActivitySim should log some information and write outputs to the ``outputs`` folder.  

The example should complete within a couple minutes since it is running a small sample of households.

.. _model_steps :

Pipeline
--------

The ``models`` setting contains the specification of the data pipeline model steps, as shown below:

::

  models:
    - initialize
    - compute_accessibility
    - school_location_sample
    - school_location_logsums
    - school_location_simulate
    - workplace_location_sample
    - workplace_location_logsums
    - workplace_location_simulate
    - auto_ownership_simulate
    - cdap_simulate
    - mandatory_tour_frequency
    - mandatory_tour_scheduling
    - non_mandatory_tour_frequency
    - non_mandatory_tour_destination_choice
    - non_mandatory_tour_scheduling
    - annotate_table.model_name=annotate_tours
    - tour_mode_choice_simulate
    - atwork_subtour_frequency
    - atwork_subtour_destination_sample
    - atwork_subtour_destination_logsums
    - atwork_subtour_destination_simulate
    - atwork_subtour_scheduling
    - atwork_subtour_mode_choice
    - create_simple_trips
    - trip_mode_choice_simulate

These model steps must be registered orca steps, as noted below.  If you provide a ``resume_after`` 
argument to :func:`activitysim.core.pipeline.run` the pipeliner will load checkpointed tables from the checkpoint store 
and resume pipeline processing on the next model step after the specified checkpoint.  

::

  resume_after = None
  #resume_after = 'school_location_logsums'

The model is run by calling the :func:`activitysim.core.pipeline.run` method.

::

  pipeline.run(models=_MODELS, resume_after=resume_after)

Outputs
-------

The key output of ActivitySim is the HDF5 data pipeline file ``outputs\pipeline.h5``.  This file contains the 
state of the key data tables after each model step in which the table was modified.  The 
``pd.io.pytables.HDFStore('output\pipeline.h5')`` command returns the following information about 
the datastore.  You can see that the number of columns changes as each model step is run.  The checkpoints
table stores the crosswalk between model steps and table states in order to reload tables for restarting
the pipeline at any step.

+----------------------------------------------------------------------+-------+-------------------+
| Table                                                                | Type  | [Rows, Columns]   |
+======================================================================+=======+===================+ 
| /checkpoints                                                         | frame |  (shape->[25,13]) |
+----------------------------------------------------------------------+-------+-------------------+
| /land_use/initialize                                                 | frame |  (shape->[25,49]) |
+----------------------------------------------------------------------+-------+-------------------+
| /accessibility/school_location_sample                                | frame |  (shape->[25,20]) |
+----------------------------------------------------------------------+-------+-------------------+
| /households/initialize                                               | frame |  (shape->[100,64])|
+----------------------------------------------------------------------+-------+-------------------+
| /households/auto_ownership_simulate                                  | frame |  (shape->[100,67])|
+----------------------------------------------------------------------+-------+-------------------+
| /households/cdap_simulate                                            | frame |  (shape->[100,68])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/initialize                                                  | frame |  (shape->[154,50])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/school_location_simulate                                    | frame |  (shape->[154,53])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/cdap_simulate                                               | frame |  (shape->[154,63])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/mandatory_tour_frequency                                    | frame |  (shape->[154,68])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/non_mandatory_tour_frequency                                | frame |  (shape->[154,73])|
+----------------------------------------------------------------------+-------+-------------------+
| /persons/workplace_location_simulate                                 | frame |  (shape->[154,58])|
+----------------------------------------------------------------------+-------+-------------------+
| /person_windows/initialize                                           | frame |  (shape->[154,21])|
+----------------------------------------------------------------------+-------+-------------------+
| /person_windows/mandatory_tour_scheduling                            | frame |  (shape->[154,21])|
+----------------------------------------------------------------------+-------+-------------------+
| /person_windows/non_mandatory_tour_scheduling                        | frame |  (shape->[154,21])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/mandatory_tour_frequency                                      | frame |  (shape->[79,11]) |
+----------------------------------------------------------------------+-------+-------------------+
| /tours/mandatory_tour_scheduling                                     | frame |  (shape->[79,15]) |
+----------------------------------------------------------------------+-------+-------------------+
| /tours/non_mandatory_tour_frequency                                  | frame |  (shape->[168,15])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/non_mandatory_tour_destination_choice                         | frame |  (shape->[168,15])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/non_mandatory_tour_scheduling                                 | frame |  (shape->[168,15])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/annotate_table.model_name=annotate_tours                      | frame |  (shape->[168,41])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/tour_mode_choice_simulate                                     | frame |  (shape->[168,42])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/atwork_subtour_frequency                                      | frame |  (shape->[173,44])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/atwork_subtour_destination_simulate                           | frame |  (shape->[173,44])|
+----------------------------------------------------------------------+-------+-------------------+
| /tours/atwork_subtour_scheduling                                     | frame |  (shape->[173,44])|
+----------------------------------------------------------------------+-------+-------------------+
| /trips/create_simple_trips                                           | frame |  (shape->[336,8]) |
+----------------------------------------------------------------------+-------+-------------------+
| /trips/trip_mode_choice_simulate                                     | frame |  (shape->[336,9]) |
+----------------------------------------------------------------------+-------+-------------------+
| /school_location_sample/school_location_sample                       | frame |  (shape->[168,5]) |
+----------------------------------------------------------------------+-------+-------------------+
| /school_location_sample/school_location_logsums                      | frame |  (shape->[168,6]) |
+----------------------------------------------------------------------+-------+-------------------+
| /workplace_location_sample/workplace_location_sample                 | frame |  (shape->[2196,4])|
+----------------------------------------------------------------------+-------+-------------------+
| /workplace_location_sample/workplace_location_logsums                | frame |  (shape->[2196,5])|
+----------------------------------------------------------------------+-------+-------------------+
| /atwork_subtour_destination_sample/atwork_subtour_destination_sample | frame |  (shape->[77,6])  |
+----------------------------------------------------------------------+-------+-------------------+
| /atwork_subtour_destination_sample/atwork_subtour_destination_logsums| frame |  (shape->[77,7])  |
+----------------------------------------------------------------------+-------+-------------------+

The example ``simulation.py`` run model script also writes the final table to a CSV file
for illustrative purposes by using the :func:`activitysim.core.pipeline.get_table` method.  This method
returns a pandas DataFrame, which can then be written to a CSV with the ``to_csv(file_path)`` method.

ActivitySim also writes log and trace files to the ``outputs`` folder.  The asim.log file, which
is the overall log file is always produced.  If tracing is specified, then trace files are output
as well.

.. _tracing :

Tracing
-------

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
