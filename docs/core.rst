
Core Components
===============

ActivitySim's core components include features for multiprocessing, data management, 
utility expressions, choice models, person time window management, and helper 
functions.  These core components include the multiprocessor, network LOS (skim) manager, the 
data pipeline manager, the random number manager, the tracer, sampling 
methods, simulation methods, model specification readers and expression 
evaluators, choice models, timetable, transit virtual path builder, and helper functions.

.. _multiprocessing_in_detail:

Multiprocessing
---------------

Parallelization using multiprocessing

API
~~~

.. automodule:: activitysim.core.mp_tasks
   :members:


Data Management
---------------

Input
~~~~~

Input data table functions

API
^^^

.. automodule:: activitysim.core.input
   :members:

.. _los_in_detail:

LOS
~~~

Network Level of Service (LOS) data access

API
^^^

.. automodule:: activitysim.core.los
   :members:

Skims
~~~~~

Skims data access

API
^^^

.. automodule:: activitysim.core.skim_dict_factory
   :members:

.. automodule:: activitysim.core.skim_dictionary
   :members:

.. _pipeline_in_detail:

Pipeline
~~~~~~~~

Data pipeline manager, which manages the list of model steps, runs them via orca, reads 
and writes data tables from/to the pipeline datastore, and supports restarting of the pipeline
at any model step.

API
^^^

.. automodule:: activitysim.core.pipeline
   :members:

.. _random_in_detail:

Random
~~~~~~

ActivitySim's random number generation has a number of important features unique to AB modeling:

* Regression testing, debugging - run the exact model with the same inputs and get exactly the same results.
* Debugging models - run the exact model with the same inputs but with changes to expression files and get the same results except where the equations differ.
* Since runs can take a while, the above cases need to work with a restartable pipeline.
* Debugging Multithreading - run the exact model with different multithreading configurations and get the same results.
* Repeatable household-level choices - results for a household are repeatable when run with different sample sizes
* Repeatable household level results with different scenarios - results for a household are repeatable with different scenario configurations sequentially up to the point at which those differences emerge, and in alternate submodels in which those differences do not apply.

Random number generation is done using the `numpy Mersenne Twister PNRG <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html>`__.  
ActivitySim seeds on-the-fly and uses a stream of random numbers seeded by the household id, person id, tour id, trip id, the model step offset, and the global seed.  
The logic for calculating the seed is something along the lines of:

::

  chooser_table.index * number_of_models_for_chooser + chooser_model_offset + global_seed_offset

  for example
    1425 * 2 + 0 + 1
  where:
    1425 = household table index - households.id
    2 = number of household level models - auto ownership and cdap
    0 = first household model - auto ownership
    1 = global seed offset for testing the same model under different random global seeds

ActivitySim generates a separate, distinct, and stable random number stream for each tour type and tour number in order to maintain as much stability as is 
possible across alternative scenarios.  This is done for trips as well, by direction (inbound versus outbound).

.. note::
   The Random module contains max model steps constants by chooser type - household, person, tour, trip - needs to be equal to the number of chooser sub-models.

API
^^^

.. automodule:: activitysim.core.random
   :members:

Tracing
~~~~~~~

Household tracer.  If a household trace ID is specified, then ActivitySim will output a 
comprehensive set of trace files for all calculations for all household members:

* ``hhtrace.log`` - household trace log file, which specifies the CSV files traced. The order of output files is consistent with the model sequence.
* ``various CSV files`` - every input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for every sub-model

With the set of output CSV files, the user can trace ActivitySim's calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.

API
^^^

.. automodule:: activitysim.core.tracing
   :members:


.. _expressions:

Utility Expressions
-------------------

Much of the power of ActivitySim comes from being able to specify Python, pandas, and 
numpy expressions for calculations. Refer to the pandas help for a general 
introduction to expressions.  ActivitySim provides two ways to evaluate expressions:

* Simple table expressions are evaluated using ``DataFrame.eval()``.  `pandas' eval <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.eval.html>`__ operates on the current table.
* Python expressions, denoted by beginning with ``@``, are evaluated with `Python's eval() <https://docs.python.org/2/library/functions.html#eval>`__.

Simple table expressions can only refer to columns in the current DataFrame.  Python expressions can refer to any Python objects currently in memory.

Conventions
~~~~~~~~~~~

There are a few conventions for writing expressions in ActivitySim:

* each expression is applied to all rows in the table being operated on
* expressions must be vectorized expressions and can use most numpy and pandas expressions
* global constants are specified in the settings file
* comments are specified with ``#``
* you can refer to the current table being operated on as ``df``
* often an object called ``skims``, ``skims_od``, or similar is available and is used to lookup the relevant skim information.  See :ref:`los_in_detail` for more information.
* when editing the CSV files in Excel, use single quote ' or space at the start of a cell to get Excel to accept the expression

Example Expressions File
~~~~~~~~~~~~~~~~~~~~~~~~

An expressions file has the following basic form:

+---------------------------------+---------------------------------+-------------------------------+-----------+---------------------------------+
| Label                           | Description                     |  Expression                   |     cars0 |    cars1                        |
+=================================+=================================+===============================+===========+=================================+
| util_drivers_2                  | 2 Adults (age 16+)              |  drivers==2                   |           |   coef_cars1_drivers_2          |
+---------------------------------+---------------------------------+-------------------------------+-----------+---------------------------------+
| util_persons_25_34              | Persons age 25-34               |  num_young_adults             |           |  coef_cars1_persons_25_34       |
+---------------------------------+---------------------------------+-------------------------------+-----------+---------------------------------+
| util_num_workers_clip_3         | Number of workers, capped at 3  |  @df.workers.clip(upper=3)    |           |   coef_cars1_num_workers_clip_3 |
+---------------------------------+---------------------------------+-------------------------------+-----------+---------------------------------+
| util_dist_0_1                   | Distance, from 0 to 1 miles     |  @skims['DIST'].clip(1)       |           |  coef_dist_0_1                  |
+---------------------------------+---------------------------------+-------------------------------+-----------+---------------------------------+

In the :ref:`tour_mode_choice` model expression file example shown below, the ``@c_ivt*(@odt_skims['SOV_TIME'] + dot_skims['SOV_TIME'])`` 
expression is travel time for the tour origin to destination at the tour start time plus the tour destination to tour origin at the tour end time.  
The ``odt_skims`` and ``dot_skims`` objects are setup ahead-of-time to refer to the relevant skims for this model.  The ``@c_ivt`` comes from the
tour mode choice coefficient file.  The tour mode choice model is a nested logit (NL) model and the nesting structure (including nesting 
coefficients) is specified in the YAML settings file.

+-----------------------------------------------------------+--------------------------------------------------------+------------------------------------------------+-----------------+---------------+
| Label                                                     | Description                                            |  Expression                                    | DRIVEALONEFREE  | DRIVEALONEPAY |
+===========================================================+========================================================+================================================+=================+===============+ 
| util_DRIVEALONEFREE_Unavailable                           | DRIVEALONEFREE - Unavailable                           | sov_available == False                         |            -999 |               | 
+-----------------------------------------------------------+--------------------------------------------------------+------------------------------------------------+-----------------+---------------+ 
| util_DRIVEALONEFREE_In_vehicle_time                       | DRIVEALONEFREE - In-vehicle time                       | odt_skims['SOV_TIME'] + dot_skims['SOV_TIME']  |       coef_ivt  |               |
+-----------------------------------------------------------+--------------------------------------------------------+------------------------------------------------+-----------------+---------------+ 
| util_DRIVEALONEFREE_Unavailable_for_persons_less_than_16  | DRIVEALONEFREE - Unavailable for persons less than 16  | age < 16                                       |            -999 |               | 
+-----------------------------------------------------------+--------------------------------------------------------+------------------------------------------------+-----------------+---------------+ 
| util_DRIVEALONEFREE_Unavailable_for_joint_tours           | DRIVEALONEFREE - Unavailable for joint tours           | is_joint == True                               |            -999 |               | 
+-----------------------------------------------------------+--------------------------------------------------------+------------------------------------------------+-----------------+---------------+ 

* Rows are vectorized expressions that will be calculated for every record in the current table being operated on
* The Label column is the unique expression name (used for model estimation integration)
* The Description column describes the expression
* The Expression column contains a valid vectorized Python/pandas/numpy expression.  In the example above, ``drivers`` is a column in the current table.  Use ``@`` to refer to data outside the current table
* There is a column for each alternative and its relevant coefficient from the submodel coefficient file

There are some variations on this setup, but the functionality is similar.  For example, 
in the example destination choice model, the size terms expressions file has market segments as rows and employment type 
coefficients as columns.  Broadly speaking, there are currently four types of model expression configurations:

* Simple :ref:`simulate` choice model - select from a fixed set of choices defined in the specification file, such as the example above.
* :ref:`simulate_with_interaction` choice model - combine the choice expressions with the choice alternatives files since the alternatives are not listed in the expressions file.  The :ref:`non_mandatory_tour_destination_choice` model implements this approach.
* Combinatorial choice model - first generate a set of alternatives based on a combination of alternatives across choosers, and then make choices.  The :ref:`cdap` model implements this approach.

Expressions
~~~~~~~~~~~

The expressions class is often used for pre- and post-processor table annotation, which read a CSV file of expression, calculate 
a number of additional table fields, and join the fields to the target table.  An example table annotation expressions 
file is found in the example configuration files for households for the CDAP model - 
`annotate_households_cdap.csv <https://github.com/activitysim/activitysim/blob/master/example/configs/annotate_households_cdap.csv>`__. 

.. automodule:: activitysim.core.expressions
   :members:

Sampling with Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for expression handling, solving, and sampling (i.e. making multiple choices), 
with interaction with the chooser table.  

Sampling is done with replacement and a sample correction factor is calculated.  The factor is 
calculated as follows:

:: 

  freq = how often an alternative is sampled (i.e. the pick_count)
  prob = probability of the alternative
  correction_factor = log(freq/prob)

  #for example:

  freq              1.00	2.00	3.00	4.00	5.00
  prob              0.30	0.30	0.30	0.30	0.30
  correction factor 1.20	1.90	2.30	2.59	2.81

As the alternative is oversampled, its utility goes up for final selection.  The unique set 
of alternatives is passed to the final choice model and the correction factor is 
included in the utility.

API
^^^

.. automodule:: activitysim.core.interaction_sample
   :members:

.. _simulate:

Simulate
~~~~~~~~

Methods for expression handling, solving, choosing (i.e. making choices) from a fixed set of choices 
defined in the specification file.

API
^^^

.. automodule:: activitysim.core.simulate
   :members:

.. _simulate_with_interaction:

Simulate with Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for expression handling, solving, choosing (i.e. making choices), 
with interaction with the chooser table.  

API
^^^

.. automodule:: activitysim.core.interaction_simulate
   :members:
   
Simulate with Sampling and Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods for expression handling, solving, sampling (i.e. making multiple choices), 
and choosing (i.e. making choices), with interaction with the chooser table.  

API
^^^

.. automodule:: activitysim.core.interaction_sample_simulate
   :members:

Assign
~~~~~~

Alternative version of the expression evaluators in :mod:`activitysim.core.simulate` that supports temporary variable assignment.  
Temporary variables are identified in the expressions as starting with "_", such as "_hh_density_bin".  These
fields are not saved to the data pipeline store.  This feature is used by the :ref:`accessibility` model.

API
^^^

.. automodule:: activitysim.core.assign
   :members:


Choice Models
-------------

.. _logit_in_detail:

Logit
~~~~~

Multinomial logit (MNL) or Nested logit (NL) choice model.  These choice models depend on the foundational components of ActivitySim, such
as the expressions and data handling described in the :ref:`how_the_system_works` section.

To specify and solve an MNL model:

* either specify ``LOGIT_TYPE: MNL`` in the model configuration YAML file or omit the setting
* call either ``simulate.simple_simulate()`` or ``simulate.interaction_simulate()`` depending if the alternatives are interacted with the choosers or because alternatives are sampled

To specify and solve an NL model:

* specify ``LOGIT_TYPE: NL`` in the model configuration YAML file
* specify the nesting structure via the NESTS setting in the model configuration YAML file.  An example nested logit NESTS entry can be found in ``example/configs/tour_mode_choice.yaml``
* call ``simulate.simple_simulate()``.  The ``simulate.interaction_simulate()`` functionality is not yet supported for NL.

API
^^^

.. automodule:: activitysim.core.logit
   :members:

.. _time_windows:

Person Time Windows
-------------------

The departure time and duration models require person time windows. Time windows are adjacent time 
periods that are available for travel. Time windows are stored in a timetable table and each row is 
a person and each time period (in the case of MTC TM1 is 5am to midnight in 1 hr increments) is a column. 
Each column is coded as follows:

* 0 - unscheduled, available
* 2 - scheduled, start of a tour, is available as the last period of another tour
* 4 - scheduled, end of a tour, is available as the first period of another tour
* 6 - scheduled, end or start of a tour, available for this period only
* 7 - scheduled, unavailable, middle of a tour

A good example of a time window expression is ``@tt.previous_tour_ends(df.person_id, df.start)``.  This 
uses the person id and the tour start period to check if a previous tour ends in the same time period.

API
~~~

.. automodule:: activitysim.core.timetable
   :members:

.. _transit_virtual_path_builder:

Transit Virtual Path Builder
----------------------------

Transit virtual path builder (TVPB) for three zone system (see :ref:`multiple_zone_systems`) transit path utility calculations.
TAP to TAP skims and walk access and egress times between MAZs and TAPs are input to the 
demand model.  ActivitySim then assembles the total transit path utility based on the user specified TVPB 
expression files for the respective components:

* from MAZ to first boarding TAP + 
* from first boarding to final alighting TAP + 
* from alighting TAP to destination MAZ

This assembling is done via the TVPB, which considers all the possible combinations of nearby boarding and alighting TAPs for each origin 
destination MAZ pair and selects the user defined N best paths to represent the transit mode.  After selecting N best paths, the logsum across
N best paths is calculated and exposed to the mode choice models and a random number is drawn and a path is chosen. The boarding TAP, 
alighting TAP, and TAP to TAP skim set for the chosen path is saved to the chooser table.  

The initialize TVPB submodel (see :ref:`initialize_los`) pre-computes TAP to TAP total utilities for the user defined attribute_segments, 
which are typically demographic segment (for example household income bin), time-of-day, and access/egress mode.  This submodel can be
run in both single process and multiprocess mode, with single process excellent for development/debugging and multiprocess excellent
for application.  ActivitySim saves the pre-calculated TAP to TAP total utilities to a memory mapped cache file for reuse by downstream models 
such as tour mode choice.  In tour mode choice, the pre-computed TAP to TAP total utilities for the attribute_segment, along with the 
access and egress impedances, are used to evaluate the best N TAP pairs for each origin MAZ destination MAZ pair being evaluated.  
Assembling the total transit path impedance and then picking the best N is quick since it is done in a de-duplicated manner within 
each chunk of multiprocessed choosers.

A model with TVPB can take considerably longer to run than a traditional TAZ based model since it does an order of magnitude more 
calculations.  Thus, it is important to be mindful of your approach to your network model as well, especially the number of TAPs 
accessible to each MAZ, which is the key determinant of runtime.  

API
~~~

.. automodule:: activitysim.core.pathbuilder
   :members:
   

Cache API
~~~~~~~~~

.. automodule:: activitysim.core.pathbuilder_cache
   :members:

Helpers
-------

.. _chunk_in_detail:

Chunk
~~~~~

Chunking management

API
^^^

.. automodule:: activitysim.core.chunk
   :members:

Utilities
~~~~~~~~~

Vectorized helper functions

API
^^^

.. automodule:: activitysim.core.util
   :members:

Config
~~~~~~

Helper functions for configuring a model run

API
^^^

.. automodule:: activitysim.core.config
   :members:

.. _inject:

Inject
~~~~~~

Wrap ORCA class to make it easier to track and manage interaction with the data pipeline.

API
^^^

.. automodule:: activitysim.core.inject
   :members:

Mem
~~~

Helper functions for tracking memory usage

API
^^^

.. automodule:: activitysim.core.mem
   :members:

Output
~~~~~~

Write output files and track skim usage.

API
^^^

.. automodule:: activitysim.core.steps.output
   :members:

Tests
~~~~~

See activitysim.core.test
