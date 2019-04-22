
Core Components
===============

ActivitySim's core components include features for multiprocessing, data management, 
utility expressions, choice models, person time window management, and helper 
functions.  These core components include the multiprocessor, skim matrix manager, the 
data pipeline manager, the random number manager, the tracer, sampling 
methods, simulation methods, model specification readers and expression 
evaluators, choice models, timetable, and helper functions.

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

.. _skims_in_detail:

Skim
~~~~

Skim matrix data access

API
^^^

.. automodule:: activitysim.core.skim
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

Simple table expressions can only refer to columns in the current DataFrame.  Python expressions can refer to any Python objects 
urrently in memory.

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

* Simple :ref:`simulate` choice model - select from a fixed set of choices defined in the specification file, such as the example above.
* :ref:`simulate_with_interaction` choice model - combine the choice expressions with the choice alternatives files since the alternatives are not listed in the expressions file.  The :ref:`non_mandatory_tour_destination_choice` model implements this approach.
* Complex choice model - an expressions file, a coefficients file, and a YAML settings file with model structural definition.  The :ref:`tour_mode_choice` models are examples of this and are illustrated below.
* Combinatorial choice model - first generate a set of alternatives based on a combination of alternatives across choosers, and then make choices.  The :ref:`cdap` model implements this approach.

The :ref:`tour_mode_choice` model is a complex choice model since the expressions file is structured a little bit differently, as shown below.  
Each row is an expression for one of the alternatives, and each column contains either -999, 1, or blank.  The coefficients for each expression
is in a separate file, with a separate column for each alternative.  In the example below, the ``@c_ivt*(@odt_skims['SOV_TIME'] + dot_skims['SOV_TIME'])`` 
expression is travel time for the tour origin to desination at the tour start time plus the tour destination to tour origin at the tour end time.  
The ``odt_skims`` and ``dot_skims`` objects are setup ahead-of-time to refer to the relevant skims for this model.  The ``@c_ivt`` comes from the
tour mode choice coefficient file.  The tour mode choice model is a nested logit (NL) model and the nesting structure (including nesting 
coefficients) is specified in the YAML settings file.

+----------------------------------------+----------------------------------------------------------+-----------------+---------------+
| Description                            |  Expression                                              | DRIVEALONEFREE  | DRIVEALONEPAY |
+========================================+==========================================================+=================+===============+ 
|DA - Unavailable                        | sov_available == False                                   |            -999 |               | 
+----------------------------------------+----------------------------------------------------------+-----------------+---------------+ 
|DA - In-vehicle time                    | @c_ivt*(odt_skims['SOV_TIME'] + dot_skims['SOV_TIME'])   |               1 |               |
+----------------------------------------+----------------------------------------------------------+-----------------+---------------+ 
|DAP - Unavailable for age less than 16  | age < 16                                                 |                 |   -999        | 
+----------------------------------------+----------------------------------------------------------+-----------------+---------------+ 
|DAP - Unavailable for joint tours       | is_joint == True                                         |                 |   -999        | 
+----------------------------------------+----------------------------------------------------------+-----------------+---------------+ 

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
