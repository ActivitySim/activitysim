Core Components
===============

ActivitySim's core components include the model specification reader and expression 
evaluator, the skim matrix manager, the simulate methods for solving
models (i.e. calculating utilities, probabilties, and making choices),  the choice 
models, the data pipeline manager, the random number generator, and the tracer.

Simulate
--------

Methods for expression handling, solving, choosing (i.e. making choices), and for identifying chunks

API
~~~

.. automodule:: activitysim.core.simulate
   :members:

.. _skims_in_detail:

Skim
----

Skim matrix data access

API
~~~

.. automodule:: activitysim.core.skim
   :members:

.. _logit_in_detail:

Logit
-----

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
~~~

.. automodule:: activitysim.core.logit
   :members:

.. _pipeline_in_detail:

Pipeline
--------

Data pipeline manager, which manages the list of model steps, runs them via orca, reads 
and writes data tables from/to the pipeline datastore, and supports restarting of the pipeline
at any model step.

API
~~~

.. automodule:: activitysim.core.pipeline
   :members:

.. _random_in_detail:

Random
------

ActivitySim's random number generation has a number of important features unique to AB modeling:

* Regression testing, debugging - run the exact model with the same inputs and get exactly the same results.
* Debugging models - run the exact model with the same inputs but with changes to expression files and get the same results except where the equations differ.
* Since runs can take a while, the above cases need to work with a restartable pipeline.
* Debugging Multithreading - run the exact model with different multithreading configurations and get the same results.
* Repeatable household-level choices - results for a household are repeatable when run with different sample sizes
* Repeatable household level results with different scenarios - results for a household are repeatable with different scenario configurations sequentially up to the point at which those differences emerge, and in alternate submodels in which those differences do not apply.

Random number generation is done using the `numpy Mersenne Twister PNRG <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html>`__.  
ActivitySim seeds on-the-fly and uses a stream of random numbers seeded by the household id, person id, tour id, trip id, the model step offset, and the global seed.  
The equation for calculating the seed looks something along the lines of:

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
~~~

.. automodule:: activitysim.core.random
   :members:

Tracing
-------

Household tracer.  If a household trace ID is specified, then ActivitySim will output a 
comprehensive set of trace files for all calculations for all household members:

* ``hhtrace.log`` - household trace log file, which specifies the CSV files traced. The order of output files is consistent with the model sequence.
* ``various CSV files`` - every input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for every sub-model

With the set of output CSV files, the user can trace ActivitySim's calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.



API
~~~

.. automodule:: activitysim.core.tracing
   :members:

Assign
------

Alternative version of the expression evaluators in :mod:`activitysim.core.simulate` that supports temporary variable assignment.  
This is used by the :py:func:`~activitysim.abm.models.accessibility.compute_accessibility`  module.

API
~~~

.. automodule:: activitysim.core.assign
   :members:


Utilities
---------

Vectorized helper functions

API
~~~

.. automodule:: activitysim.core.util
   :members:

Config
------

Helper functions for configuring a model run

API
~~~

.. automodule:: activitysim.core.config
   :members:


Inject_Defaults
---------------

Default file and folder settings

API
~~~

.. automodule:: activitysim.core.inject_defaults
   :members:

Tests
-----

See activitysim.core.tests
