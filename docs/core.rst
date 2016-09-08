Core Utilities
==============

ActivitySim's core components include the model specification reader, 
the expression evaluator, the skim matrix manager, the simulate methods for solving
models (i.e. calculating utilities, probabilties, and making choices),  the choice 
models such as :ref:`nl_in_detail`, and the tracer.

Activitysim
------------

API
~~~

.. automodule:: activitysim.activitysim
   :members:

.. _skims_in_detail:

Skim
----

Skim Abstractions

API
~~~

.. automodule:: activitysim.skim
   :members:

.. _nl_in_detail:

Nested Logit
------------

Multinomial logit (MNL) or Nested logit (NL) choice model.  These choice models depend on the foundational components of ActivitySim, such
as the expressions and data handling described in the :ref:`how_the_system_works` section.

To specify and solve an MNL model:

* either specify ``LOGIT_TYPE: MNL`` in the model configuration YAML file or omit the setting
* call either ``asim.simple_simulate()`` or ``asim.interaction_simulate()`` depending if the alternatives are interacted with the choosers or because alternatives are sampled

To specify and solve an NL model:

* specify ``LOGIT_TYPE: NL`` in the model configuration YAML file
* specify the nesting structure via the NESTS setting in the model configuration YAML file.  An example nested logit NESTS entry can be found in 
``example/configs/tour_mode_choice.yaml``
* call ``asim.simple_simulate()``.  The ``asim.interaction_simulate()`` functionality is not yet supported for NL.

API
~~~

.. automodule:: activitysim.nl
   :members:
   
Tracing
-------

Household tracer 

API
~~~

.. automodule:: activitysim.tracing
   :members:

Asim_Eval
---------

Alternative version of the expression evaluator in ``activitysim`` that supports temporary variable assignment.  
This is used by the :py:func:`~activitysim.defaults.models.compute_accessibility`  module.

API
~~~

.. automodule:: activitysim.asim_eval
   :members:


Utilities
---------

Reindex
~~~~~~~

API
^^^

.. automodule:: activitysim.util.reindex
   :members:

Testing
~~~~~~~

API
^^^

.. automodule:: activitysim.util.testing
   :members:
