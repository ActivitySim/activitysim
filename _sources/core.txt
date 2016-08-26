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

Nested or multinomial logit choice model.  These choice models depend on the foundational components of ActivitySim, such
as the expression and data handling described in the :ref:`how_the_system_works` section. To solve an MNL model, call 
``asim.simple_simulate()``. To solve an NL model, first specify the nesting structure via the NESTS setting in the 
model configuration file.  An example nested logit NESTS entry can be found in 
``example/configs/tour_mode_choice.yaml``.  With the NESTS defined, call ``asim.nested_simulate()`` to solve the 
model and make choices. 

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
