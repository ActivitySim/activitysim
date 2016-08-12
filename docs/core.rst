Core Utilities
==============

ActivitySim's core components include the model specification reader, 
the expression evaluator, the skim matrix manager, the simulate methods for solving
models (i.e. calculating utilities, probabilties, and making choices),  the choice 
models such as :ref:`mnl_in_detail`, and the tracer.

Activitysim
------------

API
~~~

.. automodule:: activitysim.activitysim
   :members:

.. _skims_in_detail :

Skim
------------

Skim Abstractions

API
~~~

.. automodule:: activitysim.skim
   :members:

.. _mnl_in_detail :

MNL
------------

Multinomial logit choice model. See `choice models 
<http://tfresource.org/Category:Choice_models>`__
for more information.

API
~~~

.. automodule:: activitysim.mnl
   :members:
   
Tracing
------------

Household tracer 

API
~~~

.. automodule:: activitysim.tracing
   :members:

Asim_Eval
------------

Alternative version of the expression evaluator in ``activitysim`` that supports temporary variable assignment.  
This is used by the :py:func:`~activitysim.defaults.models.compute_accessibility`  module.

API
~~~

.. automodule:: activitysim.asim_eval
   :members:


Utilities
------------

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
