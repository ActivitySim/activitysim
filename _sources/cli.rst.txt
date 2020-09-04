
.. _cli :

Command Line Interface
======================

ActivitySim includes a :ref:`cli` for creating examples and running the model.  See ``activitysim -h`` for 
more information.

Create
------

Create an ActivitySim example setup.  See ``activitysim create -h`` for more information.
More complete examples, including the full scale MTC 
regional demand model are available for creation by typing ``activitysim create -l``.  To create 
these examples, ActivitySim downloads the large input files from 
the `ActivitySim resources <https://github.com/rsginc/activitysim_resources>`__ repository.

API
~~~

.. automodule:: activitysim.cli.create
   :members:


Run
---

Run ActivitySim.  See ``activitysim run -h`` for more information.

API
~~~

.. automodule:: activitysim.cli.run
   :members:

