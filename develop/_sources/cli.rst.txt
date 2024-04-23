
.. _cli :

Command Line Interface
======================

ActivitySim includes a :ref:`cli` for creating examples and running the model.  See ``activitysim -h`` for 
more information.

.. note::
   The `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
   contains example commands to create and run several versions of the examples.

Create
------

Create an ActivitySim example setup.  See ``activitysim create -h`` for more information.
More complete examples, including the full scale prototype MTC
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

.. index:: _settings_file_inheritance
.. _settings_file_inheritance :

Settings File Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~

ActivitySim model runs can be configured with settings file inheritance to avoid 
duplicating settings across model configurations. The command below runs ActivitySim
with two configs folders - ``configs`` and ``configs_mp``.  This setup allows for overriding setings 
in the configs folder with additional settings in the configs_mp folder so that  
expression files and settings in the single process (e.g. configs folder) can be re-used for
the multiprocessed setup (e.g. configs_mp folder).  Settings files, as opposed to configs folders, 
can also be inherited by specifying ``-s`` multiple times.  See ``activitysim run -h`` for 
more information.

::

  # in configs_mp\settings.yaml
  inherit_settings: True

  #then on the command line
  activitysim run -c configs_mp -c configs -d data -o output


API
~~~

.. automodule:: activitysim.cli.run
   :members:

