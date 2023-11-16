Run the Primary Example
=======================

To setup and run the primary example (see :ref:`examples`) from the command line interface, do the following:

* Open the mamba or conda command prompt
* If you installed ActivitySim using conda environments, activate the conda
  environment with ActivitySim installed (i.e. ``conda activate asim``)
* Or, if you used the :ref:`pre-packaged installer<Pre-packaged Installer>`,
  replace all the commands below that call ``activitysim ...`` with the complete
  path to your installed location, which is probably something
  like ``c:\programdata\activitysim\scripts\activitysim.exe``.
* Type ``activitysim create -e prototype_mtc -d test_prototype_mtc`` to copy
  the very small prototype_mtc example to a new test_prototype_mtc directory
* Change to the test_prototype_mtc directory
* Type ``activitysim run -c configs -o output -d data`` to run the example
* Review the outputs in the output directory

.. note::
   Common configuration settings can be overridden at runtime.  See ``activitysim -h``, ``activitysim create -h`` and ``activitysim run -h``.
   ActivitySim model runs can be configured with settings file inheritance to avoid duplicating settings across model configurations.  See :ref:`cli` for more information.

Additional examples, including the full scale prototype MTC regional demand model, estimation integration examples, multiple zone system examples,
and examples for agency partners are available for creation by typing ``activitysim create -l``.  To create these examples, ActivitySim downloads the (large) input files from
the `ActivitySim resources <https://github.com/rsginc/activitysim_resources>`__ repository.  See :ref:`examples` for more information.