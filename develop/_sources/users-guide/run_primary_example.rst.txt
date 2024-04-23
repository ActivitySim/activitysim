Run the Primary Example
=======================

To setup and run the primary example (see :ref:`examples`) from the command line interface, do the following:

* Open the mamba or conda command prompt
* If you installed ActivitySim using conda environments, activate the conda
  environment with ActivitySim installed (i.e. ``conda activate asim``)

.. note::
  Most ``conda`` and ``mamba`` commands can be used interchangeably in the **mamba command prompt**. In the above step, you could instead use ``mamba activate asim`` in the mamba command prompt.
* Or, if you used the :ref:`pre-packaged installer<Pre-packaged Installer>`,
  replace all the commands below that call ``activitysim ...`` with the complete
  path to your installed location, which is probably something
  like ``c:\programdata\activitysim\scripts\activitysim.exe``.
* Type ``activitysim create -e prototype_mtc -d test_prototype_mtc`` to copy
  the very small prototype_mtc example to a new test_prototype_mtc directory
* Change to the test_prototype_mtc directory ``cd test_prototype_mtc``
* Type ``activitysim run -c configs -o output -d data`` to run the example
* Review the outputs in the output directory
* ActivitySim will log progress and write outputs to the output folder.

The example should run in a few minutes since it runs a small sample of households.


.. note::
   Common configuration settings can be overridden at runtime.  See ``activitysim -h``, ``activitysim create -h`` and ``activitysim run -h``.
   ActivitySim model runs can be configured with settings file inheritance to avoid duplicating settings across model configurations.  See :ref:`cli` for more information.

Additional examples, including the full scale prototype MTC regional demand model, estimation integration examples, multiple zone system examples,
and examples for agency partners are available for creation by typing ``activitysim create -l``.  To create these examples, ActivitySim downloads the (large) input files from
the `ActivitySim resources <https://github.com/rsginc/activitysim_resources>`__ repository.  See :ref:`examples` for more information.