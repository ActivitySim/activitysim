Ways to Run the Model
=====================

This section describes the different ways by which an ActivitySim Model can be run.


Options to Execute a Model
--------------------------

Using the Command Line Interface
________________________________

ActivitySim's **Command Line Interface (CLI)** allows the user to create examples and run models through quick DOS commands. This functionality gives users a better way to distribute multiple examples/run the model. The CLI functionality also enables to run ActivitySim across different platforms -  Linux, Windows, and macOS.

Refer to :ref:`Command Line Interface` and :ref:`Command Line Tools` for more details on the syntax and various parameters that can be passed.

Refer to the :ref:`Run the primary example<Run the Primary Example>` section for a demonstration of how to create and run the prototype_mtc model using CLI.



Using Jupyter Notebook
______________________

ActivitySim includes a `Jupyter Notebook <https://jupyter.org>`__ recipe book with interactive examples.  To run a Jupyter notebook, do the following:

* Open a conda prompt and activate the conda environment with ActivitySim installed
* If needed, ``conda install jupyterlab`` so you can run jupyter notebooks
* Type ``jupyter notebook`` to launch the web-based notebook manager
* Navigate to the ``examples/prototype_mtc/notebooks`` folder and select a notebook to learn more:

  * `Getting started <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/getting_started.ipynb/>`__
  * `Summarizing results <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/summarizing_results.ipynb/>`__
  * `Testing a change in auto ownership <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/change_in_auto_ownership.ipynb/>`__
  * `Adding TNCs <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/adding_tncs.ipynb/>`__
  * `Memory usage <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/notebooks/memory_usage.ipynb/>`__

Running Select Components of the Model
--------------------------------------

<Todo: Placeholder section for linking to component documentation and configuration>

Advanced Configuration
----------------------

There are several ways to maximize the performance of the model run, either to be able to run the model within the given hardware limitations (such as available RAM) or to reduce the run times. This section describes the various options and settings available in ActivitySim to improve the model run performance.

Chunking
________

Multi-processing
________________

Sharrow
_______

Tracing
_______


