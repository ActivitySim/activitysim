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

Chunking is designed to split up large data records in Activitysim to enable processing them in batches such that memory does not get exhausted. Chunking reduces the amount of memory needed to complete computations, and finally combines the data after running through all batches. This process will increase the total run time but it may be a good option if the maximum RAM required to run the model is more than the available memory.


Refer to the :ref:`Chunking <Chunk>` section for more details on how to configure and run ActivitySim with chunking.

In general, before running ActivitySim for the first time on a new machine, it needs to be run in training_mode to efficiently handle memory usage (to set the appropriate chunking size). RSG, however, has ran the model in training mode once and has successfully used the cache on different machines. So in general, the provided cache.csv should help run the model on a different server as well.

However, should the model crash with an OutofMemory error on a first attempt on a new machine, ActivitySim should be run in training mode first to generate a new cache.csv file. To do so, change the chunk_training_mode line to training in the settings_source.yaml file, set household_sample_size: 0 and chunk_size to 80% of the available RAM on the machine (value is in bytes, so a value of 200_000_000_000 is equivalent to 200GB).

Multiprocessing
________________

The multiprocessing feature in ActivitySim enables the model (or parts of the model) to be run as independent parallel operations, thereby helping in reducing the run time. Most models can be implemented as a series of independent vectorized operations on pandas DataFrames and numpy arrays. These vectorized operations are much faster than sequential Python because they are implemented by native code (compiled C) and are to some extent multi-threaded.
ActivitySim’s modular and extensible architecture makes it possible to not hardwire the multiprocessing architecture. The specification of which models should be run in parallel, how many processers should be used, and the segmentation of the data between processes are all specified in the settings config file.

Refer to the :ref:`Multiprocessing Configuration` section for details on how to set up multiprocessing in ActivitySim.
:ref:`This <multiprocess_example>` section shows how to run the prototype mtc example using multiprocessing.


Sharrow
_______

:ref:`Sharrow <https://activitysim.github.io/sharrow/intro.html>`__ is a Python library designed to decrease run-time for ActivitySim models. The sharrow package is an extension of *numba*, and offers access to data formatting and a just-in-time compiler specifically for converting ActivitySim-style “specification” files into optimized, runnable functions that can significantly reduce the amount of run-time. The idea is to pay the cost of compiling these specification files only once, and then re-use the optimized results many times. If there is a change to the utility functions, machine, core, or the user deletes the cached files, this will automatically trigger a recompiling process.

Please refer to :ref:`Sharrow installation <https://activitysim.github.io/sharrow/intro.html#installation>`__ for details on how to install Sharrow. Demonstrative exampls of how to use Sharrow features can be found in the :ref:`Sharrow user guide <https://activitysim.github.io/sharrow/walkthrough/index.html>`__.


Tracing
_______

Tracing allows the user to access information throughout the model run for a specified number of households/persons/zones. Enabling this feature will increase run-time and memory usage. It is recommended that this feature be turned off for typical model application.

There are two types of tracing in ActivtiySim: household and origin-destination (OD) pair.  If a household trace ID
is specified, then ActivitySim will output a comprehensive set (i.e. hundreds) of trace files for all
calculations for all household members:

* ``Several CSV files`` - each input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for each sub-model

If an OD pair trace is specified, then ActivitySim will output the acessibility calculations trace
file:

* ``accessibility.result.csv`` - accessibility expression results for the OD pair

With the set of output CSV files, the user can trace ActivitySim calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.

Refer to :ref:`trace` for more details on configuring tracing and the various output files.
