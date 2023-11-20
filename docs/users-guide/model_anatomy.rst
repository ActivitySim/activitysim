Anatomy of a Model
==================

Input Data
----------
In order to run any model, the user needs the input files in the ``data`` folder as identified in the ``configs\settings.yaml``
file and the ``configs\network_los.yaml`` file.

Common inputs include
* ``households`` - Synthetic households for the region.
* ``persons`` - Synthetic population person records with socio-demographics details.
* ``zone based land use data``: Includes total population, employments, area types, etc. data for the region.
* ``skims``: OMX matrix file containing skim matrices for the zone system of the region.

Zone System
-----------





Configuration
-------------

The ``configs`` folder for a model implementation contains settings, expressions
files, and other files required for specifying model utilities and form.  Each
component will have one or more files that control the operation of that
component.

.. currentmodule:: activitysim.core.configuration

Top Level Settings
------------------

.. autosummary::
    :toctree: _generated
    :template: autopydantic.rst
    :recursive:

    Settings
    InputTable
    OutputTable
    OutputTables
    MultiprocessStep
    MultiprocessStepSlice


File System
-----------

.. autosummary::
    :toctree: _generated
    :template: autopydantic.rst
    :recursive:

    FileSystem


Network Level of Service
------------------------

.. autosummary::
    :toctree: _generated
    :template: autopydantic.rst
    :recursive:

    NetworkSettings
    TAZ_Settings
    DigitalEncoding



Utility Specifications
----------------------

The model specifications files are typically included in the ``configs`` folder. These files store python/pandas/numpy expressions,
alternatives, and other settings for each model.


Outputs
-------

The key output of ActivitySIm is the HDF5 data pipeline file ``output\pipeline.h5``. This datastore by default contains
a copy of each data table after each model step in which the table was modified.


Logging
-------

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging
library.  The following key log files are created with a model run:

* ``activitysim.log`` - overall system log file
* ``timing_log.csv`` - submodel step runtimes
* ``omnibus_mem.csv`` - multiprocessed submodel memory usage
