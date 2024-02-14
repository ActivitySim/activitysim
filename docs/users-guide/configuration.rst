
.. index:: configuration
.. _configuration:

=============
Configuration
=============

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
