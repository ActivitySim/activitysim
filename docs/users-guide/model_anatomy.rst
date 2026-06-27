Anatomy of a Model
==================


.. index:: constants
.. index:: households
.. index:: input store
.. index:: land use
.. index:: persons
.. index:: size terms
.. index:: time windows table
.. index:: tours
.. index:: trips


Input Data
----------
In order to run any model, the user needs the input files in the ``data`` folder as identified in the ``configs\settings.yaml``
file and the ``configs\network_los.yaml`` file.


The following tables are currently implemented:

* households - household attributes for each household being simulated.  Index: ``household_id`` (see ``activitysim.abm.tables.households.py``)
* landuse - zonal land use (such as population and employment) attributes. Index: ``zone_id`` (see ``activitysim.abm.tables.landuse.py``)
* persons - person attributes for each person being simulated.  Index: ``person_id`` (see ``activitysim.abm.tables.persons.py``)
* time windows - manages person time windows throughout the simulation.  See :ref:`time_windows`.  Index:  ``person_id`` (see the person_windows table create decorator in ``activitysim.abm.tables.time_windows.py``)
* tours - tour attributes for each tour (mandatory, non-mandatory, joint, and atwork-subtour) being simulated.  Index:  ``tour_id`` (see ``activitysim.abm.models.util.tour_frequency.py``)
* trips - trip attributes for each trip being simulated.  Index: ``trip_id`` (see ``activitysim.abm.models.stop_frequency.py``)

A few additional tables are also used, which are not really tables, but classes:

* input store - reads input data tables from the input data store
* constants - various constants used throughout the model system, such as person type codes
* shadow pricing - shadow price calculator and associated utility methods, see :ref:`shadow_pricing`
* size terms - created by reading the ``destination_choice_size_terms.csv`` input file.  Index - ``segment`` (see ``activitysim.abm.tables.size_terms.py``)
* skims - each model runs requires skims, but how the skims are defined can vary significantly depending on the ActivitySim implementation. The skims class defines Inject injectables to access the skim matrices. The skims class reads the skims from the omx_file on disk.
* table dictionary - stores which tables should be registered as random number generator channels for restartability of the pipeline



.. _zone_systems :

Zone System
-----------

ActivitySim supports models with multiple zone systems.

In a multiple zone system approach, households, land use, and trips are modeled at the microzone (MAZ) level.  MAZs are smaller
than traditional TAZs and therefore make for a more precise system.  However, when considering network level-of-service (LOS)
indicators (e.g. skims), the model uses different spatial resolutions for different travel modes in order to reduce the network
modeling burden and model runtimes.  The typical multiple zone system setup is a TAZ zone system for auto travel, a MAZ zone
system for non-motorized travel, and optionally a transit access points (TAPs) zone system for transit.

The two versions of zone systems are one-zone and two-zone.

*   **One-zone**: This version is based on TM1 and supports only TAZs. All origins and
    destinations are represented at the TAZ level, and all skims including auto, transit,
    and non-motorized times and costs are also represented at the TAZ level.
*   **Two-zone**: This version is similar to many DaySim models. It uses microzones (MAZs)
    for origins and destinations, and TAZs for specification of auto and transit times and
    costs. Impedance for walk or bike all-the-way from the origin to the destination can
    be specified at the MAZ level for close together origins and destinations, and at
    the TAZ level for further origins and destinations. Users can also override transit
    walk access and egress times with times specified in the MAZ file by transit mode.
    Careful pre-calculation of the assumed transit walk access and egress time by MAZ
    and transit mode is required depending on the network scenario.

..  caution::
    Historically, there was also a three-zone option. The three-zone system has been
    removed as of version 1.5.2.

Regions that have an interest in more precise transit and non-motorized forecasts
may wish to adopt the two-zone approach, while other regions may adopt the one or two-zone approach.  The
microzone version requires coding households and land use at the microzone level.
Typically an all-streets network is used for representation of non-motorized impedances.
This requires a routable all-streets network, with centroids and connectors for
microzones.

.. _omx_skims :

Skims
~~~~~

The basic level-of-service data that represents the transportation system is
made available to ActivitySim via one or more sets of "skims".  Skims are
essentially matrices of travel times, costs, and other level of service attributes,
calculated between various zones in the system.  This skim data is made available
to ActivitySim using files in the`openmatrix <http://github.com/osPlanning/omx>`__
(OMX) format.  All of the skim data can be provided in a single OMX file, or
multiple OMX files can be used (this is typical for larger models, to keep file
sizes manageable).  If multiple files are used, the content of those files is
simply concatenated together into a single notional bucket of skim data when
the model is run.  Within that bucket, each skim variable is identified by a unique name.
For skim variables that vary across model time periods, the time period is
appended to the skim name, separated by a double underscore (e.g. ``BUS_IVT__AM``).

..  caution::
    When using "legacy" mode for ActivitySim, it is possible (but not recommended)
    to have a skim variable that has both a time period agnostic value as well as
    a set of time period dependent values, e.g. "WALK_TIME" and "WALK_TIME__AM".
    If you have conflicting names like this, a warning message will be issued, which
    will look like this in an ActivitySim log file:

    .. code-block:: text

        WARNING: activitysim/core/skim_dict_factory.py:212:
        UserWarning: some skims have both time-dependent and time-agnostic versions:
        - BIKE_LOGSUM
        - BIKE_TIME

    This is a warning, not an error, and the model will run if not using sharrow.
    However, if "sharrow" mode is activated, this will result in an error once the
    skims are actually loaded, unless instructions are included in the settings file
    to resolve the conflict.  The error message will look like this:

    .. code-block:: text

        ERROR: skims ['BIKE_TIME'] are present in both time-dependent and time-agnostic formats.
        Please add ignore rules to the omx_ignore_patterns setting to resolve this issue.
        To ignore the time dependent skims, add the following to your settings file:

        omx_ignore_patterns:
          - '^BIKE_TIME__.+'

        To ignore the time agnostic skims, add the following to your settings file:

        omx_ignore_patterns:
          - '^BIKE_TIME$'

        You can also do some variation or combination of the two, as long as you resolve
        the conflict(s). In addition, note that minor edits to model spec files may be
        needed to accommodate these changes in how skim data is represented (e.g. changing
        `odt_skims` to `od_skims`, or similar modifications wherever the offending variable
        names are used).  Alternatively, you can modify the skim data in the source files to
        remove the naming conflicts, which is typically done upstream of ActivitySim in
        whatever tool you are using to create the skims in the first place.

    It should be relatively simple to resolve the conflict by following the instructions
    in the error message. The cleaner and more reliable solution is to ensure each skim
    variable has a unique name, e.g. by changing the name on the time period agnostic
    value, so that instead of "BIKE_TIME" it is "BIKE_TIME_BASE". This may also require
    minor edits to the model spec files to accommodate the new skim name.


Examples
~~~~~~~~
Example simple test configurations and inputs for two and three-zone system models are described below.

To run the two zone and three zone system examples, do the following:

* Create a local copy of the example

::

  # simple two zone example
  uv run activitysim create -e placeholder_2_zone -d test_placeholder_2_zone

  # simple three zone example
  uv run activitysim create -e placeholder_3_zone -d test_placeholder_3_zone


* Change to the example directory
* Run the example

::

  # simple two zone example
  uv run activitysim run -c configs_2_zone -c configs -d data_2 -o output_2

  # simple three zone example, single process and multiprocess (and makes use of settings file inheritance for running)
  uv run activitysim run -c configs_3_zone -c configs -d data_3 -o output_3 -s settings_static.yaml
  uv run activitysim run -c configs_3_zone -c configs -d data_3 -o output_3 -s settings_mp.yaml

Settings
~~~~~~~~

Additional settings for running ActivitySim with two or three zone systems are specified in the ``settings.yaml`` and
``network_los.yaml`` files.  The settings are:

Two Zone
^^^^^^^^

In ``settings.yaml``:

* ``want_dest_choice_presampling`` - enable presampling for multizone systems, which
  means first select a TAZ using the sampling model and then select a microzone within
  the TAZ based on the microzone share of TAZ size term.

In ``network_los.yaml``:

The additional two zone system settings and inputs are described and illustrated below.
No additional utility expression files or expression revisions are required beyond the
one zone approach.  The MAZ data is available as zone data and the MAZ to MAZ data is
available using the existing skim expressions.  Users can specify mode utilities using
MAZ data, MAZ to MAZ impedances, and TAZ to TAZ impedances.

* ``zone_system`` - set to 2 for two zone system
* ``maz`` -  MAZ data file, with MAZ ID, TAZ, and land use and other MAZ attributes
* ``maz_to_maz:tables`` - list of MAZ to MAZ impedance tables.  These tables are read
  as pandas DataFrames and the columns are exposed to expressions.
* ``maz_to_maz:max_blend_distance`` - in order to avoid cliff effects, the lookup of
  MAZ to MAZ impedance can be a blend of origin MAZ to destination MAZ impedance and
  origin TAZ to destination TAZ impedance up to a max distance.  The blending formula
  is below.  This requires specifying a distance TAZ skim and distance columns from
  the MAZ to MAZ files.  The TAZ skim name and MAZ to MAZ column name need to be the
  same so the blending can happen on-the-fly or else a value of 0 is returned.

::

  (MAZ to MAZ distance) * (distance / max distance) * (TAZ to TAZ distance) * (1 - (distance / max distance))


* ``maz_to_maz:blend_distance_skim_name`` - Identify the distance skim for the blending calculation if different than the blend skim.

::

  zone_system: 2
  maz: maz.csv

  maz_to_maz:
    tables:
      - maz_to_maz_walk.csv
      - maz_to_maz_bike.csv

    max_blend_distance:
      DIST: 5
      DISTBIKE: 0
      DISTWALK: 1

    blend_distance_skim_name: DIST


Three Zone
^^^^^^^^^^

Three zone systems (TAZ, MAZ, TAP) are no longer supported in ActivitySim as of version 1.5.2.

Outputs
~~~~~~~

Essentially the same set of outputs is created for a two zone system model as
for a one zone system model. Logging and tracing also work for two zone models.
The :ref:`write_trip_matrices` step writes TAZ level matrices.

.. _presampling :

Presampling
~~~~~~~~~~~

In multiple zone systems models, destination choice presampling is activated by default.  Destination
choice presampling first aggregates microzone size terms to the TAZ level and then runs destination
choice sampling at the TAZ level using the destination choice sampling models.  After sampling X
number of TAZs based on impedance and size, the model selects a microzone for each TAZ based
on the microzone share of TAZ size.  Presampling significantly reduces runtime while producing
similar results.


.. _user_configuration :

Configuration
-------------

The ``configs`` folder for a model implementation contains settings, expressions
files, and other files required for specifying model utilities and form.  Each
component will have one or more files that control the operation of that
component. More information about individual configuration files can be found in the :ref:`Components <dev_components>` section of the Developers Guide.

.. currentmodule:: activitysim.core.configuration

Top Level Settings
------------------

.. autosummary::

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

    :template: autopydantic.rst
    :recursive:

    FileSystem


Network Level of Service
------------------------

.. autosummary::

    :template: autopydantic.rst
    :recursive:

    NetworkSettings
    TAZ_Settings
    DigitalEncoding



Utility Specifications
----------------------

The model specifications files are typically included in the ``configs`` folder. These files store python/pandas/numpy expressions,
alternatives, coefficients, constants and other settings for each model. For more information, see the :ref:`Utility Expressions<util_expressions>` section of the Developers Guide.


Outputs
-------

The key output of ActivitySim is the HDF5 data pipeline file ``output\pipeline.h5``. This datastore by default contains
a copy of each data table after each model step in which the table was modified. The exact fields for each set of outputs will be different for various implementations of ActivitySim.


Logging
-------

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging
library.  The following key log files are created with a model run:

* ``activitysim.log`` - overall system log file
* ``timing_log.csv`` - submodel step runtimes
* ``omnibus_mem.csv`` - multiprocessed submodel memory usage
