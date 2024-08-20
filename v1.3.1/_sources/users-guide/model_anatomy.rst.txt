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

The three versions of multiple zone systems are one-zone, two-zone, and three-zone.

  * **One-zone**: This version is based on TM1 and supports only TAZs. All origins and
    destinations are represented at the TAZ level, and all skims including auto, transit,
    and non-motorized times and costs are also represented at the TAZ level.
  * **Two-zone**: This version is similar to many DaySim models. It uses microzones (MAZs)
    for origins and destinations, and TAZs for specification of auto and transit times and
    costs. Impedance for walk or bike all-the-way from the origin to the destination can
    be specified at the MAZ level for close together origins and destinations, and at
    the TAZ level for further origins and destinations. Users can also override transit
    walk access and egress times with times specified in the MAZ file by transit mode.
    Careful pre-calculation of the assumed transit walk access and egress time by MAZ
    and transit mode is required depending on the network scenario.
  * **Three-zone**: This version is based on the SANDAG generation of CT-RAMP models.
    Origins and destinations are represented at the MAZ level. Impedance for walk or
    bike all-the-way from the origin to the destination can be specified at the MAZ
    level for close together origins and destinations, and at the TAZ level for further
    origins and destinations, just like the two-zone system. TAZs are used for auto
    times and costs. The difference between this system and the two-zone system is that
    transit times and costs are represented between Transit Access Points (TAPs), which
    are essentially dummy zones that represent transit stops or clusters of stops.
    Transit skims are built between TAPs, since there are typically too many MAZs to
    build skims between them. Often multiple sets of TAP to TAP skims (local bus only,
    all modes, etc.) are created and input to the demand model for consideration.  Walk
    access and egress times are also calculated between the MAZ and the TAP, and total
    transit path utilities are assembled from their respective components - from MAZ to
    first boarding TAP, from first boarding to final alighting TAP, and from alighting
    TAP to destination MAZ. This assembling is done via the
    :ref:`transit_virtual_path_builder` (TVPB), which considers all possible
    combinations of nearby boarding and alighting TAPs for each origin destination MAZ
    pair.

Regions that have an interest in more precise transit forecasts may wish to adopt the
three-zone approach, while other regions may adopt the one or two-zone approach.  The
microzone version requires coding households and land use at the microzone level.
Typically an all-streets network is used for representation of non-motorized impedances.
This requires a routable all-streets network, with centroids and connectors for
microzones.  If the three-zone system is adopted, procedures need to be developed to
code TAPs from transit stops and populate the all-street network with TAP centroids
and centroid connectors.  A model with transit virtual path building takes longer to
run than a traditional TAZ only model, but it provides a much richer framework for
transit modeling.

.. note::
   The two and three zone system test examples are simple test examples developed from the TM1 example.  To develop the two zone system
   example, TM1 TAZs were labeled MAZs, each MAZ was assigned a TAZ, and MAZ to MAZ impedance files were created from the
   TAZ to TAZ impedances.  To develop the three zone example system example, the TM1 TAZ model was further transformed
   so select TAZs also became TAPs and TAP to TAP skims and MAZ to TAP impedances files were created.  While sufficient for
   initial development, these examples were insufficient for validation and performance testing of the new software. As a result,
   the :ref:`prototype_marin` example was created.

Example simple test configurations and inputs for two and three-zone system models are described below.

Examples
~~~~~~~~

To run the two zone and three zone system examples, do the following:

* Activate the correct conda environment if needed
* Create a local copy of the example

::

  # simple two zone example
  activitysim create -e placeholder_2_zone -d test_placeholder_2_zone

  # simple three zone example
  activitysim create -e placeholder_3_zone -d test_placeholder_3_zone


* Change to the example directory
* Run the example

::

  # simple two zone example
  activitysim run -c configs_2_zone -c configs -d data_2 -o output_2

  # simple three zone example, single process and multiprocess (and makes use of settings file inheritance for running)
  activitysim run -c configs_3_zone -c configs -d data_3 -o output_3 -s settings_static.yaml
  activitysim run -c configs_3_zone -c configs -d data_3 -o output_3 -s settings_mp.yaml

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

In addition to the extra two zone system settings and inputs above, the following additional settings and inputs are required for a three zone system model.  Examples values are illustrated below.

In ``settings.yaml``:

* ``models`` - add initialize_los and initialize_tvpb to load network LOS inputs / skims and pre-compute TAP to TAP utilities for TVPB.  See :ref:`initialize_los`.
* ``want_dest_choice_presampling`` - enable presampling for multizone systems, which means first select a TAZ using the sampling model and then select a microzone within the TAZ based on the microzone share of TAZ size term.

::

  models:
    - initialize_landuse
    - compute_accessibility
    - initialize_households
    # ---
    - initialize_los
    - initialize_tvpb
    # ---
    - school_location
    - workplace_location

In ``network_los.yaml``:

* ``zone_system`` - set to 3 for three zone system
* ``rebuild_tvpb_cache`` - rebuild and overwrite existing pre-computed TAP to TAP utilities cache
* ``trace_tvpb_cache_as_csv`` - write a CSV version of TVPB cache for tracing
* ``tap_skims`` - TAP to TAP skims OMX file name. The time period for the matrix must be represented at the end of the matrix name and be seperated by a double_underscore (e.g. BUS_IVT__AM indicates base skim BUS_IVT with a time period of AM).
* ``tap`` - TAPs table
* ``tap_lines`` - table of transit line names served for each TAP.  This file is used to trimmed the set of nearby TAP for each MAZ so only TAPs that are further away and serve new service are included in the TAP set for consideration.  It is a very important file to include as it can considerably reduce runtimes.
* ``maz_to_tap`` - list of MAZ to TAP access/egress impedance files by user defined mode.  Examples include walk and drive.  The file also includes MAZ to TAP impedances.
* ``maz_to_tap:{walk}:max_dist`` - max distance from MAZ to TAP to consider TAP
* ``maz_to_tap:{walk}:tap_line_distance_col`` - MAZ to TAP data field to use for TAP lines distance filter
* ``demographic_segments`` - list of user defined demographic_segments for pre-computed TVPB impedances.  Each chooser is coded with a user defined demographic segment.
* ``TVPB_SETTINGS:units`` - specify the units for calculations, e.g. utility or time.
* ``TVPB_SETTINGS:path_types`` - user defined set of TVPB path types to be calculated and available to the mode choice models.  Examples include walk transit walk (WTW), drive transit walk (DTW), and walk transit drive (WTD).
* ``TVPB_SETTINGS:path_types:{WTW}:access`` - access mode for the path type
* ``TVPB_SETTINGS:path_types:{WTW}:egress`` - egress mode for the path type
* ``TVPB_SETTINGS:path_types:{WTW}:max_paths_across_tap_sets`` - max paths to keep across all skim sets, for example, 3 TAP to TAP pairs per origin MAZ destination MAZ pair
* ``TVPB_SETTINGS:path_types:{WTW}:max_paths_per_tap_set`` - max paths to keep per skim set, for example 1 per skim set - all transit submodes, local bus only, etc.

Unlike the one and two zone system approach, the three zone system approach requires additional expression files for the TVPB.  The additional expression files for the TVPB are:

* ``TVPB_SETTINGS:tap_tap_settings:SPEC`` - TAP to TAP expressions, e.g. tvpb_utility_tap_tap.csv
* ``TVPB_SETTINGS:tap_tap_settings:PREPROCESSOR:SPEC`` - TAP to TAP chooser preprocessor, e.g. tvpb_utility_tap_tap_annotate_choosers_preprocessor.csv
* ``TVPB_SETTINGS:maz_tap_settings:walk:SPEC`` - MAZ to TAP {walk} expressions, e.g. tvpb_utility_walk_maz_tap.csv
* ``TVPB_SETTINGS:maz_tap_settings:drive:SPEC`` - MAZ to TAP {drive} expressions, e.g. tvpb_utility_drive_maz_tap.csv
* ``TVPB_SETTINGS:accessibility:tap_tap_settings:SPEC`` - TAP to TAP expressions for the accessibility calculator, e.g. tvpb_accessibility_tap_tap.csv
* ``TVPB_SETTINGS:accessibility:maz_tap_settings:walk:SPEC`` - MAz to TAP {walk} expressions for the accessibility calculator, e.g. tvpb_accessibility_walk_maz_tap.csv

Additional settings to configure the TVPB are:

* ``TVPB_SETTINGS:tap_tap_settings:attribute_segments:demographic_segment`` - TVPB pre-computes TAP to TAP total utilities for demographic segments.  These are defined using the attribute_segments keyword.  In the example below, the segments are demographic_segment (household income bin), tod (time-of-day), and access_mode (drive, walk).
* ``TVPB_SETTINGS:maz_tap_settings:{walk}:CHOOSER_COLUMNS`` - input impedance columns to expose for TVPB calculations.
* ``TVPB_SETTINGS:maz_tap_settings:{walk}:CONSTANTS`` - constants for TVPB calculations.
* ``accessibility:...`` - for the accessibility model step, the same basic set of TVPB configurations are available.

::

  zone_system: 3

  rebuild_tvpb_cache: False
  trace_tvpb_cache_as_csv: False
  tap_skims: tap_skims.omx
  tap: tap.csv
  maz_to_tap:
    walk:
      table: maz_to_tap_walk.csv
    drive:
      table: maz_to_tap_drive.csv

  demographic_segments: &demographic_segments
  - &low_income_segment_id 0
  - &high_income_segment_id 1

  TVPB_SETTINGS:
    tour_mode_choice:
      units: utility
      path_types:
        WTW:
          access: walk
          egress: walk
          max_paths_across_tap_sets: 3
          max_paths_per_tap_set: 1
        DTW:
          access: drive
          egress: walk
          max_paths_across_tap_sets: 3
          max_paths_per_tap_set: 1
        WTD:
          access: walk
          egress: drive
          max_paths_across_tap_sets: 3
          max_paths_per_tap_set: 1
      tap_tap_settings:
        SPEC: tvpb_utility_tap_tap.csv
        PREPROCESSOR:
          SPEC: tvpb_utility_tap_tap_annotate_choosers_preprocessor.csv
          DF: df
        attribute_segments:
          demographic_segment: *demographic_segments
          tod: *skim_time_period_labels
          access_mode: ['drive', 'walk']
        attributes_as_columns:
          - demographic_segment
          - tod
      maz_tap_settings:
        walk:
          SPEC: tvpb_utility_walk_maz_tap.csv
          CHOOSER_COLUMNS:
            - walk_time
        drive:
          SPEC: tvpb_utility_drive_maz_tap.csv
          CHOOSER_COLUMNS:
            - drive_time
            - DIST
      CONSTANTS:
        c_ivt_high_income: -0.028
        ...

    accessibility:
      units: time
      path_types:
        WTW:
          access: walk
          egress: walk
          max_paths_across_tap_sets: 1
          max_paths_per_tap_set: 1
      tap_tap_settings:
        SPEC: tvpb_accessibility_tap_tap_.csv
      maz_tap_settings:
          walk:
            SPEC: tvpb_accessibility_walk_maz_tap.csv
            CHOOSER_COLUMNS:
              - walk_time
      CONSTANTS:
          out_of_vehicle_walk_time_weight: 1.5
          out_of_vehicle_wait_time_weight: 2.0

Outputs
~~~~~~~

Essentially the same set of outputs is created for a two or three zone system
model as for a one zone system model.  However, the one key additional bit of
information for a three zone system model is the boarding TAP, alighting TAP, and
transit skim set is added to the relevant chooser table (e.g. tours and trips) when the
chosen mode is transit.  Logging and tracing also work for two and three zone models,
including tracing of the TVPB calculations. The :ref:`write_trip_matrices` step writes
both TAZ and TAP level matrices depending on the configured number of zone systems.

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

The key output of ActivitySIm is the HDF5 data pipeline file ``output\pipeline.h5``. This datastore by default contains
a copy of each data table after each model step in which the table was modified. The exact fields for each set of outputs will be different for various implementations of ActivitySim.


Logging
-------

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging
library.  The following key log files are created with a model run:

* ``activitysim.log`` - overall system log file
* ``timing_log.csv`` - submodel step runtimes
* ``omnibus_mem.csv`` - multiprocessed submodel memory usage
