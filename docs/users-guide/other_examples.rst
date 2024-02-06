
Other Example Models
====================

There are several example models available in `ActivitySim's GitHub repository <https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples>`__. These example models are categorized into three types:


* "production" examples are calibrated and validated by the relevant agency, and
  are intended to be replicas of an "official" travel model used by that agency,
  although generally we expect agencies to maintain independent repositories and
  not rely on the Consortium maintained version as a single source for the model.
  At this time there are no production examples in the consortium's collection,
  but we expect that to change soon.
* "prototype" examples are not representative of any "official" travel model
  used by the relevant agency, but they are viewed as "ok" models by the
  consortium: they are usually at least loosely calibrated and/or validated, and
  typically contain at least some components or parameters specialized for the
  relevant region. They may be in-development models that are not quite finished,
  or consortium maintained models that are derived from but now different from
  the official model of some region. They should not be used in place of
  "official" models for policy analysis in any given region, but could serve as
  a donor model for new users who want to implement ActivitySim somewhere new
  (subject to all the caveats that go along with transferring models).
* "placeholder" examples are computational testbeds that technically run but
  have not been calibrated nor validated in any meaningful way. These examples
  are early stage development models used for testing purposes, and users are
  strongly cautioned not to use them for any policy or planning purpose.

Some available examples include those listed in the table below.

.. note::
   Additional details on models provided below may not be complete or up-to-date.

+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| Example                                   | Purpose                                                   | Zone Systems | Status               |
+===========================================+===========================================================+==============+======================+
| :ref:`prototype_mtc`                      | Original ActivitySim Example, derived from MTC TM1        | 1            | Mature               |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| prototype_mtc_extended                    | Prototype MTC example with additional optional models     | 1            | In development       |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| example_estimation                        | Estimation example with prototype_mtc                     | 1            | Mature               |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| placeholder_multiple_zone                 | 2 or 3 zone system example using MTC data                 | 2 or 3       | Simple test example  |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| prototype_marin                           | 3 zone system example using Marin tour mode choice model  | 3            | Mature               |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| prototype_arc                             | ARC agency example                                        | 1            | In development       |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| :ref:`prototype_semcog`                   | SEMCOG agency example                                     | 1            | In production        |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| placeholder_psrc                          | PSRC agency example                                       | 2            | Future development   |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| placeholder_sandag                        | SANDAG agency example                                     | 3            | Future development   |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| prototype_sandag_xborder                  | SANDAG agency example                                     | 3            | In development       |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+
| prototype_mwcog                           | MWCOG agency example                                      | 2            | In development       |
+-------------------------------------------+-----------------------------------------------------------+--------------+----------------------+

.. note::
   The `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
   contains example commands to create and run several versions of the examples.  See also :ref:`adding_agency_examples` for more
   information on agency example models.

.. _prototype_mtc_extended :

**prototype_mtc_extended**


prototype_mtc_extended contains additional models that were developed to enhance ActivitySim's modeling
capabilities. This example inherets
the data and configuration files from prototype_mtc. The current list of additional models included
in this example are:

* :ref:`vehicle_type_choice`: Selects a vehicle type for each household vehicle. Runs after auto_ownership.
* :ref:`vehicle_allocation`: Allocates a vehicle for each tour and each occupancy level.  Tour and trip mode choice
  auto operating costs are modified to reflect the allocated vehicle option.
* :ref:`school_escorting`: Explicitly models school drop-off / pick-up of students to and from school.

The prototype_mtc_extended example also contains changes to test the flexible number of tour and trip ids.
(Information in why this is important can be found `here <https://github.com/ActivitySim/activitysim/wiki/Project-Meeting-2022.08.09>`__.)
The following changes were made to demonstrate this:

* An additional alternative was added to the non-mandatory tour frequency alternatives file containing 2 other discretionary tours.
* An additional alternative was added to the stop_frequency_alts.csv for 4 outbound stops and 3 inbound stops. This alternative was then
  included as part of the stop_frequency_othdiscr.csv specification with an added calibration constant to control that alternative.
  Because an additional trip may now happen in the outbound direction, the trip scheduling probabilities table was extended for the
  other discretionary tour purpose where the fourth outbound trip rows were copied for the now availabile fifth trip.

.. _prototype_marin :

**prototype_marin**


To finalize development and verification of the multiple zone system and transit virtual path building components, the
`Transportation Authority of Marin County <https://www.tam.ca.gov/>`__ version of MTC travel model two (TM2) work
tour mode choice model was implemented.  This example was also developed to test multiprocessed runtime performance.
The complete runnable setup is available from the ActivitySim command line interface as `prototype_3_marin_full`.  This example
has essentially the same configuration as the simpler three zone example above.

*Example*


To run prototype_marin, do the following:

* Activate the correct conda environment if needed
* Create a local copy of the example

::

  # Marin TM2 work tour mode choice for the MTC region
  activitysim create -e prototype_3_marin_full -d test_prototype_3_marin_full

* Change to the example directory
* Run the example

::

  # Marin TM2 work tour mode choice for the MTC region
  activitysim run -c configs -d data -o output -s settings_mp.yaml

* For optimal performance, configure multiprocessing and chunk_size based on machine hardware.


*Settings*


Additional settings for running the Marin TM2 tour mode choice example are in the ``network_los.yaml`` file.  The
only additional notable setting is the ``tap_lines`` setting, which identifies a table of transit line names
served for each TAP.  This file is used to trimmed the set of nearby TAP for each MAZ so only TAPs that are
further away and serve new service are included in the TAP set for consideration.  It is a very important
file to include as it can considerably reduce runtimes.

::

  tap_lines: tap_lines.csv


.. _prototype_arc :

**prototype_arc**


.. note::

  This example is in development


The prototype_arc added a :ref:`trip_scheduling_choice`, :ref:`trip_departure_choice`, and :ref:`parking_location_choice`
submodel.  These submodel specification files are below, and are in addition to the :ref:`prototype_mtc`
submodel :ref:`sub-model-spec-files`.

.. _arc-sub-model-spec-files:

*Example ARC Sub-Model Specification Files*


+------------------------------------------------+--------------------------------------------------------------------+
|            Model                               |    Specification Files                                             |
+================================================+====================================================================+
|  :ref:`trip_scheduling_choice`                 |  - trip_scheduling_choice.yaml                                     |
|                                                |  - trip_scheduling_choice_preprocessor.csv                         |
|                                                |  - trip_scheduling_choice.csv                                      |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`trip_departure_choice`                  |  - trip_departure_choice.yaml                                      |
|                                                |  - trip_departure_choice_preprocessor.csv                          |
|                                                |  - trip_departure_choice.csv                                       |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`parking_location_choice`                |  - parking_location_choice.yaml                                    |
|                                                |  - parking_location_choice_annotate_trips_preprocessor.csv         |
|                                                |  - parking_location_choice_coeffs.csv                              |
|                                                |  - parking_location_choice.csv                                     |
+------------------------------------------------+--------------------------------------------------------------------+

*Example*


See example commands in `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
for running prototype_arc.  For optimal performance, configure multiprocessing and chunk_size based on machine hardware.

.. _placeholder_psrc :

**placeholder_psrc**


.. note::

  This example is a placeholder model used only for code development and debugging, and is not suitable for policy analysis


The placeholder_psrc is a two zone system (MAZs and TAZs) implementation of the
prototype_mtc model design.  It uses PSRC zones, land use, synthetic population, and network LOS (skims).

*Example*


See example commands in `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
for running placeholder_psrc.  For optimal performance, configure multiprocessing and chunk_size based on machine hardware.

.. _placeholder_sandag :

**placeholder_sandag**


.. note::

  This example is in development


The placeholder_sandag is a multi-part model, containing one-, two-, and three- zone system (MAZs, TAZs, and TAPs) implementation of the
prototype_mtc model design.  It uses SANDAG zones, land use, synthetic population, and network LOS (skims).

*Example*


See example commands in `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
for running placeholder_sandag.  For optimal performance, configure multiprocessing and chunk_size based on machine hardware.

.. _prototype_sandag_xborder :

**prototype_sandag_xborder**


.. note::

  This example is in development


The prototype_sandag_xborder is a three zone system (MAZs, TAZs, and TAPs) that
generates cross-border activities for a tour-based population of Mexican residents.
In addition to the normal SANDAG zones, there are external MAZs and TAZs defined for
each border crossing station (Port of Entry). Because the model is tour-based, there
are no household or person-level attributes in the synthetic population. The
principal difference between this and the standard 3-zone implementation is that
since household do not have a default tour origin (home zones), a tour OD choice
model is required to assign tour origins and destinations simultaneously.

*Example*


See example commands in `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
for running prototype_sandag_xborder.  For optimal performance, configure multiprocessing and chunk_size based on machine hardware.

.. _prototype_mwcog :

**prototype_mwcog**


The prototype_mwcog is a one zone system (TAZs only).

*Example*


See example commands in `example_manifest.yaml <https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_manifest.yaml>`_
for running prototype_mwcog.  For optimal performance, configure multiprocessing and chunk_size based on machine hardware.
