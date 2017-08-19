Models
======

The currently implemented example ActivitySim AB models are described below.

Accessibility
--------------

The main interface to the accessibility model is the 
:py:func:`~activitysim.abm.models.accessibility.compute_accessibility` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``skims`` | Result Table: ``accessibility`` | Skims Keys: ``O-D, D-O``

API
~~~

.. automodule:: activitysim.abm.models.accessibility
   :members:
   
Auto Ownership
--------------

The main interface to the auto ownership model is the 
:py:func:`~activitysim.abm.models.auto_ownership.auto_ownership_simulate` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.auto_ownership
   :members:

.. _cdap:

Coordinated Daily Activity Pattern (CDAP)
-----------------------------------------

The main interface to the CDAP model is the :py:func:`~activitysim.abm.models.util.cdap.run_cdap` 
function.  This function is called by the orca step ``cdap_simulate`` which is 
registered as an orca step in the example Pipeline.  There are two cdap class definitions in
ActivitySim.  The first is at :py:func:`~activitysim.abm.models.cdap` and contains the orca 
wrapper for running it as part of the model pipeline.  The second is 
at :py:func:`~activitysim.abm.models.util.cdap` and contains CDAP model logic.

Core Table: ``persons`` | Result Field: ``cdap_activity`` | Skims Keys: NA

API
~~~

cdap
^^^^

.. automodule:: activitysim.abm.models.cdap
   :members:

util.cdap
^^^^^^^^^^^

.. automodule:: activitysim.abm.models.util.cdap
   :members:

Destination Choice
------------------

The main interface to the destination choice model is the 
:py:func:`~activitysim.abm.models.destination.destination_choice` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``destination`` | Skims Keys: ``TAZ, TAZ_r``

API
~~~

.. automodule:: activitysim.abm.models.destination
   :members:


Mandatory Scheduling
--------------------

The main interface to the mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.mandatory_scheduling.mandatory_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.mandatory_scheduling
   :members:

Mandatory Tour Frequency
------------------------

The main interface to the mandatory tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.mandatory_tour_frequency.mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``mandatory_tour_frequency`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.mandatory_tour_frequency
   :members:

Create Trips
--------------

The main interface to the create trips model is the 
:py:func:`~activitysim.abm.models.create_trips.create_simple_trips` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.create_trips
   :members:

.. _mode_choice:

Mode (Tour and Trip)
--------------------

Tour
~~~~

The main interface to the tour mode model is the 
:py:func:`~activitysim.abm.models.mode.tour_mode_choice_simulate` function.  This function is 
called in the orca step ``tour_mode_choice_simulate`` and 
is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``mode`` | Skims od_skims Keys: ``TAZ, destination`` | 
SkimStackWrapper odt_skims Keys: ``TAZ, destination, in_period`` | SkimStackWrapper dot_skims Keys: 
``destination, TAZ, out_period``

Trip
~~~~

The main interface to the trip mode model is the 
:py:func:`~activitysim.abm.models.mode.trip_mode_choice_simulate` 
function.  This function is called in the orca step ``trip_mode_choice_simulate`` and 
is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``mode`` | Skims od_skims Keys: ``TAZ,destination``
SkimStackWrapper odt_skims Keys: ``TAZ,destination,start_period``


API
~~~

.. automodule:: activitysim.abm.models.mode
   :members:

.. automodule:: activitysim.abm.models.util.mode
   :members:

Non-Mandatory Scheduling
------------------------

The main interface to the non-mandatory scheduling model is the 
:py:func:`~activitysim.abm.models.non_mandatory_scheduling.non_mandatory_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.non_mandatory_scheduling
   :members:


Non-Mandatory Tour Frequency
----------------------------

The main interface to the non-mandatory tour frequency model is the 
:py:func:`~activitysim.abm.models.non_mandatory_tour_frequency.non_mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``persons_nmtf`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.non_mandatory_tour_frequency
   :members:


School Location
---------------

The school location model is made up of three model steps:
  * school_location_sample - selects a sample of alternative school locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * school_location_logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative school location.
  * school_location_simulate - starts with the table created above and chooses a final school location, this time with the mode choice logsum included.

The interfaces to the model steps are 
:py:func:`~activitysim.abm.models.school_location.school_location_sample`,
:py:func:`~activitysim.abm.models.school_location.school_location_logsums`,
:py:func:`~activitysim.abm.models.school_location.school_location_simulate`.  
These functions are registered as orca steps in the example Pipeline.

Core Table: ``persons`` | Result Field: ``school_taz`` | Skims Keys: ``TAZ, TAZ_r``

API
~~~

.. automodule:: activitysim.abm.models.school_location
   :members:


Workplace Location
------------------
 
The work location model is made up of three model steps:
  * workplace_location_sample - selects a sample of alternative work locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * workplace_location_logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative work location.
  * workplace_location_simulate - starts with the table created above and chooses a final work location, this time with the mode choice logsum included.

The interfaces to the model steps are 
:py:func:`~activitysim.abm.models.workplace_location.workplace_location_sample`,
:py:func:`~activitysim.abm.models.workplace_location.workplace_location_logsums`,
:py:func:`~activitysim.abm.models.workplace_location.workplace_location_simulate`.  
These functions are registered as orca steps in the example Pipeline.

Core Table: ``persons`` | Result Field: ``workplace_taz`` | Skims Keys: ``TAZ, TAZ_r``

API
~~~

.. automodule:: activitysim.abm.models.workplace_location
   :members:

Util
----
 
Helper classes

API
~~~

.. automodule:: activitysim.abm.models.util.cdap
   :members:
   
.. automodule:: activitysim.abm.models.util.logsums
   :members:
   
.. automodule:: activitysim.abm.models.util.mode
   :members:
   
.. automodule:: activitysim.abm.models.util.tour_frequency
   :members:

.. automodule:: activitysim.abm.models.util.vectorize_tour_scheduling
   :members:

Tests
-----
 
See activitysim.abm.test and activitysim.abm.models.util.test
