Models
======

The currently implemented example ActivitySim models are described below.

Accessibility
--------------

The main interface to the accessibility model is the 
:py:func:`~activitysim.defaults.models.compute_accessibility` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["compute_accessibility"])``

Core Table: ``skims`` | Result Table: ``accessibility`` | Skims Keys: ``O-D, D-O``

API
~~~

.. automodule:: activitysim.defaults.models.accessibility
   :members:
   
Auto Ownership
--------------

The main interface to the auto ownership model is the 
:py:func:`~activitysim.defaults.models.auto_ownership_simulate` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["auto_ownership_simulate"])``

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.defaults.models.auto_ownership
   :members:

Coordinated Daily Activity Pattern (CDAP)
-----------------------------------------

The main interface to the CDAP model is the :py:func:`~activitysim.cdap.cdap.run_cdap` 
function.  This function is called by the orca step ``cdap_simulate`` which is 
registered as an orca step and is called in the example via 
``orca.run(["cdap_simulate"])``.  There are two cdap class definitions in
ActivitySim.  The first is at :py:func:`~activitysim.cdap.cdap` and contains the
core cdap module.  The second is at :py:func:`~activitysim.defaults.models.cdap` and
contains the orca wrapper for running it as part of the model pipeline.  

Core Table: ``persons`` | Result Field: ``cdap_activity`` | Skims Keys: NA

API
~~~

cdap
^^^^

.. automodule:: activitysim.cdap.cdap
   :members:

models.cdap
^^^^^^^^^^^

.. automodule:: activitysim.defaults.models.cdap
   :members:

Destination Choice
------------------

The main interface to the destination choice model is the 
:py:func:`~activitysim.defaults.models.destination_choice.destination_choice` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["destination_choice"])``

Core Table: ``tours`` | Result Field: ``destination`` | Skims Keys: ``TAZ,TAZ_r``

API
~~~

.. automodule:: activitysim.defaults.models.destination
   :members:


Mandatory Scheduling
--------------------

The main interface to the mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.defaults.models.mandatory_scheduling.mandatory_scheduling` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["mandatory_scheduling"])``

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.defaults.models.mandatory_scheduling
   :members:


Mandatory Tour Frequency
------------------------

The main interface to the mandatory tour purpose frequency model is the 
:py:func:`~activitysim.defaults.models.mandatory_tour_frequency.mandatory_tour_frequency` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["mandatory_tour_frequency"])``

Core Table: ``persons`` | Result Field: ``mandatory_tour_frequency`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.defaults.models.mandatory_tour_frequency
   :members:


Mode (Tour and Trip)
--------------------

The trip mode model currently operates on the tour table.  The trip model model repeats the same 
operation as the tour mode model, but uses the trip mode expression specification instead.

Tour
~~~~~

The main interface to the tour mode model is the 
:py:func:`~activitysim.defaults.models.mode._mode_choice_simulate` 
function.  This function is called in the orca step ``tour_mode_choice_simulate`` and in
the example via ``orca.run(["tour_mode_choice_simulate"])``.

Core Table: ``tours`` | Result Field: ``mode`` | Skims Keys: ``TAZ,destination`` | 
Skims3D in_skims Keys: ``TAZ,destination,in_period`` | Skims3D out_skims Keys: ``destination,TAZ,out_period``

Trip
~~~~

The main interface to the trip mode model is the 
:py:func:`~activitysim.defaults.models.mode._mode_choice_simulate` 
function.  This function is called in the orca step ``trip_mode_choice_simulate`` and in 
the example via ``orca.run(["trip_mode_choice_simulate"])``.

Core Table: ``trips`` | Result Field: ``mode`` | Skims Keys: ``TAZ,destination``
Skims3D in_skims Keys: ``TAZ,destination,in_period`` | Skims3D out_skims Keys: ``destination,TAZ,out_period``


API
~~~

.. automodule:: activitysim.defaults.models.mode
   :members:


Non-Mandatory Scheduling
------------------------

The main interface to the non-mandatory scheduling model is the 
:py:func:`~activitysim.defaults.models.non_mandatory_scheduling.non_mandatory_scheduling` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["non_mandatory_scheduling"])``

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.defaults.models.non_mandatory_scheduling
   :members:


Non-Mandatory Tour Frequency
----------------------------

The main interface to the non-mandatory tour frequency model is the 
:py:func:`~activitysim.defaults.models.non_mandatory_tour_frequency.non_mandatory_tour_frequency` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["non_mandatory_tour_frequency"])``

Core Table: ``persons`` | Result Field: ``persons_nmtf`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.defaults.models.non_mandatory_tour_frequency
   :members:


School Location
---------------

The main interface to the school location model is the 
:py:func:`~activitysim.defaults.models.school_location.school_location_simulate` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["school_location_simulate"])``

Core Table: ``persons`` | Result Field: ``school_taz`` | Skims Keys: ``TAZ,TAZ_r``

API
~~~

.. automodule:: activitysim.defaults.models.school_location
   :members:


Workplace Location
------------------
 
The main interface to the workplace location model is the 
:py:func:`~activitysim.defaults.models.workplace_location.workplace_location_simulate` 
function.  This function is registered as an orca step and is called in the example
via ``orca.run(["workplace_location_simulate"])``

Core Table: ``persons`` | Result Field: ``workplace_taz`` | Skims Keys: ``TAZ,TAZ_r``

API
~~~

.. automodule:: activitysim.defaults.models.workplace_location
   :members:
