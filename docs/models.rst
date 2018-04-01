Models
======

The currently implemented example ActivitySim AB models are described below.

.. _initialize:

Initialize
----------

The initialize model isn't really a model, but a step in the data pipeline.  This step pre-loads the land_use, 
households, persons, and person_windows tables because random seeds are set differently for each step 
and therefore the sampling of households depends on which step they are initially loaded in.  So, we 
load them explicitly up front. 

The main interface to the initialize model is the :py:func:`~activitysim.abm.models.initialize.initialize` 
function.  This function is registered as an orca step in the example Pipeline.

API
~~~

.. automodule:: activitysim.abm.models.initialize
   :members:
   
.. _accessibility:

Accessibility
-------------

The accessibilities model is an aggregate model that calculates multiple origin-based accessibility 
measures by origin zone to all destination zones. 

The accessibility measure first multiplies an employment variable by a mode-specific decay function.  The 
product reflects the difficulty of accessing the activities the farther (in terms of round-trip travel time) 
the jobs are from the location in question. The products to each destination zone are next summed over 
each origin zone, and the logarithm of the product mutes large differences.  The decay function on 
the walk accessibility measure is steeper than automobile or transit.  The minimum accessibility is zero.  

Level-of-service variables from three time periods are used, specifically the AM peak period (6 am to 10 am), the 
midday period (10 am to 3 pm), and the PM peak period (3 pm to 7 pm).

*Inputs*

* Highway skims for the three periods.  Each skim is expected to include a table named "TOLLTIMEDA", which is the drive alone in-vehicle travel time for automobiles willing to pay a "value" (time-savings) toll.
* Transit skims for the three periods.  Each skim is expected to include the following tables: (i) "IVT", in-vehicle time; (ii) "IWAIT", initial wait time; (iii) "XWAIT", transfer wait time; (iv) "WACC", walk access time; (v) "WAUX", auxiliary walk time; and, (vi) "WEGR", walk egress time.
* Zonal data with the following fields: (i) "TOTEMP", total employment; (ii) "RETEMPN", retail trade employment per the NAICS classification. 

*Outputs*

* taz, travel analysis zone number
* autoPeakRetail, the accessibility by automobile during peak conditions to retail employment for this TAZ
* autoPeakTotal, the accessibility by automobile during peak conditions to all employment
* autoOffPeakRetail, the accessibility by automobile during off-peak conditions to retail employment
* autoOffPeakTotal, the accessibility by automobile during off-peak conditions to all employment
* transitPeakRetail, the accessibility by transit during peak conditions to retail employment
* transitPeakTotal, the accessibility by transit during peak conditions to all employment
* transitOffPeakRetail, the accessiblity by transit during off-peak conditions to retail employment
* transitOffPeakTotal, the accessiblity by transit during off-peak conditions to all employment
* nonMotorizedRetail, the accessibility by walking during all time periods to retail employment
* nonMotorizedTotal, the accessibility by walking during all time periods to all employment

The main interface to the accessibility model is the 
:py:func:`~activitysim.abm.models.accessibility.compute_accessibility` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``skims`` | Result Table: ``accessibility`` | Skims Keys: ``O-D, D-O``

API
~~~

.. automodule:: activitysim.abm.models.accessibility
   :members:

.. _school_location:

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

.. _work_location:

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

.. _auto_ownership:

Auto Ownership
--------------

The auto ownership model selects a number of autos for each household in the simulation.  

The main interface to the auto ownership model is the 
:py:func:`~activitysim.abm.models.auto_ownership.auto_ownership_simulate` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.auto_ownership
   :members:

.. _cdap:

Coordinated Daily Activity Pattern
----------------------------------

The Coordinated Daily Activity Pattern (CDAP) model is a sequence of vectorized table operations:

* create a person level table and rank each person in the household for inclusion in the CDAP model
* solve individual M/N/H utilities for each person
* take as input an interaction coefficients table and then programatically produce and write out the expression files for households size 1, 2, 3, 4, and 5 models independent of one another
* select households of size 1, join all required person attributes, and then read and solve the automatically generated expressions
* repeat for households size 2, 3, 4, and 5. Each model is independent of one another.

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
^^^^^^^^^

.. automodule:: activitysim.abm.models.util.cdap
   :members:

.. _mandatory_tour_frequency:

Mandatory Tour Frequency
------------------------

The mandatory tour frequency model selects the number of mandatory tours made by each person on the simulation day.
It also creates mandatory tours in the data pipeline.

The main interface to the mandatory tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.mandatory_tour_frequency.mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``mandatory_tour_frequency, persons_mtf`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.mandatory_tour_frequency
   :members:

.. _mandatory_tour_scheduling:

Mandatory Tour Scheduling
-------------------------

The mandatory tour scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each mandatory tour.  This model uses person :ref:`time_windows`.

The main interface to the mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.mandatory_scheduling.mandatory_tour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.mandatory_scheduling
   :members:

.. _non_mandatory_tour_frequency:

Non-Mandatory Tour Frequency
----------------------------

The non-mandatory tour frequency model selects the number of non-mandatory tours made by each person on the simulation day.
It also creates non-mandatory tours in the data pipeline.

The main interface to the non-mandatory tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.non_mandatory_tour_frequency.non_mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``non_mandatory_tour_frequency, persons_nmtf`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.non_mandatory_tour_frequency
   :members:

.. _non_mandatory_tour_destination_choice:

Non-Mandatory Tour Destination Choice
-------------------------------------

The non-mandatory tour destination choice model chooses a destination zone for
non-mandatory tours.

The main interface to the non-mandatory tour destination choice model is the 
:py:func:`~activitysim.abm.models.non_mandatory_destination.non_mandatory_tour_destination_choice` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``destination`` | Skims Keys: ``TAZ, TAZ_r``

API
~~~

.. automodule:: activitysim.abm.models.non_mandatory_destination
   :members:

.. _annotate_table:

Annotate Table
--------------

The annotate table model isn't really a model, but a step in the data pipeline.  This step adds fields
to a user specified data table in the pipeline.  The main interface to annotate table is the 
:py:func:`~activitysim.abm.models.annotate_table.annotate_table` function.  This function is registered 
as an orca step in the example Pipeline.  Annotate_table make use of optional step arguments, as shown 
below.  In the example below, ``annotate_tours`` is passed as the ``model_name``, which causes annotate table
to read the annotate_tours.yaml settings file and the annotate_tours.csv expressions file.  These expressions
are processed are added to the dataframe specified in the yaml file.  

::

  - annotate_table.model_name=annotate_tours

API
~~~

.. automodule:: activitysim.abm.models.annotate_table
   :members:

.. _non_mandatory_tour_scheduling:

Non-Mandatory Tour Scheduling
-----------------------------

The non-mandatory tour scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each non-mandatory tour.  This model uses person :ref:`time_windows`.

The main interface to the non-mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.non_mandatory_scheduling.non_mandatory_tour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.non_mandatory_scheduling
   :members:
   
.. _tour_mode_choice:

Tour Mode
---------

The tour mode choice model assigns a travel mode to each tour.

The main interface to the tour mode model is the 
:py:func:`~activitysim.abm.models.mode_choice.tour_mode_choice_simulate` function.  This function is 
called in the orca step ``tour_mode_choice_simulate`` and is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``mode`` | Skims od_skims Keys: ``TAZ, destination`` | 
SkimStackWrapper odt_skims Keys: ``TAZ, destination, in_period`` | SkimStackWrapper dot_skims Keys: 
``destination, TAZ, out_period``

API
~~~

.. automodule:: activitysim.abm.models.mode_choice
   :members:

.. automodule:: activitysim.abm.models.util.mode
   :members:

.. _atwork_subtour_frequency:

At-work Subtours Frequency
--------------------------

The at-work subtour frequency model selects the number of at-work subtours made for each work tour.
It also creates at-work subtours by adding them to the tours table in the data pipeline.

Choosers: work tours
Alternatives: none, 1 eating out tour, 1 business tour, 1 maintenance tour, 2 business tours, 1 eating out tour + 1 business tour
Dependent tables: household, person, accessibility
Outputs: work tour subtour frequency choice, at-work tours table (with only tour origin zone at this point)

The main interface to the at-work subtours model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_frequency.atwork_subtour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Table: ``tours`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.atwork_subtour_frequency
   :members:
   
.. _atwork_subtour_destination:

At-work Subtours Destination Choice
-----------------------------------

The at-work subtours destination choice model is made up of three model steps:

  * atwork_subtour_destination_sample - selects a sample of alternative locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * atwork_subtour_destination_logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative location.
  * atwork_subtour_destination_simulate - starts with the table created above and chooses a final location, this time with the mode choice logsum included.

At-work Subtour Location Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model is the same as the workplace location sample model except it operates on the at-work tours.

Choosers: at-work tours
Alternatives: all zones
Dependent tables: skims, size terms, land use
Outputs: alternative zone set for each at-work tour

Core Table: ``tours`` | Result Table: ``atwork_subtour_destination_sample`` | Skims Keys: ``workplace_taz, alt zone, MD time period``

The main interface to the at-work subtours destination choice sample model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination_sample` 
function.  This function is registered as an orca step in the example Pipeline.

At-work Subtour Logsums
~~~~~~~~~~~~~~~~~~~~~~~

This model is the same as the workplace location logsums model except it operates on the at-work tours. 

Choosers: at-work tours * number of alternative zones
Alternatives: N/A
Dependent tables: skims, size terms, household, person, land use, accessibility, work tour
Outputs: logsums for the alternative zone set for each at-work tour

Core Table: ``atwork_subtour_destination_sample`` | Result Table: ``atwork_subtour_destination_sample`` | Skims Keys: ``workplace_taz, alt zone, MD time period``

The main interface to the at-work subtours destination choice logsums model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination_logsums` 
function.  This function is registered as an orca step in the example Pipeline.

At-work Subtour Location
~~~~~~~~~~~~~~~~~~~~~~~~

This model is the same as the workplace location model except it operates on the at-work tours. 

Choosers: at-work tours
Alternatives: alternative zones from the sample step
Dependent tables: skims, size terms, land use
Outputs: at-work tour destination zone

Core Table: ``atwork_subtour_destination_sample`` | Result Table: ``tours`` | Skims Keys: ``workplace_taz, alt zone, MD time period``

The main interface to the at-work subtours destination choice location model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination_simulate` 
function.  This function is registered as an orca step in the example Pipeline.

API
~~~

.. automodule:: activitysim.abm.models.atwork_subtour_destination
   :members:
   

.. _atwork_subtour_scheduling:

At-work Subtour Scheduling
--------------------------

The at-work subtours scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each at-work subtour.  This model uses person :ref:`time_windows`.

This model is the same as the mandatory tour scheduling model except it operates on the at-work tours and 
constrains the alternative set to available person :ref:`time_windows`.  Unlike the other departure time and duration models, 
the at-work subtour model does not require mode choice logsums. The at-work subtour frequency model can choose multiple 
tours so this model must process all first tours and then second tours since isFirstAtWorkTour is an explanatory variable.

Choosers: at-work tours
Alternatives: alternative departure time and arrival back at origin time pairs WITHIN the work tour departure time and arrival time back at origin AND the person time window. If no time window is available for the tour, make the first and last time periods within the work tour available, make the choice, and log the number of times this occurs.
Dependent tables: skims, person, land use, work tour
Outputs: at-work tour departure time and arrival back at origin time, updated person time windows

The main interface to the at-work subtours scheduling model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_scheduling.atwork_subtour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``tour_departure_and_duration`` | Skims Keys: Currently not implemented

API
~~~

.. automodule:: activitysim.abm.models.atwork_subtour_scheduling
   :members:

.. _atwork_subtour_mode_choice:

At-work Subtour Mode
--------------------

The at-work subtour mode choice model assigns a travel mode to each at-work subtour.

Choosers: at-work tours
Alternatives: modes
Dependent tables: skims, size terms, household, person, land use, accessibility, work tour
Outputs: at-work tour mode

The main interface to the at-work subtour mode choice model is the 
:py:func:`~activitysim.abm.models.mode_choice.atwork_subtour_mode_choice`
function.  This function is called in the orca step ``atwork_subtour_mode_choice`` and
is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``mode`` | Skims od_skims Keys: ``workplace_taz,destination`` |
SkimStackWrapper odt_skims Keys: ``workplace_taz,destination,out_period`` | 
SkimStackWrapper dot_skims Keys: ``destination,workplace_taz,in_period``

API
~~~

See :ref:`tour_mode_choice` API.

   
Create Trips
------------

The create trips model simply creates a new trips table in the data pipeline with two trips for each tour:

* outbound - from tour origin to tour destination 
* inbound - from tour destination to tour origin

The main interface to the create trips model is the 
:py:func:`~activitysim.abm.models.create_trips.create_simple_trips` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Table: ``trips`` | Skims Keys: NA

API
~~~

.. automodule:: activitysim.abm.models.create_trips
   :members:

.. _trip_mode_choice:

Trip Mode
---------

The trip mode choice model assigns a travel mode to each trip.

The main interface to the trip mode model is the 
:py:func:`~activitysim.abm.models.mode_choice.trip_mode_choice_simulate` 
function.  This function is called in the orca step ``trip_mode_choice_simulate`` and 
is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``mode`` | Skims od_skims Keys: ``TAZ,destination`` | 
SkimStackWrapper odt_skims Keys: ``TAZ,destination,start_period``

API
~~~

See :ref:`tour_mode_choice` API.




.. _utility_steps:

Utility Steps
-------------
Model step utilities for writing out pipeline data tables and for understanding 
data table size.

The main interfaces to the utility steps models are 
:py:func:`~activitysim.abm.models.utility_steps.write_data_dictionary` and 
:py:func:`~activitysim.abm.models.utility_steps.write_tables`.  These 
functions are registered as orca steps in the example Pipeline.

API
~~~

.. automodule:: activitysim.abm.models.utility_steps
   :members:

Util
----
 
Additional helper classes

API
~~~

.. automodule:: activitysim.abm.models.util.expressions
   :members:
      
.. automodule:: activitysim.abm.models.util.logsums
   :members:
      
.. automodule:: activitysim.abm.models.util.tour_frequency
   :members:

.. automodule:: activitysim.abm.models.util.vectorize_tour_scheduling
   :members:

Tests
-----
 
See activitysim.abm.test and activitysim.abm.models.util.test
