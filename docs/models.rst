
.. index:: models
.. _models:

Models
======

The currently implemented example ActivitySim AB models are described below.  See the example 
model :ref:`sub-model-spec-files` for more information.

.. _initialize_landuse:
.. _initialize_households:

Initialize
----------

The initialize model isn't really a model, but rather a few data processing steps in the data pipeline.  
The initialize data processing steps code variables used in downstream models, such as household and person
value-of-time.  This step also pre-loads the land_use, households, persons, and person_windows tables because 
random seeds are set differently for each step and therefore the sampling of households depends on which step 
they are initially loaded in. 

The main interface to the initialize land use step is the :py:func:`~activitysim.abm.models.initialize.initialize_landuse` 
function. The main interface to the initialize household step is the :py:func:`~activitysim.abm.models.initialize.initialize_households` 
function.  These functions are registered as orca steps in the example Pipeline.

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


.. automodule:: activitysim.abm.models.accessibility
   :members:

.. _school_location:
.. _work_location:

School Location
---------------

The usual school location choice models assign a usual school location for the primary
mandatory activity of each child and university student in the
synthetic population. The models are composed of a set of accessibility-based parameters
(including one-way distance between home and primary destination and the tour mode choice
logsum - the expected maximum utility in the mode choice model which is given by the
logarithm of the sum of exponentials in the denominator of the logit formula) and size terms,
which describe the quantity of grade-school or university opportunities in each possible
destination.

The school location model is made up of four steps:
  * sampling - selects a sample of alternative school locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative school location.
  * simulate - starts with the table created above and chooses a final school location, this time with the mode choice logsum included.
  * shadow prices - compare modeled zonal destinations to target zonal size terms and calculate updated shadow prices.
  
These steps are repeated until shadow pricing convergence criteria are satisfied or a max number of iterations is reached.  See :ref:`shadow_pricing`. 

The main interfaces to the model is the :py:func:`~activitysim.abm.models.location_choice.school_location` function.  
This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``school_taz`` | Skims Keys: ``TAZ, alt_dest, AM time period, MD time period``

Work Location
-------------
 
The usual work location choice models assign a usual work location for the primary
mandatory activity of each employed person in the
synthetic population. The models are composed of a set of accessibility-based parameters
(including one-way distance between home and primary destination and the tour mode choice
logsum - the expected maximum utility in the mode choice model which is given by the
logarithm of the sum of exponentials in the denominator of the logit formula) and size terms,
which describe the quantity of work opportunities in each possible destination.

The work location model is made up of four steps:
  * sample - selects a sample of alternative work locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative work location.
  * simulate - starts with the table created above and chooses a final work location, this time with the mode choice logsum included.
  * shadow prices - compare modeled zonal destinations to target zonal size terms and calculate updated shadow prices.

These steps are repeated until shadow pricing convergence criteria are satisfied or a max number of iterations is reached.  See :ref:`shadow_pricing`.

The main interfaces to the model is the :py:func:`~activitysim.abm.models.location_choice.workplace_location` function.  
This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``workplace_taz`` | Skims Keys: ``TAZ, alt_dest, AM time period, PM time period``


.. automodule:: activitysim.abm.models.location_choice
   :members:

.. index:: shadow pricing

.. _shadow_pricing:

Shadow Pricing
--------------

The shadow pricing calculator used by work and school location choice. 

.. automodule:: activitysim.abm.tables.shadow_pricing
   :members:

.. _auto_ownership:

Auto Ownership
--------------

The auto ownership model selects a number of autos for each household in the simulation. 
The primary model components are household demographics, zonal density, and accessibility.

The main interface to the auto ownership model is the 
:py:func:`~activitysim.abm.models.auto_ownership.auto_ownership_simulate` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.auto_ownership
   :members:

.. _freeparking:

Free Parking Eligibility
------------------------

The Free Parking Eligibility model predicts the availability of free parking at a person's
workplace.  It is applied for people who work in zones that have parking charges, which are 
generally located in the Central Business Districts. The purpose of the model is to adequately 
reflect the cost of driving to work in subsequent models, particularly in mode choice. 

The main interface to the free parking eligibility model is the 
:py:func:`~activitysim.abm.models.free_parking.free_parking` function.  This function is registered 
as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``free_parking_at_work`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.free_parking
   :members:


.. _cdap:

Coordinated Daily Activity Pattern
----------------------------------

The Coordinated Daily Activity Pattern (CDAP) model predicts the choice of daily activity pattern (DAP)
for each member in the household, simultaneously. The DAP is categorized in to three types as
follows:

  * Mandatory: the person engages in travel to at least one out-of-home mandatory activity - work, university, or school. The mandatory pattern may also include non-mandatory activities such as separate home-based tours or intermediate stops on mandatory tours.
  * Non-mandatory: the person engages in only maintenance and discretionary tours, which, by definition, do not contain mandatory activities.
  * Home: the person does not travel outside the home.

The CDAP model is a sequence of vectorized table operations:

* create a person level table and rank each person in the household for inclusion in the CDAP model.  Priority is given to full time workers (up to two), then to part time workers (up to two workers, of any type), then to children (youngest to oldest, up to three).  Additional members up to five are randomly included for the CDAP calculation.
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


.. automodule:: activitysim.abm.models.cdap
   :members:


.. _mandatory_tour_frequency:

Mandatory Tour Frequency
------------------------

The individual mandatory tour frequency model predicts the number of work and school tours
taken by each person with a mandatory DAP. The primary drivers of mandatory tour frequency
are demographics, accessibility-based parameters such as drive time to work, and household
automobile ownership.  It also creates mandatory tours in the data pipeline.

The main interface to the mandatory tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.mandatory_tour_frequency.mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``mandatory_tour_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.mandatory_tour_frequency
   :members:

.. _mandatory_tour_scheduling:

Mandatory Tour Scheduling
-------------------------

The mandatory tour scheduling model selects a tour departure and duration period (and therefore a 
start and end period as well) for each mandatory tour.   The primary drivers in the model are
accessibility-based parameters such as the mode choice logsum for the departure/arrival hour
combination, demographics, and time pattern characteristics such as the time windows available 
from previously scheduled tours. This model uses person :ref:`time_windows`.

The main interface to the mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.mandatory_scheduling.mandatory_tour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``TAZ, workplace_taz, school_taz, start, end``


.. automodule:: activitysim.abm.models.mandatory_scheduling
   :members:


.. _joint_tour_frequency:

Joint Tour Frequency
--------------------

The joint tour generation models are divided into three sub-models: the joint tour frequency
model, the party composition model, and the person participation model. In the joint tour
frequency model, the household chooses the purposes and number (up to two) of its fully joint
travel tours.  It also creates joints tours in the data pipeline.

The main interface to the joint tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.joint_tour_frequency.joint_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``households`` | Result Fields: ``num_hh_joint_tours`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.joint_tour_frequency
   :members:


.. _joint_tour_composition:

Joint Tour Composition
----------------------

In the joint tour party composition model, the makeup of the travel party (adults, children, or 
mixed - adults and children) is determined for each joint tour.  The party composition determines the 
general makeup of the party of participants in each joint tour in order to allow the micro-simulation 
to faithfully represent the prevalence of adult-only, children-only, and mixed joint travel tours 
for each purpose while permitting simplicity in the subsequent person participation model.

The main interface to the joint tour composition model is the 
:py:func:`~activitysim.abm.models.joint_tour_composition.joint_tour_composition` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Fields: ``composition`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.joint_tour_composition
   :members:


.. _joint_tour_participation:

Joint Tour Participation
------------------------

In the joint tour person participation model, each eligible person sequentially makes a 
choice to participate or not participate in each joint tour.  Since the party composition model 
determines what types of people are eligible to join a given tour, the person participation model 
can operate in an iterative fashion, with each household member choosing to join or not to join 
a travel party independent of the decisions of other household members. In the event that the 
constraints posed by the result of the party composition model are not met, the person 
participation model cycles through the household members multiple times until the required 
types of people have joined the travel party.

This step also creates the ``joint_tour_participants`` table in the pipeline, which stores the 
person ids for each person on the tour.

The main interface to the joint tour participation model is the 
:py:func:`~activitysim.abm.models.joint_tour_participation.joint_tour_participation` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Fields: ``number_of_participants, person_id (for the point person)`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.joint_tour_participation
   :members:


.. _joint_tour_destination_choice:

Joint Tour Destination Choice
-----------------------------

The joint tour destination choice model operate similarly to the usual work and
school location choice model, selecting the primary destination for travel tours. The only
procedural difference between the models is that the usual work and school location choice
model selects the usual location of an activity whether or not the activity is undertaken during the
travel day, while the joint tour destination choice model selects the location for an
activity which has already been generated.

The tour's primary destination is the location of the activity that is assumed to provide the greatest 
impetus for engaging in the travel tour. In the household survey, the primary destination was not asked, but
rather inferred from the pattern of stops in a closed loop in the respondents' travel diaries. The
inference was made by weighing multiple criteria including a defined hierarchy of purposes, the
duration of activities, and the distance from the tour origin. The model operates in the reverse
direction, designating the primary purpose and destination and then adding intermediate stops
based on spatial, temporal, and modal characteristics of the inbound and outbound journeys to
the primary destination.

The joint tour destination choice model is made up of three model steps:
  * sample - selects a sample of alternative locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative location.
  * simulate - starts with the table created above and chooses a final location, this time with the mode choice logsum included.

The main interface to the model is the :py:func:`~activitysim.abm.models.joint_tour_destination.joint_tour_destination`
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Fields: ``destination`` | Skims Keys: ``TAZ, alt_dest, MD time period``


.. automodule:: activitysim.abm.models.joint_tour_destination
   :members:

.. _joint_tour_scheduling:

Joint Tour Scheduling
---------------------

The joint tour scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each joint tour.  This model uses person :ref:`time_windows`. The primary drivers in the 
models are accessibility-based parameters such
as the auto travel time for the departure/arrival hour combination, demographics, and time
pattern characteristics such as the time windows available from previously scheduled tours.
The joint tour scheduling model does not use mode choice logsums. 

The main interface to the joint tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.joint_tour_scheduling.joint_tour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: `` TAZ, destination, MD time period, MD time period``


.. automodule:: activitysim.abm.models.joint_tour_scheduling
   :members:
   

.. _non_mandatory_tour_frequency:

Non-Mandatory Tour Frequency
----------------------------

The non-mandatory tour frequency model selects the number of non-mandatory tours made by each person on the simulation day.
It also adds non-mandatory tours to the tours in the data pipeline. The individual non-mandatory tour frequency model 
operates in two stages:

  * A choice is made using a random utility model between combinations of tours containing zero, one, and two or more escort tours, and between zero and one or more tours of each other purpose. 
  * Up to two additional tours of each purpose are added according to fixed extension probabilities.

The main interface to the non-mandatory tour purpose frequency model is the 
:py:func:`~activitysim.abm.models.non_mandatory_tour_frequency.non_mandatory_tour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``non_mandatory_tour_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.non_mandatory_tour_frequency
   :members:

.. _non_mandatory_tour_destination_choice:

Non-Mandatory Tour Destination Choice
-------------------------------------

The non-mandatory tour destination choice model chooses a destination zone for
non-mandatory tours.  The three step (sample, logsums, final choice) process also used for 
mandatory tour destination choice is used for non-mandatory tour destination choice.

The main interface to the non-mandatory tour destination choice model is the 
:py:func:`~activitysim.abm.models.non_mandatory_destination.non_mandatory_tour_destination` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``destination`` | Skims Keys: ``TAZ, alt_dest, MD time period, MD time period``


.. automodule:: activitysim.abm.models.non_mandatory_destination
   :members:
   

.. _non_mandatory_tour_scheduling:

Non-Mandatory Tour Scheduling
-----------------------------

The non-mandatory tour scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each non-mandatory tour.  This model uses person :ref:`time_windows`.  
The non-mandatory tour scheduling model does not use mode choice logsums. 

The main interface to the non-mandatory tour purpose scheduling model is the 
:py:func:`~activitysim.abm.models.non_mandatory_scheduling.non_mandatory_tour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``TAZ, destination, MD time period, MD time period``


.. automodule:: activitysim.abm.models.non_mandatory_scheduling
   :members:


.. _tour_mode_choice:

Tour Mode Choice
----------------

The mandatory, non-mandatory, and joint tour mode choice model assigns to each tour the "primary" mode that 
is used to get from the origin to the primary destination. The tour-based modeling approach requires a reconsideration
of the conventional mode choice structure. Instead of a single mode choice model used in a fourstep
structure, there are two different levels where the mode choice decision is modeled: (a) the
tour mode level (upper-level choice); and, (b) the trip mode level (lower-level choice conditional
upon the upper-level choice).

The mandatory, non-mandatory, and joint tour mode level represents the decisions that apply to the entire tour, and 
that will affect the alternatives available for each individual trip or joint trip. These decisions include the choice to use a private
car versus using public transit, walking, or biking; whether carpooling will be considered; and
whether transit will be accessed by car or by foot. Trip-level decisions correspond to details of
the exact mode used for each trip, which may or may not change over the trips in the tour.

The mandatory, non-mandatory, and joint tour mode choice structure is a nested logit model which separates 
similar modes into different nests to more accurately model the cross-elasticities between the alternatives. The
eighteen modes are incorporated into the nesting structure specified in the model settings file. The 
first level of nesting represents the use a private car, non-motorized
means, or transit. In the second level of nesting, the auto nest is divided into vehicle occupancy
categories, and transit is divided into walk access and drive access nests. The final level splits
the auto nests into free or pay alternatives and the transit nests into the specific line-haul modes.

The primary variables are in-vehicle time, other travel times, cost (the influence of which is derived
from the automobile in-vehicle time coefficient and the persons' modeled value of time),
characteristics of the destination zone, demographics, and the household's level of auto
ownership.

The main interface to the mandatory, non-mandatory, and joint tour tour mode model is the 
:py:func:`~activitysim.abm.models.tour_mode_choice.tour_mode_choice_simulate` function.  This function is 
called in the orca step ``tour_mode_choice_simulate`` and is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``mode`` | Skims Keys: ``TAZ, destination, start, end``

.. automodule:: activitysim.abm.models.tour_mode_choice
   :members:


.. _atwork_subtour_frequency:

At-work Subtours Frequency
--------------------------

The at-work subtour frequency model selects the number of at-work subtours made for each work tour.
It also creates at-work subtours by adding them to the tours table in the data pipeline.
These at-work sub-tours are travel tours taken during the workday with their origin at the work 
location, rather than from home. Explanatory variables include employment status,
income, auto ownership, the frequency of other tours, characteristics of the parent work tour, and
characteristics of the workplace zone.

Choosers: work tours
Alternatives: none, 1 eating out tour, 1 business tour, 1 maintenance tour, 2 business tours, 1 eating out tour + 1 business tour
Dependent tables: household, person, accessibility
Outputs: work tour subtour frequency choice, at-work tours table (with only tour origin zone at this point)

The main interface to the at-work subtours frequency model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_frequency.atwork_subtour_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``atwork_subtour_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.atwork_subtour_frequency
   :members:
   
.. _atwork_subtour_destination:

At-work Subtours Destination Choice
-----------------------------------

The at-work subtours destination choice model is made up of three model steps:

  * sample - selects a sample of alternative locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative location.
  * simulate - starts with the table created above and chooses a final location, this time with the mode choice logsum included.

Core Table: ``tours`` | Result Table: ``destination`` | Skims Keys: ``workplace_taz, alt_dest, MD time period``

The main interface to the at-work subtour destination model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination` 
function.  This function is registered as an orca step in the example Pipeline.

.. automodule:: activitysim.abm.models.atwork_subtour_destination
   :members:

.. _atwork_subtour_scheduling:

At-work Subtour Scheduling
--------------------------

The at-work subtours scheduling model selects a tour departure and duration period (and therefore a start and end 
period as well) for each at-work subtour.  This model uses person :ref:`time_windows`.

This model is the same as the mandatory tour scheduling model except it operates on the at-work tours and 
constrains the alternative set to available person :ref:`time_windows`.  The at-work subtour scheduling model does not use mode choice logsums. 
The at-work subtour frequency model can choose multiple tours so this model must process all first tours and then second 
tours since isFirstAtWorkTour is an explanatory variable.

Choosers: at-work tours
Alternatives: alternative departure time and arrival back at origin time pairs WITHIN the work tour departure time and arrival time back at origin AND the person time window. If no time window is available for the tour, make the first and last time periods within the work tour available, make the choice, and log the number of times this occurs.
Dependent tables: skims, person, land use, work tour
Outputs: at-work tour departure time and arrival back at origin time, updated person time windows

The main interface to the at-work subtours scheduling model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_scheduling.atwork_subtour_scheduling` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``workplace_taz, alt_dest, MD time period, MD time period``

.. automodule:: activitysim.abm.models.atwork_subtour_scheduling
   :members:


.. _atwork_subtour_mode_choice:

At-work Subtour Mode
--------------------

The at-work subtour mode choice model assigns a travel mode to each at-work subtour using the :ref:`tour_mode_choice` model.

The main interface to the at-work subtour mode choice model is the 
:py:func:`~activitysim.abm.models.atwork_subtour_mode_choice.atwork_subtour_mode_choice`
function.  This function is called in the orca step ``atwork_subtour_mode_choice`` and
is registered as an orca step in the example Pipeline.  

Core Table: ``tour`` | Result Field: ``tour_mode`` | Skims Keys: ``workplace_taz, destination, start, end``

.. automodule:: activitysim.abm.models.atwork_subtour_mode_choice
   :members:


.. _intermediate_stop_frequency:

Intermediate Stop Frequency
---------------------------

The stop frequency model assigns to each tour the number of intermediate destinations a person
will travel to on each leg of the tour from the origin to tour primary destination and back.
The model incorporates the ability for more than one stop in each direction,
up to a maximum of 3, for a total of 8 trips per tour (four on each tour leg).

Intermediate stops are not modeled for drive-transit tours because doing so can have unintended
consequences because of the difficulty of tracking the location of the vehicle. For example,
consider someone who used a park and ride for work and then took transit to an intermediate
shopping stop on the way home. Without knowing the vehicle location, it cannot be determined
if it is reasonable to allow the person to drive home. Even if the tour were constrained to allow
driving only on the first and final trip, the trip home from an intermediate stop may not use the
same park and ride where the car was dropped off on the outbound leg, which is usually as close
as possible to home because of the impracticality of coding drive access links from every park
and ride lot to every zone. 

This model also creates a trips table in the pipeline for later models.

The main interface to the intermediate stop frequency model is the 
:py:func:`~activitysim.abm.models.stop_frequency.stop_frequency` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``stop_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.stop_frequency
   :members:
   

.. _trip_purpose:

Trip Purpose
------------

For trip other than the last trip outbound or inbound, assign a purpose based on an 
observed frequency distribution.  The distribution is segmented by tour purpose, tour 
direction and person type. Work tours are also segmented by departure or arrival time period.

The main interface to the trip purpose model is the 
:py:func:`~activitysim.abm.models.trip_purpose.trip_purpose` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``purpose`` | Skims Keys: NA

.. note::
   Trip purpose and trip destination choice can be run iteratively together via :ref:`trip_purpose_and_destination`.
   

.. automodule:: activitysim.abm.models.trip_purpose
   :members:


.. _trip_destination_choice:

Trip Destination Choice
-----------------------

The trip (or stop) location choice model predicts the location of trips (or stops) along the tour other than the primary
destination. The stop-location model is structured as a multinomial logit model using a zone
attraction size variable and route deviation measure as impedance. The alternatives are sampled from
the full set of zones, subject to availability of a zonal attraction size term. The sampling mechanism
is also based on accessibility between tour origin and primary destination, and is subject to certain rules
based on tour mode. 

All destinations are available for auto tour modes, so long as there is a positive
size term for the zone. Intermediate stops on walk tours must be within X miles of both the tour
origin and primary destination zones. Intermediate stops on bike tours must be within X miles of both
the tour origin and primary destination zones. Intermediate stops on walk-transit tours must either be
within X miles walking distance of both the tour origin and primary destination, or have transit access to
both the tour origin and primary destination. Additionally, only short and long walk zones are
available destinations on walk-transit tours.

The intermediate stop location choice model works by cycling through stops on tours. The level-ofservice
variables (including mode choice logsums) are calculated as the additional utility between the
last location and the next known location on the tour. For example, the LOS variable for the first stop
on the outbound direction of the tour is based on additional impedance between the tour origin and the
tour primary destination. The LOS variable for the next outbound stop is based on the additional
impedance between the previous stop and the tour primary destination. Stops on return tour legs work
similarly, except that the location of the first stop is a function of the additional impedance between the
tour primary destination and the tour origin. The next stop location is based on the additional
impedance between the first stop on the return leg and the tour origin, and so on. 

The main interface to the trip destination choice model is the 
:py:func:`~activitysim.abm.models.trip_destination.trip_destination` function.  
This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``(trip) destination`` | Skims Keys: ``origin, (tour primary) destination, dest_taz, trip_period``

.. note::
   Trip purpose and trip destination choice can be run iteratively together via :ref:`trip_purpose_and_destination`.
   

.. automodule:: activitysim.abm.models.trip_destination
   :members:

.. _trip_purpose_and_destination:

Trip Purpose and Destination 
----------------------------

After running trip purpose and trip destination separately, the two model can be ran together in an iterative fashion on 
the remaining failed trips (i.e. trips that cannot be assigned a destination).  Each iteration uses new random numbers.

The main interface to the trip purpose model is the 
:py:func:`~activitysim.abm.models.trip_purpose_and_destination.trip_purpose_and_destination` 
function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``purpose, destination`` | Skims Keys: ``origin, (tour primary) destination, dest_taz, trip_period``

.. automodule:: activitysim.abm.models.trip_purpose_and_destination
   :members:


.. _trip_scheduling:

Trip Scheduling
---------------

For each trip, assign a departure hour based on an input lookup table of percents by tour purpose, 
direction (inbound/outbound), tour hour, and trip index.

  * The tour hour is the tour start hour for outbound trips and the tour end hour for inbound trips.  The trip index is the trip sequence on the tour, with up to four trips per half tour
  * For outbound trips, the trip depart hour must be greater than or equal to the previously selected trip depart hour
  * For inbound trips, trips are handled in reverse order from the next-to-last trip in the leg back to the first. The tour end hour serves as the anchor time point from which to start assigning trip time periods.
  * Outbound trips on at-work subtours are assigned the tour depart hour and inbound trips on at-work subtours are assigned the tour end hour.

The assignment of trip depart time is run iteratively up to a max number of iterations since it is possible that 
the time period selected for an earlier trip in a half-tour makes selection of a later trip time 
period impossible (or very low probability). Thus, the sampling is re-run until a feasible set of trip time 
periods is found. If a trip can't be scheduled after the max iterations, then the trip is assigned 
the previous trip's choice (i.e. assumed to happen right after the previous trip) or dropped, as configured by the user.
The trip scheduling model does not use mode choice logsums. 

Alternatives: Available time periods in the tour window (i.e. tour start and end period).  When processing stops on 
work tours, the available time periods is constrained by the at-work subtour start and end period as well.

The main interface to the trip scheduling model is the 
:py:func:`~activitysim.abm.models.trip_scheduling.trip_scheduling` function.  
This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``depart`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.trip_scheduling
   :members:
   
.. _trip_mode_choice:

Trip Mode Choice
----------------

The trip mode choice model assigns a travel mode for each trip on a given tour. It
operates similarly to the tour mode choice model, but only certain trip modes are available for
each tour mode. The correspondence rules are defined according to the following principles:

  * Pay trip modes are only available for pay tour modes (for example, drive-alone pay is only available at the trip mode level if drive-alone pay is selected as a tour mode).
  * The auto occupancy of the tour mode is determined by the maximum occupancy across all auto trips that make up the tour. Therefore, the auto occupancy for the tour mode is the maximum auto occupancy for any trip on the tour.
  * Transit tours can include auto shared-ride trips for particular legs. Therefore, 'casual carpool', wherein travelers share a ride to work and take transit back to the tour origin, is explicitly allowed in the tour/trip mode choice model structure.
  * The walk mode is allowed for any trip.
  * The availability of transit line-haul submodes on transit tours depends on the skimming and tour mode choice hierarchy. Free shared-ride modes are also available in walk-transit tours, albeit with a low probability. Paid shared-ride modes are not allowed on transit tours because no stated preference data is available on the sensitivity of transit riders to automobile value tolls, and no observed data is available to verify the number of people shifting into paid shared-ride trips on transit tours. 

The trip mode choice models explanatory variables include household and person variables, level-ofservice
between the trip origin and destination according to the time period for the tour leg, urban form
variables, and alternative-specific constants segmented by tour mode.

The main interface to the trip mode choice model is the 
:py:func:`~activitysim.abm.models.trip_mode_choice.trip_mode_choice` function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``trip_mode`` | Skims Keys: ``origin, destination, trip_period``

.. automodule:: activitysim.abm.models.trip_mode_choice
   :members:
   
.. _trip_cbd_parking:

Trip CBD Parking
----------------

**NOT YET IMPLEMENTED**

The parking location choice model is applied to tours with a destination in the urban area/city center
with parking charges. The model incorporates three of the following interrelated sub-models to
capture the current parking conditions in CBD, and allows for testing various policies:

  * Parking cost model: determines the average cost of parking in each CBD zone.
  * Person-free parking eligibility model: determines if each worker pays for parking in the CBD.
  * Parking location choice model: determines for each tour the primary destination parking location zone. 
  
The nested logit structure consists of an upper level binary choice between parking inside versus outside 
the modeled destination zone. At the lower level, the choice of parking zone is modeled for those who did 
not park in the destination zone.   

The main interface to the CBD parking model is the 
XXXXX function.  This function is registered as an orca step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``XXXXX`` | Skims Keys: ``XXXXX``


.. _utility_steps:

Util
----
 
Additional helper classes

CDAP
~~~~

.. automodule:: activitysim.abm.models.util.cdap
   :members:

.. index:: table annotation
.. _table_annotation:

Expressions
~~~~~~~~~~~

The expressions class is often used for pre- and post-processor table annotation, which read a CSV file of expression, calculate 
a number of additional table fields, and join the fields to the target table.  An example table annotation expressions 
file is found in the example configuration files for households for the CDAP model - 
`annotate_households_cdap.csv <https://github.com/activitysim/activitysim/blob/master/example/configs/annotate_households_cdap.csv>`__. 

.. automodule:: activitysim.abm.models.util.expressions
   :members:

Logsums
~~~~~~~
      
.. automodule:: activitysim.abm.models.util.logsums
   :members:

Mode
~~~~

.. automodule:: activitysim.abm.models.util.mode
   :members:

Overlap
~~~~~~~
      
.. automodule:: activitysim.abm.models.util.overlap
   :members:

Tour Destination 
~~~~~~~~~~~~~~~~

.. automodule:: activitysim.abm.models.util.tour_destination
   :members:

Tour Frequency
~~~~~~~~~~~~~~

.. automodule:: activitysim.abm.models.util.tour_frequency
   :members:

Trip
~~~~

.. automodule:: activitysim.abm.models.util.trip
   :members:
   
Vectorize Tour Scheduling 
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: activitysim.abm.models.util.vectorize_tour_scheduling
   :members:

Tests
-----
 
See ``activitysim.abm.test`` and ``activitysim.abm.models.util.test``
