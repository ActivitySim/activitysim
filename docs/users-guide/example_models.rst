
.. index:: tutorial
.. index:: example
.. _example :
.. _examples :

Example Models
==============

The ActivitySim consortium maintains two supported "canonical" example
implementations:

- the `SANDAG Model <https://github.com/ActivitySim/sandag-abm3-example>`__ is a two-zone
  model based on the SANDAG ABM3 model, and
- the `MTC Model <https://github.com/ActivitySim/activitysim-prototype-mtc>`__ is a
  one-zone model based on the MTC's Travel Model One.

Each example implementation includes a complete set of model components, input data,
and configuration files, and is intended to serve as a reference for users to build
their own models. They are provided as stand-alone repositories, to highlight the
fact that model implementations are separate from the ActivitySim core codebase,
and to make it easier for users to fork and modify the examples for their own use
without needing to modify the ActivitySim core codebase. The examples are maintained
by the ActivitySim Consortium and are kept up-to-date with the latest version of
ActivitySim.

.. note:

    The two example models are not identical to the original agency models from which
    they were created. They are generally similar to those models, and have been calibrated
    and validated to reproduce reasonable results. They are intended to demonstrate the
    capabilities of ActivitySim and to provide a starting point for users to build their own
    models. However, they are not intended to be used as-is for policy analysis or forecasting.

A discussion of the runtime performance of the example models is available in the
:ref:`example performance benchmarking <example-performance>` section.

This page describes the structure of the MTC example model in more detail.

.. _prototype_mtc :

prototype_mtc
-------------

Introduction
~~~~~~~~~~~~

The initial example implemented in ActivitySim was prototype_mtc.  This section described the prototype_mtc
model design, how to setup and run the example, and how to review outputs. The default configuration of the
example is limited to a small sample of households and zones so that it can be run quickly and require
less than 1 GB of RAM.  The full scale example can be configured and run as well.

Model Design
~~~~~~~~~~~~

Overview
________

The prototype_mtc example is based on (but has evolved away from) the
`Bay Area Metro Travel Model One <https://github.com/BayAreaMetro/travel-model-one>`__ (TM1).
TM1 has its roots in a wide array of analytical approaches, including discrete
choice forms (multinomial and nested logit models), activity duration models, time-use models,
models of individual micro-simulation with constraints, entropy-maximization models, etc.
These tools are combined in the model design to realistically represent travel behavior,
adequately replicate observed activity-travel patterns, and ensure model sensitivity to
infrastructure and policies. The model is implemented in a micro-simulation framework. Microsimulation
methods capture aggregate outcomes through the representation of the behavior of
individual decision-makers.

Zone System
___________

The prototype MTC model uses the 1454 TAZ zone system developed for the MTC trip-based model.  The zones are fairly large for the region,
which may somewhat distort the representation of transit access in mode choice. To ameliorate this problem, the
original model zones were further sub-divided into three categories of transit access: short walk, long walk, and not
walkable.  However, support for transit subzones is not included in the activitysim implementation since the latest generation
of activity-based models typically use an improved approach to spatial representation called multiple zone systems.  See
:ref:`multiple_zone_systems` for more information.

Decision-making units
_____________________

Decision-makers in the model system are households and persons. These decision-makers are
created for each simulation year based on a population synthesis process such as
`PopulationSim <https://github.com/ActivitySim/PopulationSim>`__. The decision-makers are used in the
subsequent discrete-choice models to select a single alternative from a list of available
alternatives according to a probability distribution. The probability distribution is generated
from various logit-form models which take into account the attributes of the decision-maker and
the attributes of the various alternatives. The decision-making unit is an important element of
model estimation and implementation, and is explicitly identified for each model.

Person type segmentation
________________________

TM1 is implemented in a micro-simulation framework. A key advantage of the
micro-simulation approach is that there are essentially no computational constraints on the
number of explanatory variables which can be included in a model specification. However, even
with this flexibility, the model system includes some segmentation of decision-makers.
Segmentation is a useful tool both to structure models and also as a way to characterize person
roles within a household.

The person types shown below are used for the example model. The person types are mutually exclusive
with respect to age, work status, and school status.

+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| Person Type Code | Person Type                                               | Age     | Work Status      | School Status |
+==================+===========================================================+=========+==================+===============+
| 1                | Full-time worker (30+ hours a week)                       | 18+     | Full-time        | None          |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 2                | Part-time worker (<30 hours but works on a regular basis) | 18+     | Part-time        | None          |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 3                | College student                                           | 18+     | Any              | College       |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 4                | Non-working adult                                         | 18 - 64 | Unemployed       | None          |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 5                | Retired person                                            | 65+     | Unemployed       | None          |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 6                | Driving age student                                       | 16 - 17 | Any              | Pre-college   |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 7                | Non-driving student                                       | 6 - 16  | None             | Pre-college   |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+
| 8                | Pre-school child                                          | 0 - 5   | None             | Preschool     |
+------------------+-----------------------------------------------------------+---------+------------------+---------------+

Household type segments are useful for pre-defining certain data items (such as destination
choice size terms) so that these data items can be pre-calculated for each segment. Precalculation
of these data items reduces model complexity and runtime. The segmentation is based on household income,
and includes four segments - low, medium, high, very high.

In the model, the persons in each household are assigned a simulated but fixed value of time
that modulates the relative weight the decision-maker places on time and cost. The probability
distribution from which the value of time is sampled was derived from a toll choice model
estimated using data from a stated preference survey performed for the SFCTA Mobility, Access, and
Pricing Study, and is a lognormal distribution with a mean that varies by income segment.

Activity type segmentation
__________________________

The activity types are used in most model system components, from developing daily activity patterns
and to predicting tour and trip destinations and modes by purpose.  The set of activity types is shown below.
The activity types are also grouped according to whether the activity is mandatory or non-mandatory and
eligibility requirements are assigned determining which person-types can be used for generating each
activity type. The classification scheme of each activity type reflects the relative importance or
natural hierarchy of the activity, where work and school activities are typically the most inflexible
in terms of generation, scheduling and location, and discretionary activities are typically the most
flexible on each of these dimensions. Each out-of-home location that a person travels to in the
simulation is assigned one of these activity types.

+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Purpose             | Description                                                              | Classification| Eligibility                           |
+=====================+==========================================================================+===============+=======================================+
| Work                | Working at regular workplace or work-related activities outside the home | Mandatory     | Workers and students                  |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| University          | College or university                                                    | Mandatory     | Age 18+                               |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| High School         | Grades 9-12                                                              | Mandatory     | Age 14-17                             |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Grade School        | Grades preschool, K-8                                                    | Mandatory     | Age 0-13                              |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Escorting           | Pick-up/drop-off passengers (auto trips only)                            | NonMandatory  | Age 16+                               |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Shopping            | Shopping away from home                                                  | NonMandatory  | Age 5+ (if joint travel, all persons) |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Other Maintenance   | Personal business/services and medical appointments                      | NonMandatory  | Age 5+ (if joint travel, all persons) |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Social/Recreational | Recreation, visiting friends/family                                      | NonMandatory  | Age 5+ (if joint travel, all persons) |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Eat Out             | Eating outside of home                                                   | NonMandatory  | Age 5+ (if joint travel, all persons) |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+
| Other Discretionary | Volunteer work, religious activities                                     | NonMandatory  | Age 5+ (if joint travel, all persons) |
+---------------------+--------------------------------------------------------------------------+---------------+---------------------------------------+

Treatment of time
_________________

The TM1 example model system functions at a temporal resolution of one hour. These one hour increments
begin with 3 AM and end with 3 AM the next day. Temporal integrity is ensured so that no
activities are scheduled with conflicting time windows, with the exception of short
activities/tours that are completed within a one hour increment. For example, a person may have
a short tour that begins and ends within the 8 AM to 9 AM period, as well as a second longer tour
that begins within this time period, but ends later in the day.

A critical aspect of the model system is the relationship between the temporal resolution used for
scheduling activities and the temporal resolution of the network assignment periods. Although
each activity generated by the model system is identified with a start time and end time in one hour
increments, LOS matrices are only created for five aggregate time periods. The trips occurring in each time period
reference the appropriate transport network depending on their trip mode and the mid-point trip
time. The definition of time periods for LOS matrices is given below.

+---------------+------------+
|  Time Period  | Start Hour |
+===============+============+
|  EA           |  3         |
+---------------+------------+
|  AM           |  5         |
+---------------+------------+
|  MD           |  9         |
+---------------+------------+
|  PM           |  14        |
+---------------+------------+
|  EV           |  18        |
+---------------+------------+

Trip modes
__________

The trip modes defined in the example model are below. The modes include auto by
occupancy and toll/non-toll choice, walk and bike, walk and drive access to five different
transit line-haul modes, and ride hail with taxi, single TNC (Transportation Network Company), and shared TNC.

  * Auto

    * SOV Free
    * SOV Pay
    * 2 Person Free
    * 2 Person Pay
    * 3+ Person Free
    * 3+ Person Pay

  * Nonmotorized

    * Walk
    * Bike

  * Transit

    * Walk

      * Walk to Local Bus
      * Walk to Light-Rail Transit
      * Walk to Express Bus
      * Walk to Bus Rapid Transit
      * Walk to Heavy Rail

    * Drive

      * Drive to Local Bus
      * Drive to Light-Rail Transit
      * Drive to Express Bus
      * Drive to Bus Rapid Transit
      * Drive to Heavy Rail

  * Ride Hail

    * Taxi
    * Single TNC
    * Shared TNC

Sub-models
__________

The general design of the prototype_mtc model is presented below.  Long-term choices that relate to
the usual workplace/university/school for each worker and student, household car ownership, and the
availability of free parking at workplaces are first.

The coordinated daily activity pattern type of each household member is the first travel-related
sub-model in the hierarchy. This model classifies daily patterns by three types:

  * Mandatory, which includes at least one out-of-home mandatory activity (work or school)
  * Non-mandatory, which includes at least one out-of-home non-mandatory activity, but does not include out-of-home mandatory activities
  * Home, which does not include any out-of-home activity or travel

The pattern type sub-model leaves open the frequency of tours for mandatory and nonmandatory
purposes since these sub-models are applied later in the model sequence. Daily
pattern-type choices of the household members are linked in such a way that decisions made by
members are reflected in the decisions made by the other members.

After the frequency and time-of-day for work and school tours are determined, the
next major model component relates to joint household travel. This component produces a
number of joint tours by travel purpose for the entire household, travel party composition
in terms of adults and children, and then defines the participation of each household
member in each joint household tour. It is followed by choice of destination and time-of-day.

The next stage relates to maintenance and discretionary tours that are modeled at the individual
person level. The models include tour frequency, choice of destination and time
of day. Next, a set of sub-models relate tour-level details on mode, exact number of
intermediate stops on each half-tour and stop location. It is followed by the last set of
sub-models that add details for each trip including trip departure time, trip mode details and parking
location for auto trips.

.. image:: images/abmexample.jpg

The output of the model is a disggregate table of trips with individual attributes for custom analysis.  The trips
can be aggregated into travel demand matrices for network loading.

Setup
~~~~~

The following describes the prototype_mtc model setup.


Folder and File Setup

The prototype_mtc has the following root folder/file setup:

  * configs - settings, expressions files, etc.
  * configs_mp - override settings for the multiprocess configuration
  * data - input data such as land use, synthetic population files, and network LOS / skims
  * output - outputs folder

Inputs
______

In order to run prototype_mtc, you first need the input files in the ``data`` folder as identified in the ``configs\settings.yaml`` file and the ``configs\network_los.yaml`` file:

* input_table_list: the input CSV tables for MTC (see below for column definitions):

    * households - Synthetic population household records for a subset of zones.
    * persons - Synthetic population person records for a subset of zones.
    * land_use - Zone-based land use data (population and employment for example) for a subset of zones.

* taz_skims: skims.omx - an OMX matrix file containing the MTC TM1 skim matrices for a subset of zones.  The time period for the matrix must be represented at the end of the matrix name and be seperated by a double_underscore (e.g. BUS_IVT__AM indicates base skim BUS_IVT with a time period of AM).

These files are used in the tests as well.  The full set
of MTC households, persons, and OMX skims are on the ActivitySim `resources repository <https://github.com/rsginc/activitysim_resources>`__.



Additional details on these files is available in the original `Travel Model 1 repository <https://github.com/BayAreaMetro/modeling-website/wiki/DataDictionary>`_,
although many of the files described there are not used in ActivitySim.

Data Schema
+++++++++++

The following table lists the pipeline data tables, field names, the data types and other details associated with each table in the model:


+----------------------------+-------------------------------+---------+------------------------------+------+------+
| Table                      | Field                         | DType   | Creator                      |NCol  |NRow  |
+============================+===============================+=========+==============================+======+======+
| accessibility              | auPkRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auPkTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auOpRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auOpTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trPkRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trPkTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trOpRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trOpTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | nmRetail                      | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | nmTotal                       | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | TAZ                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | SERIALNO                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | PUMA5                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hhsize                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | HHT                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | UNITTYPE                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | NOC                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | BLDGSZ                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | TENURE                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | VEHICL                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hinccat1                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hinccat2                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hhagecat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hsizecat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hfamily                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hunittype                     | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hNOCcat                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwrkrcat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h0004                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h0511                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1215                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1617                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1824                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h2534                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h3549                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h5064                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h6579                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h80up                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_workers                   | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwork_f                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwork_p                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | huniv                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hnwork                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hretire                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hpresch                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hschpred                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hschdriv                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | htypdwel                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hownrent                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadnwst                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadwpst                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadkids                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | bucketBin                     | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | originalPUMA                  | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hmultiunit                    | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | chunk_id                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income_in_thousands           | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income_segment                | int32   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | median_value_of_time          | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hh_value_of_time              | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_non_workers               | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_drivers                   | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_adults                    | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children                  | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_young_children            | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children_5_to_15          | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children_16_to_17         | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_college_age               | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_young_adults              | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | non_family                    | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | family                        | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | home_is_urban                 | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | home_is_rural                 | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | auto_ownership                | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hh_work_auto_savings_ratio    | float32 | workplace_location           | 66   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_under16_not_at_school     | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active             | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_adults      | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_preschoolers| int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_children    | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 |num_travel_active_non_presch   | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | participates_in_jtf_model     | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | joint_tour_frequency          | object  | joint_tour_frequency         | 75   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_hh_joint_tours            | int8    | joint_tour_frequency         | 75   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | tour_id                       | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | household_id                  | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | person_id                     | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | participant_num               | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | DISTRICT                      | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SD                            | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | county_id                     | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTHH                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHPOP                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTPOP                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | EMPRES                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SFDU                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | MFDU                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ1                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ2                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ3                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ4                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTACRE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | RESACRE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | CIACRE                        | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SHPOP62P                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTEMP                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE0004                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE0519                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE2044                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE4564                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE65P                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | RETEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | FPSEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HEREMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | OTHEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGREMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | MWTEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | PRKCST                        | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | OPRKCST                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | area_type                     | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HSENROLL                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | COLLFTE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | COLLPTE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOPOLOGY                      | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TERMINAL                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | ZERO                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | hhlds                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | sftaz                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | gqpop                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | household_density             | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | employment_density            | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | density_index                 | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 4                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 5                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 6                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 7                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 8                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 9                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 10                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 11                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 12                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 13                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 14                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 15                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 16                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 17                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 18                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 19                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 20                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 21                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 22                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 23                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 24                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | household_id                  | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | RELATE                        | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | ESR                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | GRADE                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | PNUM                          | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | PAUG                          | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | DDP                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | sex                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | WEEKS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | HOURS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | MSP                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | POVERTY                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | EARNS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pagecat                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pemploy                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pstudent                      | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | ptype                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | padkid                        | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age_16_to_19                  | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age_16_p                      | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | adult                         | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | male                          | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | female                        | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_non_worker                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_retiree                   | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_preschool_kid             | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_driving_kid               | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_school_kid                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_full_time                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_part_time                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_university                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | student_is_employed           | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | nonstudent_to_school          | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_student                    | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_gradeschool                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_highschool                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_university                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | school_segment                | int8    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_worker                     | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | home_taz                      | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | value_of_time                 | float64 | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | school_taz                    | int32   | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | distance_to_school            | float32 | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | roundtrip_auto_time_to_school | float32 | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | workplace_taz                 | int32   | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | distance_to_work              | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | workplace_in_cbd              | bool    | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_zone_area_type           | float64 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | roundtrip_auto_time_to_work   | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_auto_savings             | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_auto_savings_ratio       | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | free_parking_at_work          | bool    | free_parking                 | 53   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | cdap_activity                 | object  | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | cdap_rank                     | int64   | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | travel_active                 | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | under16_not_at_school         | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_preschool_kid_at_home     | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_school_kid_at_home        | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | mandatory_tour_frequency      | object  | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_and_school_and_worker    | bool    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_and_school_and_student   | bool    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_mand                      | int8    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_work_tours                | int8    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_joint_tours               | int8    | joint_tour_participation     | 65   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | non_mandatory_tour_frequency  | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_non_mand                  | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_escort_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_eatout_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_shop_tours                | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_maint_tours               | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_discr_tours               | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_social_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_non_escort_tours          | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | gradeschool                   | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | highschool                    | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | university                    | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | gradeschool                   | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | highschool                    | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | university                    | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | person_id                     | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type                     | object  | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type_count               | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type_num                 | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_num                      | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_count                    | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_category                 | object  | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | number_of_participants        | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | destination                   | int32   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | origin                        | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | household_id                  | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | start                         | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | end                           | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | duration                      | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tdd                           | int64   | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | composition                   | object  | joint_tour_composition       | 16   | 159  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_mode                     | object  | tour_mode_choice_simulate    | 17   | 319  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | atwork_subtour_frequency      | object  | atwork_subtour_frequency     | 19   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | parent_tour_id                | float64 | atwork_subtour_frequency     | 19   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | stop_frequency                | object  | stop_frequency               | 21   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | primary_purpose               | object  | stop_frequency               | 21   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | person_id                     | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | household_id                  | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | tour_id                       | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | primary_purpose               | object  | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_num                      | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | outbound                      | bool    | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_count                    | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | purpose                       | object  | trip_purpose                 | 8    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | destination                   | int32   | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | origin                        | int32   | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | failed                        | bool    | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | depart                        | float64 | trip_scheduling              | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_mode                     | object  | trip_mode_choice             | 12   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_high                     | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_low                      | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_med                      | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_veryhigh                 | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_high                     | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_low                      | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_med                      | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_veryhigh                 | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+





Households
++++++++++

The households table contains the following synthetic population columns:

* household_id: numeric ID of this household, used in persons table to join with household characteristics
* TAZ: zone where this household lives
* income: Annual household income, in 2000 dollars
* hhsize: Household size
* HHT: Household type (see below)
* auto_ownership: number of cars owned by this household (0-6)
* num_workers: number of workers in the household
* sample_rate

Household types
"""""""""""""""

These are household types defined by the Census Bureau and used in `ACS table B11001 <https://censusreporter.org/tables/B11001/>`_.

+------+------------------------------------------+
| Code | Description                              |
+======+==========================================+
| 0    | None                                     |
+------+------------------------------------------+
| 1    | Married-couple family                    |
+------+------------------------------------------+
| 2    | Male householder, no spouse present      |
+------+------------------------------------------+
| 3    | Female householder, no spouse present    |
+------+------------------------------------------+
| 4    | Nonfamily household, male alone          |
+------+------------------------------------------+
| 5    | Nonfamily household, male not alone      |
+------+------------------------------------------+
| 6    | Nonfamily household, female alone        |
+------+------------------------------------------+
| 7    | Nonfamily household, female not alone    |
+------+------------------------------------------+


Persons
+++++++

This table describes attributes of the persons that constitute each household. This file contains the following columns:

* person_id: Unique integer identifier for each person. This value is globally unique, i.e.
  no two individuals have the same person ID, even if they are in different households.
* household_id: Household identifier for this person, foreign key to households table
* age: Age in years
* PNUM: Person number in household, starting from 1.
* sex: Sex, 1 = Male, 2 = Female
* pemploy: Employment status (see below)
* pstudent: Student status (see below)
* ptype: Person type (see person type segmentation above)

Employment status
"""""""""""""""""

+------+------------------------------------------+
| Code | Description                              |
+======+==========================================+
| 1    | Full-time worker                         |
+------+------------------------------------------+
| 2    | Part-time worker                         |
+------+------------------------------------------+
| 3    | Not in labor force                       |
+------+------------------------------------------+
| 4    | Student under 16                         |
+------+------------------------------------------+

Student status
""""""""""""""

+------+------------------------------------------+
| Code | Description                              |
+======+==========================================+
| 1    | Preschool through Grade 12 student       |
+------+------------------------------------------+
| 2    | University/professional school student   |
+------+------------------------------------------+
| 3    | Not a student                            |
+------+------------------------------------------+

Land use
++++++++

All values are raw numbers and not proportions of the total.

* TAZ: Zone which this row describes
* DISTRICT: Superdistrict where this TAZ is (34 superdistricts in the Bay Area)
* SD: Duplicate of DISTRICT
* COUNTY: County within the Bay Area (see below)
* TOTHH: Total households in TAZ
* TOTPOP: Total population in TAZ
* TOTACRE: Area of TAZ, acres
* RESACRE: Residential area of TAZ, acres
* CIACRE: Commercial/industrial area of TAZ, acres
* TOTEMP: Total employment
* AGE0519: Persons age 5 to 19 (inclusive)
* RETEMPN: NAICS-based total retail employment
* FPSEMPN: NAICS-based financial and professional services employment
* HEREMPN: NAICS-based health, education, and recreational service employment
* AGREMPN: NAICS-based agricultural and natural resources employment
* MWTEMPN: NAICS-based manufacturing and wholesale trade employment
* OTHEMP: NAICS-based other employment
* PRKCST: Hourly cost paid by long-term (8+ hours) parkers, year 2000 cents
* OPRKCST: Hourly cost paid by short term parkers, year 2000 cents
* area_type: Area type designation (see below)
* HSENROLL: High school students enrolled at schools in this TAZ
* COLLFTE: College students enrolled full-time at colleges in this TAZ
* COLLPTE: College students enrolled part-time at colleges in this TAZ
* TERMINAL: Average time to travel from automobile storage location to origin/destination (floating-point minutes)

Counties
""""""""

+------+------------------------------------------+
| Code | Name                                     |
+======+==========================================+
| 1    | San Francisco                            |
+------+------------------------------------------+
| 2    | San Mateo                                |
+------+------------------------------------------+
| 3    | Santa Clara                              |
+------+------------------------------------------+
| 4    | Alameda                                  |
+------+------------------------------------------+
| 5    | Contra Costa                             |
+------+------------------------------------------+
| 6    | Solano                                   |
+------+------------------------------------------+
| 7    | Napa                                     |
+------+------------------------------------------+
| 8    | Sonoma                                   |
+------+------------------------------------------+
| 9    | Marin                                    |
+------+------------------------------------------+

Area types
""""""""""

+------+------------------------------------------+
| Code | Description                              |
+======+==========================================+
| 0    | Regional core                            |
+------+------------------------------------------+
| 1    | Central business district                |
+------+------------------------------------------+
| 2    | Urban business                           |
+------+------------------------------------------+
| 3    | Urban                                    |
+------+------------------------------------------+
| 4    | Suburban                                 |
+------+------------------------------------------+
| 5    | Rural                                    |
+------+------------------------------------------+

.. note::

  ActivitySim can optionally build an HDF5 file of the input CSV tables for use in subsequent runs since
  HDF5 is binary and therefore results in faster read times. see :ref:`configuration`

  OMX and HDF5 files can be viewed with the `OMX Viewer <https://github.com/osPlanning/omx/wiki/OMX-Viewer>`__ or
  `HDFView <https://www.hdfgroup.org/downloads/hdfview>`__.

  The ``other_resources\scripts\build_omx.py`` script will build one OMX file containing all the skims. The original MTC TM1 skims were converted for the prototype from
  Cube to OMX using the ``other_resources\scripts\mtc_tm1_omx_export.s`` script.

  The prototype_mtc_sf inputs were created by the ``other_resources\scripts\create_sf_example.py`` script, which creates the land use, synthetic population, and
  skim inputs for a subset of user-defined zones.


.. index:: skims
.. index:: omx_file
.. index:: skim matrices
.. _skims:

Skims
_____

The injectables and omx_file for the example are listed below.
The skims are float64 matrix.

Skims are named <PATH TYPE>_<MEASURE>__<TIME PERIOD>:

* Highway paths:

  * SOV - SOV free
  * HOV2 - HOV2 free
  * HOV3 - HOV3 free
  * SOVTOLL - SOV toll
  * HOV2TOLL - HOV2 toll
  * HOV3TOLL - HOV3 toll

* Transit paths:

  * Walk access and walk egress - WLK_COM_WLK, WLK_EXP_WLK, WLK_HVY_WLK, WLK_LOC_WLK, WLK_LRF_WLK
  * Walk access and drive egress - WLK_COM_DRV, WLK_EXP_DRV, WLK_HVY_DRV, WLK_LOC_DRV, WLK_LRF_DRV
  * Drive access and walk egress - DRV_COM_WLK, DRV_EXP_WLK, DRV_HVY_WLK, DRV_LOC_WLK, DRV_LRF_WLK
  * COM = commuter rail, EXP = express bus, HVY = heavy rail, LOC = local bus, LRF = light rail ferry

* Non-motorized paths:

  * WALK
  * BIKE

* Measures:

  * TIME - Time (minutes)
  * DIST - Distance (miles)
  * BTOLL - Bridge toll (cents)
  * VTOLL - Value toll (cents)

  * IVT - In-vehicle time, time (minutes x 100)
  * IWAIT - Initial wait time, time (minutes x 100)
  * XWAIT - Transfer wait time, time (minutes x 100)
  * WACC - Walk access time, time (minutes x 100)
  * WAUX - Auxiliary walk time, time (minutes x 100)
  * WEGR - Walk egress time, time (minutes x 100)
  * DTIM - Drive access and/or egress time, time (minutes x 100)
  * DDIST - Drive access and/or egress distance, distance (miles x 100)
  * FAR - Fare, cents
  * BOARDS - Boardings, number
  * TOTIVT - Total in-vehicle time, time (minutes x 100)
  * KEYIVT - Transit submode in-vehicle time, time (minutes x 100)
  * FERRYIVT - Ferry in-vehicle time, time (minutes x 100)

* Time periods:

  * EA
  * AM
  * MD
  * PM
  * EV

+------------------------------+-----------------+
|                        Field |            Type |
+==============================+=================+
|                 SOV_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                    \DIST\    |  float64 matrix |
+------------------------------+-----------------+
|                \DISTWALK\    |  float64 matrix |
+------------------------------+-----------------+
|                \DISTBIKE\    |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__PM |  float64 matrix |
+------------------------------+-----------------+




Configuration
_____________

This section has been moved to :ref:`user_configuration`.

.. _sub-model-spec-files:

Sub-Model Specification Files
_____________________________

Included in the ``configs`` folder are the model specification files that store the
Python/pandas/numpy expressions, alternatives, and other settings used by each model.  Some models includes an
alternatives file since the alternatives are not easily described as columns in the expressions file.  An example
of this is the ``non_mandatory_tour_frequency_alternatives.csv`` file, which lists each alternative as a row and each
columns indicates the number of non-mandatory tours by purpose.  The  set of files for the prototype_mtc are below.  The
:ref:`prototype_arc`, :ref:`prototype_semcog`, and :ref:`prototype_mtc_extended` examples added additional submodels.

+------------------------------------------------+--------------------------------------------------------------------+
|            Model                               |    Specification Files                                             |
+================================================+====================================================================+
|  :ref:`initialize_landuse`                     |  - initialize_landuse.yaml                                         |
|                                                |  - annotate_landuse.csv                                            |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`accessibility`                          |  - accessibility.yaml                                              |
|                                                |  - accessibility.csv                                               |
+------------------------------------------------+--------------------------------------------------------------------+
|                                                |  - initialize_households.yaml                                      |
|  :ref:`initialize_households`                  |  - annotate_persons.csv                                            |
|                                                |  - annotate_households.csv                                         |
|                                                |  - annotate_persons_after_hh.csv                                   |
+------------------------------------------------+--------------------------------------------------------------------+
|   :ref:`school_location`                       |  - school_location.yaml                                            |
|                                                |  - school_location_coeffs.csv                                      |
|                                                |  - annotate_persons_school.csv                                     |
|                                                |  - school_location_sample.csv                                      |
|                                                |  - tour_mode_choice.yaml (and related files)                       |
|                                                |  - school_location.csv                                             |
|                                                |  - destination_choice_size_terms.csv                               |
|                                                |  - shadow_pricing.yaml                                             |
+------------------------------------------------+--------------------------------------------------------------------+
|    :ref:`work_location`                        |  - workplace_location.yaml                                         |
|                                                |  - workplace_location_coeffs.csv                                   |
|                                                |  - annotate_persons_workplace.csv                                  |
|                                                |  - annotate_households_workplace.csv                               |
|                                                |  - workplace_location_sample.csv                                   |
|                                                |  - tour_mode_choice.yaml (and related files)                       |
|                                                |  - workplace_location.csv                                          |
|                                                |  - destination_choice_size_terms.csv                               |
|                                                |  - shadow_pricing.yaml                                             |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`auto_ownership`                          |  - auto_ownership.yaml                                             |
|                                                |  - auto_ownership_coeffs.csv                                       |
|                                                |  - auto_ownership.csv                                              |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`freeparking`                             |  - free_parking.yaml                                               |
|                                                |  - free_parking_coeffs.csv                                         |
|                                                |  - free_parking.csv                                                |
|                                                |  - free_parking_annotate_persons_preprocessor.csv                  |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`cdap`                                    |  - cdap.yaml                                                       |
|                                                |  - annotate_persons_cdap.csv                                       |
|                                                |  - annotate_households_cdap.csv                                    |
|                                                |  - cdap_indiv_and_hhsize1.csv                                      |
|                                                |  - cdap_interaction_coefficients.csv                               |
|                                                |  - cdap_fixed_relative_proportions.csv                             |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`mandatory_tour_frequency`               |  - mandatory_tour_frequency.yaml                                   |
|                                                |  - mandatory_tour_frequency_coeffs.csv                             |
|                                                |  - mandatory_tour_frequency.csv                                    |
|                                                |  - mandatory_tour_frequency_alternatives.csv                       |
|                                                |  - annotate_persons_mtf.csv                                        |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`mandatory_tour_scheduling`               |  - mandatory_tour_scheduling.yaml                                  |
|                                                |  - tour_scheduling_work_coeffs.csv                                 |
|                                                |  - tour_scheduling_work.csv                                        |
|                                                |  - tour_scheduling_school.csv                                      |
|                                                |  - tour_departure_and_duration_alternatives.csv                    |
|                                                |  - tour_departure_and_duration_segments.csv                        |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`joint_tour_frequency`                    |  - joint_tour_frequency.yaml                                       |
|                                                |  - joint_tour_frequency_coeffs.csv                                 |
|                                                |  - annotate_persons_jtf.csv                                        |
|                                                |  - joint_tour_frequency_annotate_households_preprocessor.csv       |
|                                                |  - joint_tour_frequency_alternatives.csv                           |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`joint_tour_composition`                  |  - joint_tour_composition.yaml                                     |
|                                                |  - joint_tour_composition_coefficients.csv                         |
|                                                |  - joint_tour_composition_annotate_households_preprocessor.csv     |
|                                                |  - joint_tour_composition.csv                                      |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`joint_tour_participation`                |  - joint_tour_participation.yaml                                   |
|                                                |  - joint_tour_participation_coefficients.csv                       |
|                                                |  - joint_tour_participation_annotate_participants_preprocessor.csv |
|                                                |  - joint_tour_participation.csv                                    |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`joint_tour_destination_choice`           |  - joint_tour_destination.yaml                                     |
|                                                |  - non_mandatory_tour_destination_coefficients.csv                 |
|                                                |  - non_mandatory_tour_destination_sample.csv                       |
|                                                |  - non_mandatory_tour_destination.csv                              |
|                                                |  - tour_mode_choice.yaml (and related files)                       |
|                                                |  - destination_choice_size_terms.csv                               |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`joint_tour_scheduling`                   |  - joint_tour_scheduling.yaml                                      |
|                                                |  - tour_scheduling_joint_coefficients.csv                          |
|                                                |  - joint_tour_scheduling_annotate_tours_preprocessor.csv           |
|                                                |  - tour_scheduling_joint.csv                                       |
|                                                |  - tour_departure_and_duration_alternatives.csv                    |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`non_mandatory_tour_frequency`            |  - non_mandatory_tour_frequency.yaml                               |
|                                                |  - non_mandatory_tour_frequency_coefficients_{ptype}.csv           |
|                                                |  - non_mandatory_tour_frequency.csv                                |
|                                                |  - non_mandatory_tour_frequency_alternatives.csv                   |
|                                                |  - non_mandatory_tour_frequency_annotate_persons_preprocessor.csv  |
|                                                |  - non_mandatory_tour_frequency_extension_probs.csv                |
|                                                |  - annotate_persons_nmtf.csv                                       |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`non_mandatory_tour_destination_choice`   |  - non_mandatory_tour_destination.yaml                             |
|                                                |  - non_mandatory_tour_destination_coefficients.csv                 |
|                                                |  - non_mandatory_tour_destination.csv                              |
|                                                |  - non_mandatory_tour_destination_sample.csv                       |
|                                                |  - tour_mode_choice.yaml (and related files)                       |
|                                                |  - destination_choice_size_terms.csv                               |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`non_mandatory_tour_scheduling`           |  - non_mandatory_tour_scheduling.yaml                              |
|                                                |  - tour_scheduling_nonmandatory_coefficients.csv                   |
|                                                |  - non_mandatory_tour_scheduling_annotate_tours_preprocessor.csv   |
|                                                |  - tour_scheduling_nonmandatory.csv                                |
|                                                |  - tour_departure_and_duration_alternatives.csv                    |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`tour_mode_choice`                        |  - tour_mode_choice.yaml                                           |
|                                                |  - tour_mode_choice_annotate_choosers_preprocessor.csv             |
|                                                |  - tour_mode_choice.csv                                            |
|                                                |  - tour_mode_choice_coefficients.csv                               |
|                                                |  - tour_mode_choice_coeffs_template.csv                            |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`atwork_subtour_frequency`               |  - atwork_subtour_frequency.yaml                                   |
|                                                |  - atwork_subtour_frequency_coefficients.csv                       |
|                                                |  - atwork_subtour_frequency.csv                                    |
|                                                |  - atwork_subtour_frequency_alternatives.csv                       |
|                                                |  - atwork_subtour_frequency_annotate_tours_preprocessor.csv        |
+------------------------------------------------+--------------------------------------------------------------------+
|   :ref:`atwork_subtour_destination`            |  - atwork_subtour_destination.yaml                                 |
|                                                |  - atwork_subtour_destination_coefficients.csv                     |
|                                                |  - atwork_subtour_destination_sample.csv                           |
|                                                |  - atwork_subtour_destination.csv                                  |
|                                                |  - tour_mode_choice.yaml (and related files)                       |
|                                                |  - destination_choice_size_terms.csv                               |
+------------------------------------------------+--------------------------------------------------------------------+
| :ref:`atwork_subtour_scheduling`               |  - tour_scheduling_atwork.yaml                                     |
|                                                |  - tour_scheduling_atwork_coefficients.csv                         |
|                                                |  - tour_scheduling_atwork.csv                                      |
|                                                |  - tour_scheduling_atwork_preprocessor.csv                         |
|                                                |  - tour_departure_and_duration_alternatives.csv                    |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`atwork_subtour_mode_choice`             |  - tour_mode_choice.yaml (and related files)                       |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`intermediate_stop_frequency`            |  - stop_frequency.yaml                                             |
|                                                |  - stop_frequency_annotate_tours_preprocessor.csv                  |
|                                                |  - stop_frequency_alternatives.csv                                 |
|                                                |  - stop_frequency_atwork.csv                                       |
|                                                |  - stop_frequency_eatout.csv                                       |
|                                                |  - stop_frequency_escort.csv                                       |
|                                                |  - stop_frequency_othdiscr.csv                                     |
|                                                |  - stop_frequency_othmaint.csv                                     |
|                                                |  - stop_frequency_school.csv                                       |
|                                                |  - stop_frequency_shopping.csv                                     |
|                                                |  - stop_frequency_social.csv                                       |
|                                                |  - stop_frequency_subtour.csv                                      |
|                                                |  - stop_frequency_univ.csv                                         |
|                                                |  - stop_frequency_work.csv                                         |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`trip_purpose`                           |  - trip_purpose.yaml (+ trip_purpose_and_destination.yaml)         |
|                                                |  - trip_purpose_annotate_trips_preprocessor.csv                    |
|                                                |  - trip_purpose_probs.csv                                          |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`trip_destination_choice`                |  - trip_destination.yaml (+ trip_purpose_and_destination.yaml)     |
|                                                |  - trip_destination.csv                                            |
|                                                |  - trip_destination_annotate_trips_preprocessor.csv                |
|                                                |  - trip_destination_sample.csv                                     |
|                                                |  - trip_mode_choice.yaml (and related files)                       |
|                                                |  - destination_choice_size_terms.csv                               |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`trip_scheduling`                        |  - trip_scheduling.yaml                                            |
|                                                |  - trip_scheduling_probs.csv                                       |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`trip_mode_choice`                       |  - trip_mode_choice.yaml                                           |
|                                                |  - trip_mode_choice_annotate_trips_preprocessor.csv                |
|                                                |  - trip_mode_choice_coefficients.csv                               |
|                                                |  - trip_mode_choice.csv                                            |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`parking_location_choice`                |  - parking_location_choice.yaml                                    |
|                                                |  - parking_location_choice_annotate_trips_preprocessor.csv         |
|                                                |  - parking_location_choice_coeffs.csv                              |
|                                                |  - parking_location_choice.csv                                     |
+------------------------------------------------+--------------------------------------------------------------------+
|  :ref:`write_trip_matrices`                    |  - write_trip_matrices.yaml                                        |
|                                                |  - write_trip_matrices_annotate_trips_preprocessor.csv             |
+------------------------------------------------+--------------------------------------------------------------------+

.. _model_example_run :

Running the model
~~~~~~~~~~~~~~~~~

To run the example, do the following:

* Activate the correct conda environment if needed
* View the list of available examples

::

  activitysim create --list

* Create a local copy of an example folder

::

  activitysim create --example prototype_mtc --destination my_test_example

* Run the example

::

  cd my_test_example
  activitysim run -c configs -d data -o output


* ActivitySim will log progress and write outputs to the output folder.

The example should run in a few minutes since it runs a small sample of households.

.. note::

  A customizable run script for power users can be found in the `Github repo <https://github.com/ActivitySim/activitysim/tree/main/other_resources/scripts>`__.
  This script takes many of the same arguments as the ``activitysim run`` command, including paths to
  ``--config``, ``--data``, and ``--output`` directories. The script looks for these folders in the current
  working directory by default.

  ::

    python simulation.py

.. _multiprocess_example :

Multiprocessing
_______________

The model system is parallelized via :ref:`multiprocessing`.  To setup and run the :ref:`example` using
multiprocessing, follow the same steps as the above :ref:`model_example_run`, but add an additional ``-c`` flag to
include the multiprocessing configuration settings via settings file inheritance (see :ref:`cli`) as well:

::

  activitysim run -c configs_mp -c configs -d data -o output

The multiprocessing example also writes outputs to the output folder.

The default multiprocessed example is configured to run with two processors and chunking training: ``num_processes: 2``,
``chunk_size: 0``, and ``chunk_training_mode: training``.  Additional more performant configurations are included and
commented out in the example settings file.  For example, the 100 percent sample full scale multiprocessing example
- ``prototype_mtc_full`` - was run on a Windows Server machine with 28 cores and 256GB RAM with the configuration below.
The default setup runs with ``chunk_training_mode: training`` since no chunk cache file is present. To run the example
significantly faster, try ``chunk_training_mode: disabled`` if the machine has sufficient RAM, or try
``chunk_training_mode: production``.  To configure ``chunk_training_mode: production``, first configure chunking as
discussed below. See :ref:`multiprocessing` and :ref:`chunk_size` for more information.

::

  households_sample_size: 0
  num_processes: 24
  chunk_size: 0
  chunk_training_mode: production


.. _configuring_chunking :

Configuring chunking
____________________

To configure chunking, ActivitySim must first be trained to determine reasonable chunking settings given the
model setup and machine.  The steps to configure chunking are:

* Run the full scale model with ``chunk_training_mode: training``.
  Set ``num_processors`` to about 80% of the available physical processors
  and ``chunk_size`` to about 80% of the available RAM.  This will run the model
  and create the ``chunk_cache.csv`` file in the output\cache directory for reuse.
* The ``households_sample_size`` for training chunking should be at least 1 / num_processors
  to provide sufficient data for training and the ``chunk_method: hybrid_uss``
  typically performs best.
* Run the full scale model with ``chunk_training_mode: production``.  Experiment
  with different ``num_processors`` and ``chunk_size`` settings depending on desired
  runtimes and machine resources.

See :ref:`chunk_size` for more information.  Users can run ``chunk_training_mode: disabled`` if the machine has an abundance of RAM for the model setup.

Outputs
~~~~~~~

The key output of ActivitySim is the HDF5 data pipeline file ``outputs\pipeline.h5``.  By default, this datastore file
contains a copy of each data table after each model step in which the table was modified.

The example also writes the final tables to CSV files by using the ``write_tables`` step.  This step calls
:func:`activitysim.core.pipeline.get_table` to get a pandas DataFrame and write a CSV file for each table
specified in ``output_tables`` in the settings.yaml file.

::

  output_tables:
    h5_store: False
    action: include
    prefix: final_
    tables:
      - checkpoints
      - accessibility
      - land_use
      - households
      - persons
      - tours
      - trips
      - joint_tour_participants



The ``other_resources\scripts\make_pipeline_output.py`` script uses the information stored in the pipeline file to create
the table below for a small sample of households.  The table shows that for each table in the pipeline, the number of rows
and/or columns changes as a result of the relevant model step.  A ``checkpoints`` table is also stored in the
pipeline, which contains the crosswalk between model steps and table states in order to reload tables for
restarting the pipeline at any step.

+-----------------------------------+------------------------------------+------+------+
| Table                             | Creator                            | NRow | NCol |
+===================================+====================================+======+======+
| accessibility                     | compute_accessibility              | 1454 | 10   |
+-----------------------------------+------------------------------------+------+------+
| households                        | initialize                         | 100  | 65   |
+-----------------------------------+------------------------------------+------+------+
| households                        | workplace_location                 | 100  | 66   |
+-----------------------------------+------------------------------------+------+------+
| households                        | cdap_simulate                      | 100  | 73   |
+-----------------------------------+------------------------------------+------+------+
| households                        | joint_tour_frequency               | 100  | 75   |
+-----------------------------------+------------------------------------+------+------+
| joint_tour_participants           | joint_tour_participation           | 13   | 4    |
+-----------------------------------+------------------------------------+------+------+
| land_use                          | initialize_landuse                 | 1454 | 44   |
+-----------------------------------+------------------------------------+------+------+
| person_windows                    | initialize_households              | 271  | 21   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | initialize_households              | 271  | 42   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | school_location                    | 271  | 45   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | workplace_location                 | 271  | 52   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | free_parking                       | 271  | 53   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | cdap_simulate                      | 271  | 59   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | mandatory_tour_frequency           | 271  | 64   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | joint_tour_participation           | 271  | 65   |
+-----------------------------------+------------------------------------+------+------+
| persons                           | non_mandatory_tour_frequency       | 271  | 74   |
+-----------------------------------+------------------------------------+------+------+
| school_destination_size           | initialize_households              | 1454 | 3    |
+-----------------------------------+------------------------------------+------+------+
| school_modeled_size               | school_location                    | 1454 | 3    |
+-----------------------------------+------------------------------------+------+------+
| tours                             | mandatory_tour_frequency           | 153  | 11   |
+-----------------------------------+------------------------------------+------+------+
| tours                             | mandatory_tour_scheduling          | 153  | 15   |
+-----------------------------------+------------------------------------+------+------+
| tours                             | joint_tour_composition             | 159  | 16   |
+-----------------------------------+------------------------------------+------+------+
| tours                             | tour_mode_choice_simulate          | 319  | 17   |
+-----------------------------------+------------------------------------+------+------+
| tours                             | atwork_subtour_frequency           | 344  | 19   |
+-----------------------------------+------------------------------------+------+------+
| tours                             | stop_frequency                     | 344  | 21   |
+-----------------------------------+------------------------------------+------+------+
| trips                             | stop_frequency                     | 859  | 7    |
+-----------------------------------+------------------------------------+------+------+
| trips                             | trip_purpose                       | 859  | 8    |
+-----------------------------------+------------------------------------+------+------+
| trips                             | trip_destination                   | 859  | 11   |
+-----------------------------------+------------------------------------+------+------+
| trips                             | trip_scheduling                    | 859  | 11   |
+-----------------------------------+------------------------------------+------+------+
| trips                             | trip_mode_choice                   | 859  | 12   |
+-----------------------------------+------------------------------------+------+------+
| workplace_destination_size        | initialize_households              | 1454 | 4    |
+-----------------------------------+------------------------------------+------+------+
| workplace_modeled_size            | workplace_location                 | 1454 | 4    |
+-----------------------------------+------------------------------------+------+------+

.. index:: logs
.. _logs :

Logging
_______

Included in the ``configs`` folder is the ``logging.yaml``, which configures Python logging
library.  The following key log files are created with a model run:

* ``activitysim.log`` - overall system log file
* ``timing_log.csv`` - submodel step runtimes
* ``omnibus_mem.csv`` - multiprocessed submodel memory usage

Refer to the :ref:`tracing` section for more detail on tracing.

Trip Matrices
_____________

The ``write_trip_matrices`` step processes the trips table to create open matrix (OMX) trip matrices for
assignment.  The matrices are configured and coded according to the expressions in the model step
trip annotation file.  See :ref:`write_trip_matrices` for more information.

.. _tracing :

Tracing
_______

There are two types of tracing in ActivtiySim: household and origin-destination (OD) pair.  If a household trace ID
is specified, then ActivitySim will output a comprehensive set (i.e. hundreds) of trace files for all
calculations for all household members:

* ``Several CSV files`` - each input, intermediate, and output data table - chooser, expressions/utilities, probabilities, choices, etc. - for the trace household for each sub-model

If an OD pair trace is specified, then ActivitySim will output the acessibility calculations trace
file:

* ``accessibility.result.csv`` - accessibility expression results for the OD pair

With the set of output CSV files, the user can trace ActivitySim calculations in order to ensure they are correct and/or to
help debug data and/or logic errors.

.. _writing_logsums :

Writing Logsums
_______________

The tour and trip destination and mode choice models calculate logsums but do not persist them by default.
Mode and destination choice logsums are essential for re-estimating these models and can therefore be
saved to the pipeline if desired.  To save the tour and trip destination and mode choice model logsums, include
the following optional settings in the model settings file.  The data is saved to the pipeline for later use.

::

  # in workplace_location.yaml for example
  DEST_CHOICE_LOGSUM_COLUMN_NAME: workplace_location_logsum
  DEST_CHOICE_SAMPLE_TABLE_NAME: workplace_location_sample

  # in tour_mode_choice.yaml for example
  MODE_CHOICE_LOGSUM_COLUMN_NAME: mode_choice_logsum

The `DEST_CHOICE_SAMPLE_TABLE_NAME` contains the fields in the table below.  Writing out the
destination choice sample table, which includes the mode choice logsum for each sampled
alternative destination, adds significant size to the pipeline.  Therefore, this feature should
only be activated when writing logsums for a small set of households for model estimation.

+-----------------------------------+---------------------------------------+
| Field                             | Description                           |
+===================================+=======================================+
| chooser_id                        | chooser id such as person or tour id  |
+-----------------------------------+---------------------------------------+
| alt_dest                          | destination alternative id            |
+-----------------------------------+---------------------------------------+
| prob                              | alternative probability               |
+-----------------------------------+---------------------------------------+
| pick_count                        | sampling with replacement pick count  |
+-----------------------------------+---------------------------------------+
| mode_choice_logsum                | mode choice logsum                    |
+-----------------------------------+---------------------------------------+
