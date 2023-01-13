
.. index:: models
.. _models:

Models
======

The currently implemented example ActivitySim AB models are described below.  See the example
model :ref:`sub-model-spec-files`, :ref:`arc-sub-model-spec-files`, and :ref:`semcog-sub-model-spec-files` for more information.

.. _initialize_landuse:
.. _initialize_households:
.. _initialize_tours:

Initialize
----------

The initialize model isn't really a model, but rather a few data processing steps in the data pipeline.
The initialize data processing steps code variables used in downstream models, such as household and person
value-of-time.  This step also pre-loads the land_use, households, persons, and person_windows tables because
random seeds are set differently for each step and therefore the sampling of households depends on which step
they are initially loaded in.

The main interface to the initialize land use step is the :py:func:`~activitysim.abm.models.initialize.initialize_landuse`
function. The main interface to the initialize household step is the :py:func:`~activitysim.abm.models.initialize.initialize_households`
function.  The main interface to the initialize tours step is the :py:func:`~activitysim.abm.models.initialize_tours.initialize_tours`
function.  These functions are registered as Inject steps in the example Pipeline.

.. automodule:: activitysim.abm.models.initialize
   :members:

.. automodule:: activitysim.abm.models.initialize_tours
   :members:

.. _initialize_los:
.. _initialize_tvpb:

Initialize LOS
--------------

The initialize LOS model isn't really a model, but rather a series of data processing steps in the data pipeline.
The initialize LOS model does two things:

  * Loads skims and cache for later if desired
  * Loads network LOS inputs for transit virtual path building (see :ref:`transit_virtual_path_builder`), pre-computes tap-to-tap total utilities and cache for later if desired

The main interface to the initialize LOS step is the :py:func:`~activitysim.abm.models.initialize_los.initialize_los`
function.  The main interface to the initialize TVPB step is the :py:func:`~activitysim.abm.models.initialize_los.initialize_tvpb`
function.  These functions are registered as Inject steps in the example Pipeline.

.. automodule:: activitysim.abm.models.initialize_los
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
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``skims`` | Result Table: ``accessibility`` | Skims Keys: ``O-D, D-O``


.. automodule:: activitysim.abm.models.accessibility
   :members:


.. _disaggregate_accessibility:

Disaggregate Accessibility
--------------

The disaggregate accessibility model is an extension of the base accessibility model.
While the base accessibility model is based on a mode-specific decay function and uses fixed market
segments in the population (i.e., income), the disaggregate accessibility model extracts the actual
destination choice logsums by purpose (i.e., mandatory fixed school/work location and non-mandatory
tour destinations by purpose) from the actual model calculations using a user-defined proto-population.
This enables users to include features that may be more critical to destination
choice than just income (e.g., automobile ownership).


Inputs:
  * disaggregate_accessibility.yaml - Configuration settings for disaggregate accessibility model.
  * annotate.csv [optional] - Users can specify additional annotations specific to disaggregate accessibility. For example, annotating the proto-population tables.

Outputs:
  * final_disaggregate_accessibility.csv [optional]
  * final_non_mandatory_tour_destination_accesibility.csv [optional]
  * final_workplace_location_accessibility.csv [optional]
  * final_school_location_accessibility.csv [optional]
  * final_proto_persons.csv [optional]
  * final_proto_households.csv [optional]
  * final_proto_tours.csv [optional]

The above tables are created in the model pipeline, but the model will not save
any outputs unless specified in settings.yaml - output_tables. Users can return
the proto population tables for inspection, as well as the raw logsum accessibilities
for mandatory school/work and non-mandatory destinations. The logsums are then merged
at the household level in final_disaggregate_accessibility.csv, which each tour purpose
logsums shown as separate columns.


**Usage**
The disaggregate accessibility model is run as a model step in the model list.
There are two necessary steps:

``- initialize_proto_population`` | ``- compute_disaggregate_accessibility``

The reason the steps must be separate is to enable multiprocessing.
The proto-population must be fully generated and initialized before activitysim
slices the tables into separate threads. These steps must also occur before
initialize_households in order to avoid conflict with the shadow_pricing model.


The model steps can be run either as part the activitysim model run, or setup
to run as a standalone run to pre-computing the accessibility values.
For standalone implementations, the final_disaggregate_accessibility.csv is read
into the pipeline and initialized with the initialize_household model step.


**Configuration of disaggregate_accessibility.yaml:**
  * CREATE_TABLES - Users define the variables to be generated for PROTO_HOUSEHOLDS, PROTO_PERSONS, and PROTO_TOURS tables. These tables must include all basic fields necessary for running the actual model. Additional fields can be annotated in pre-processing using the annotation settings of this file. The base variables in each table are defined using the following parameters:

    - VARIABLES - The base variable, must be a value or a list. Results in the cartesian product (all non-repeating combinations) of the fields.
    - mapped_fields [optional] - For non-combinatorial fields, users can map a variable to the fields generated in VARIABLES (e.g., income category bins mapped to median dollar values).
    - filter_rows [optional] - Users can also filter rows using pandas expressions if specific variable combinations are not desired.
    - JOIN_ON [required only for PROTO_TOURS] - specify the persons variable to join the tours to (e.g., person_number).
  * MERGE_ON - User specified fields to merge the proto-population logsums onto the full synthetic population. The proto-population should be designed such that the logsums are able to be joined exactly on these variables specified to the full population. Users specify the to join on using:

    - by: An exact merge will be attempted using these discrete variables.
    - asof [optional]: The model can peform an "asof" join for continuous variables, which finds the nearest value. This method should not be necessary since synthetic populations are all discrete.

    - method [optional]: Optional join method can be "soft", default is None. For cases where a full inner join is not possible, a Naive Bayes clustering method is fast but discretely constrained method. The proto-population is treated as the "training data" to match the synthetic population value to the best possible proto-population candidate. The Some refinement may be necessary to make this procedure work.

  * annotate_proto_tables [optional] - Annotation configurations if users which to modify the proto-population beyond basic generation in the YAML.
  * DESTINATION_SAMPLE_SIZE - The *destination* sample size (0 = all zones), e.g., the number of destination zone alternatives sampled for calculating the destination logsum. Decimal values < 1 will be interpreted as a percentage, e.g., 0.5 = 50% sample.
  * ORIGIN_SAMPLE_SIZE - The *origin* sample size (0 = all zones), e.g., the number of origins where logsum is calculated. Origins without a logsum will draw from the nearest zone with a logsum. This parameter is useful for systems with a large number of zones with similar accessibility. Decimal values < 1 will be interpreted as a percentage, e.g., 0.5 = 50% sample.
  * ORIGIN_SAMPLE_METHOD - The method in which origins are sampled. Population weighted sampling can be TAZ-based or "TAZ-agnostic" using KMeans clustering. The potential advantage of KMeans is to provide a more geographically even spread of MAZs sampled that do not rely on TAZ hierarchies. Unweighted sampling is also possible using 'uniform' and 'uniform-taz'.

    - None [Default] - Sample zones weighted by population, ensuring at least one TAZ is sampled per MAZ. If n-samples > n-tazs then sample 1 MAZ from each TAZ until n-remaining-samples < n-tazs, then sample n-remaining-samples TAZs and sample an MAZ within each of those TAZs. If n-samples < n-tazs, then it proceeds to the above 'then' condition.

    - "kmeans" - K-Means clustering is performed on the zone centroids (must be provided as maz_centroids.csv), weighted by population. The clustering yields k XY coordinates weighted by zone population for n-samples = k-clusters specified. Once k new cluster centroids are found, these are then approximated into the nearest available zone centroid and used to calculate accessibilities on. By default, the k-means method is run on 10 different initial cluster seeds (n_init) using using "k-means++" seeding algorithm (https://en.wikipedia.org/wiki/K-means%2B%2B). The k-means method runs for max_iter iterations (default=300).

    - "uniform" - Unweighted sample of N zones independent of each other.

    - "uniform-taz" - Unweighted sample of 1 zone per taz up to the N samples specified.


.. _work_from_home:

Work From Home
--------------

Telecommuting is defined as workers who work from home instead of going
to work. It only applies to workers with a regular workplace outside of home.
The telecommute model consists of two submodels - this work from home model and a
person :ref:`telecommute_frequency` model. This model predicts for all workers whether they
usually work from home.

The work from home model includes the ability to adjust a work from home alternative
constant to attempt to realize a work from home percent for what-if type analysis.
This iterative single process procedure takes as input a number of iterations, a filter on
the choosers to use for the calculation, a target work from home percent, a tolerance percent
for convergence, and the name of the coefficient to adjust.  An example setup is provided and
the coefficient adjustment at each iteration is:
``new_coefficient = log( target_percent / current_percent ) + current_coefficient``.

The main interface to the work from home model is the
:py:func:`~activitysim.abm.models.work_from_home` function.  This
function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``work_from_home`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.work_from_home
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

School location choice for :ref:`multiple_zone_systems` models uses :ref:`presampling` by default.

The main interfaces to the model is the :py:func:`~activitysim.abm.models.location_choice.school_location` function.
This function is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums` for how to write logsums for estimation.

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

Work location choice for :ref:`multiple_zone_systems` models uses :ref:`presampling` by default.

The main interfaces to the model is the :py:func:`~activitysim.abm.models.location_choice.workplace_location` function.
This function is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums` for how to write logsums for estimation.

Core Table: ``persons`` | Result Field: ``workplace_taz`` | Skims Keys: ``TAZ, alt_dest, AM time period, PM time period``


.. automodule:: activitysim.abm.models.location_choice
   :members:

.. index:: shadow pricing

.. _shadow_pricing:

Shadow Pricing
--------------

The shadow pricing calculator used by work and school location choice.

**Turning on and saving shadow prices**

Shadow pricing is activated by setting the ``use_shadow_pricing`` to True in the settings.yaml file.
Once this setting has been activated, ActivitySim will search for shadow pricing configuration in
the shadow_pricing.yaml file. When shadow pricing is activated, the shadow pricing outputs will be
exported by the tracing engine. As a result, the shadow pricing output files will be prepended with
``trace`` followed by the iteration number the results represent. For example, the shadow pricing
outputs for iteration 3 of the school location model will be called
``trace.shadow_price_school_shadow_prices_3.csv``.

In total, ActivitySim generates three types of output files for each model with shadow pricing:

- ``trace.shadow_price_<model>_desired_size.csv`` The size terms by zone that the ctramp and daysim
  methods are attempting to target. These equal the size term columns in the land use data
  multiplied by size term coefficients.

- ``trace.shadow_price_<model>_modeled_size_<iteration>.csv`` These are the modeled size terms after
  the iteration of shadow pricing identified by the <iteration> number. In other words, these are
  the predicted choices by zone and segment for the model after the iteration completes. (Not
  applicable for ``simulation`` option.)

- ``trace.shadow_price_<model>_shadow_prices_<iteration>.csv`` The actual shadow price for each zone
  and segment after the <iteration> of shadow pricing. This is the file that can be used to warm
  start the shadow pricing mechanism in ActivitySim. (Not applicable for ``simulation`` option.)

There are three shadow pricing methods in activitysim: ``ctramp``, ``daysim``, and ``simulation``.
The first two methods try to match model output with workplace/school location model size terms,
while the last method matches model output with actual employment/enrollmment data.

The simulation approach operates the following steps.  First, every worker / student will be
assigned without shadow prices applied. The modeled share and the target share for each zone are
compared. If the zone is overassigned, a sample of people from the over-assigned zones will be
selected for re-simulation.  Shadow prices are set to -999 for the next iteration for overassigned
zones which removes the zone from the set of alternatives in the next iteration. The sampled people
will then be forced to choose from one of the under-assigned zones that still have the initial
shadow price of 0. (In this approach, the shadow price variable is really just a switch turning that
zone on or off for selection in the subsequent iterations. For this reason, warm-start functionality
for this approach is not applicable.)  This process repeats until the overall convergence criteria
is met or the maximum number of allowed iterations is reached.

Because the simulation approach only re-simulates workers / students who were over-assigned in the
previous iteration, run time is significantly less (~90%) than the CTRAMP or DaySim approaches which
re-simulate all workers and students at each iteration.

**shadow_pricing.yaml Attributes**

- ``shadow_pricing_models`` List model_selectors and model_names of models that use shadow pricing.
  This list identifies which size_terms to preload which must be done in single process mode, so
  predicted_size tables can be scaled to population
- ``LOAD_SAVED_SHADOW_PRICES`` global switch to enable/disable loading of saved shadow prices. From
  the above example, this would be trace.shadow_price_<model>_shadow_prices_<iteration>.csv renamed
  and stored in the ``data_dir``.
- ``MAX_ITERATIONS`` If no loaded shadow prices, maximum number of times shadow pricing can be run
  on each model before proceeding to the next model.
- ``MAX_ITERATIONS_SAVED`` If loaded shadow prices, maximum number of times shadow pricing can be
  run.
- ``SIZE_THRESHOLD`` Ignore zones in failure calculation (ctramp or daysim method) with smaller size
  term value than size_threshold.
- ``TARGET_THRESHOLD`` Ignore zones in failure calculation (simulation method) with smaller
  employment/enrollment than target_threshold.
- ``PERCENT_TOLERANCE`` Maximum percent difference between modeled and desired size terms
- ``FAIL_THRESHOLD`` percentage of zones exceeding the PERCENT_TOLERANCE considered a failure
- ``SHADOW_PRICE_METHOD`` [ctramp | daysim | simulation]
- ``workplace_segmentation_targets`` dict matching school segment to landuse employment column
  target. Only used as part of simulation option. If mutiple segments list the same target column,
  the segments will be added together for comparison. (Same with the school option below.)
- ``school_segmentation_targets`` dict matching school segment to landuse enrollment column target.
  Only used as part of simulation option.
- ``DAMPING_FACTOR`` On each iteration, ActivitySim will attempt to adjust the model to match
  desired size terms. The number is multiplied by adjustment factor to dampen or amplify the
  ActivitySim calculation. (only for CTRAMP)
- ``DAYSIM_ABSOLUTE_TOLERANCE`` Absolute tolerance for DaySim option
- ``DAYSIM_PERCENT_TOLERANCE`` Relative tolerance for DaySim option
- ``WRITE_ITERATION_CHOICES`` [True | False ] Writes the choices of each person out to the trace
  folder. Used for debugging or checking itration convergence. WARNING: every person is written for
  each sub-process so the disc space can get large.


.. automodule:: activitysim.abm.tables.shadow_pricing
   :members:

.. _transit_pass_subsidy:

Transit Pass Subsidy
--------------------

The transit fare discount model is defined as persons who purchase or are
provided a transit pass.  The transit fare discount consists of two submodels - this
transit pass subsidy model and a person :ref:`transit_pass_ownership` model.  The
result of this model can be used to condition downstream models such as the
person :ref:`transit_pass_ownership` model and the tour and trip mode choice models
via fare discount adjustments.

The main interface to the transit pass subsidy model is the
:py:func:`~activitysim.abm.models.transit_pass_subsidy` function.  This
function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``transit_pass_subsidy`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.transit_pass_subsidy
   :members:

.. _transit_pass_ownership:

Transit Pass Ownership
----------------------

The transit fare discount is defined as persons who purchase or are
provided a transit pass.  The transit fare discount consists of two submodels - this
transit pass ownership model and a person :ref:`transit_pass_subsidy` model. The
result of this model can be used to condition downstream models such as the tour and trip
mode choice models via fare discount adjustments.

The main interface to the transit pass ownership model is the
:py:func:`~activitysim.abm.models.transit_pass_ownership` function.  This
function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``transit_pass_ownership`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.transit_pass_ownership
   :members:

.. _auto_ownership:

Auto Ownership
--------------

The auto ownership model selects a number of autos for each household in the simulation.
The primary model components are household demographics, zonal density, and accessibility.

The main interface to the auto ownership model is the
:py:func:`~activitysim.abm.models.auto_ownership.auto_ownership_simulate`
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``households`` | Result Field: ``auto_ownership`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.auto_ownership
   :members:

.. _vehicle_type_choice:

Vehicle Type Choice
-------------------

The vehicle type choice model selects a vehicle type for each household vehicle. A vehicle type
is a combination of the vehicle's body type, age, and fuel type.  For example, a 13 year old
gas powered van would have a vehicle type of *van_13_gas*.

There are two vehicle type choice model structures implemented:

1. Simultaneous choice of body type, age, and fuel type.
2. Simultaneous choice of body type and age, with fuel type assigned from a probability distribution.

The *vehicle_type_choice.yaml* file contains the following model specific options:

* ``SPEC``: Filename for input utility expressions
* ``COEFS``: Filename for input utility expression coefficients
* ``LOGIT_TYPE``: Specifies whether you are using a nested or multinomial logit structure
* ``combinatorial_alts``: Specifies the alternatives for the choice model.
  Has sub-categories of ``body_type``, ``age``, and ``fuel_type``.
* ``PROBS_SPEC``: Filename for input fuel type probabilities. Supplying probabilities
  corresponds to implementation structure 2 above, and not supplying probabilities would correspond to implementation structure 1.
  If provided, the ``fuel_type`` category in ``combinatorial_alts``
  will be excluded from the model alternatives such that only body type and age are selected.  Input ``PROBS_SPEC`` table will have an index
  column named *vehicle_type* which is a combination of body type and age in the form ``{body type}_{age}``.  Subsequent column names
  specify the fuel type that will be added and the column values are the probabilities of that fuel type.
  The vehicle type model will select a fuel type for each vehicle based on the provided probabilities.
* ``VEHICLE_TYPE_DATA_FILE``: Filename for input vehicle type data. Must have columns ``body_type``, ``fuel_type``, and ``vehicle_year``.
  Vehicle ``age`` is computed using the ``FLEET_YEAR`` option. Data for every alternative specified in the ``combinatorial_alts`` option must be included
  in the file. Vehicle type data file will be joined to the alternatives and can be used in the utility expressions if ``PROBS_SPEC`` is not provided.
  If ``PROBS_SPEC`` is provided, the vehicle type data will be joined after a vehicle type is decided so the data can be used in downstream models.
* ``COLS_TO_INCLUDE_IN_VEHICLE_TABLE``: List of columns from the vehicle type data file to include in the vehicle table that can be used in downstream models.
  Examples of data that might be needed is vehicle range for the :ref:`vehicle_allocation` model, auto operating costs to use in tour and trip mode choice,
  and emissions data for post-model-run analysis.
* ``FLEET_YEAR``: Integer specifying the fleet year to be used in the model run. This is used to compute ``age`` in the
  vehicle type data table where ``age = (1 + FLEET_YEAR - vehicle_year)``. Computing age on the fly with the ``FLEET_YEAR`` variable allows the
  user flexibility to compile and share a single vehicle type data file containing all years and simply change the ``FLEET_YEAR`` to run
  different scenario years.
* Optional additional settings that work the same in other models are constants, expression preprocessor, and annotate tables.

Input vehicle type data included in :ref:`prototype_mtc_extended` came from a variety of sources. The number of vehicle makes, models, MPG, and
electric vehicle range was sourced from the Enivornmental Protection Agency (EPA).  Additional data on vehicle costs were derived from the
National Household Travel Survey. Auto operating costs in the vehicle type data file were a sum of fuel costs and maintenance costs.
Fuel costs were calculated from MPG assuming a $3.00 cost for a gallon of gas. When MPG was not available to calculate fuel costs,
the closest year, vehicle type, or body type available was used. Maintenance costs were taken from AAA's
`2017 driving cost study <https://exchange.aaa.com/wp-content/uploads/2017/08/17-0013_Your-Driving-Costs-Brochure-2017-FNL-CX-1.pdf>`_.
Size categories within body types were averaged, e.g. car was an average of AAA's small, medium, and large sedan categories.
Motorcycles were assigned the small sedan maintenance costs since they were not included in AAA's report.
Maintenance costs were not varied by vehicle year. (According to
`data from the U.S. Bureau of Labor Statistics <https://www.bls.gov/opub/btn/volume-3/pdf/americans-aging-autos.pdf>`_,
there was no consistent relationship between vehicle age and maintenance costs.)

Using the above methodology, the average auto operating costs of vehicles output from :ref:`prototype_mtc_extended` was 18.4 cents.
This value is very close to the auto operating cost of 18.3 cents used in :ref:`prototype_mtc`.
Non-household vehicles in prototype_mtc_extended use the auto operating cost of 18.3 cents used in prototype_mtc.
Users are encouraged to make their own assumptions and calculate auto operating costs as they see fit.

The distribution of fuel type probabilities included in :ref:`prototype_mtc_extended` are computed directly from the National Household Travel Survey data
and include the entire US. Therefore, there is "lumpiness" in probabilities due to poor statistics in the data for some vehicle types.
The user is encouraged to adjust the probabilities to their modeling region and "smooth" them for more consistent results.

Further discussion of output results and model sensitivities can be found `here <https://github.com/ActivitySim/activitysim/wiki/Project-Meeting-2022.05.05>`_.

.. automodule:: activitysim.abm.models.vehicle_type_choice
   :members:

.. _telecommute_frequency:

Telecommute Frequency
---------------------

Telecommuting is defined as workers who work from home instead of going to work. It only applies to
workers with a regular workplace outside of home. The telecommute model consists of two
submodels - a person :ref:`work_from_home` model and this person telecommute frequency model.

For all workers that work out of the home, the telecommute models predicts the
level of telecommuting. The model alternatives are the frequency of telecommuting in
days per week (0 days, 1 day, 2 to 3 days, 4+ days).

The main interface to the work from home model is the
:py:func:`~activitysim.abm.models.telecommute_frequency` function.  This
function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Field: ``telecommute_frequency`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.telecommute_frequency
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
as an Inject step in the example Pipeline.

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
* take as input an interaction coefficients table and then programmatically produce and write out the expression files for households size 1, 2, 3, 4, and 5 models independent of one another
* select households of size 1, join all required person attributes, and then read and solve the automatically generated expressions
* repeat for households size 2, 3, 4, and 5. Each model is independent of one another.

The main interface to the CDAP model is the :py:func:`~activitysim.abm.models.util.cdap.run_cdap`
function.  This function is called by the Inject step ``cdap_simulate`` which is
registered as an Inject step in the example Pipeline.  There are two cdap class definitions in
ActivitySim.  The first is at :py:func:`~activitysim.abm.models.cdap` and contains the Inject
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
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``mandatory_tour_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.mandatory_tour_frequency
   :members:

.. _mandatory_tour_scheduling:
.. _representative_logsums:

Mandatory Tour Scheduling
-------------------------

The mandatory tour scheduling model selects a tour departure and duration period (and therefore a
start and end period as well) for each mandatory tour.   The primary drivers in the model are
accessibility-based parameters such as the mode choice logsum for the departure/arrival hour
combination, demographics, and time pattern characteristics such as the time windows available
from previously scheduled tours. This model uses person :ref:`time_windows`.


.. note::
   For ``prototype_mtc``, the modeled time periods for all submodels are hourly from 3 am to 3 am the next day, and any times before 5 am are shifted to time period 5, and any times after 11 pm are shifted to time period 23.


If ``tour_departure_and_duration_segments.csv`` is included in the configs, then the model
will use these representative start and end time periods when calculating mode choice logsums
instead of the specific start and end combinations for each alternative to reduce runtime.  This
feature, know as ``representative logsums``, takes advantage of the fact that the mode choice logsum,
say, from 6 am to 2 pm is very similar to the logsum from 6 am to 3 pm, and 6 am to 4 pm, and so using
just 6 am to 3 pm (with the idea that 3 pm is the "representative time period") for these alternatives is
sufficient for tour scheduling.  By reusing the 6 am to 3 pm mode choice logsum, ActivitySim saves
significant runtime.

The main interface to the mandatory tour purpose scheduling model is the
:py:func:`~activitysim.abm.models.mandatory_scheduling.mandatory_tour_scheduling`
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``TAZ, workplace_taz, school_taz, start, end``

.. automodule:: activitysim.abm.models.mandatory_scheduling
   :members:


.. _school_escorting:

School Escorting
----------------

The school escort model determines whether children are dropped-off at or picked-up from school,
simultaneously with the chaperone responsible for chauffeuring the children,
which children are bundled together on half-tours, and the type of tour (pure escort versus rideshare).
The model is run after work and school locations have been chosen for all household members,
and after work and school tours have been generated and scheduled.
The model labels household members of driving age as potential ‘chauffeurs’ and children with school tours as potential ‘escortees’.
The model then attempts to match potential chauffeurs with potential escortees in a choice model whose alternatives
consist of ‘bundles’ of escortees with a chauffeur for each half tour.

School escorting is a household level decision – each household will choose an alternative from the ``school_escorting_alts.csv`` file,
with the first alternative being no escorting. This file contains the following columns:

+------------------------------------------------+--------------------------------------------------------------------+
|  Column Name                                   |    Column Description                                              |
+================================================+====================================================================+
|  Alt                                           |  Alternative number                                                |
+------------------------------------------------+--------------------------------------------------------------------+
|  bundle[1,2,3]                                 |  bundle number for child 1,2, and 3                                |
+------------------------------------------------+--------------------------------------------------------------------+
|  chauf[1,2,3]                                  |  chauffeur number for child 1,2, and 3                             |
|                                                |  - 0 = child not escorted                                          |
|                                                |  - 1 = chauffeur 1 as ride share                                   |
|                                                |  - 2 = chauffeur 1 as pure escort                                  |
|                                                |  - 3 = chauffeur 2 as ride share                                   |
|                                                |  - 4 = chauffeur 3 as pure escort                                  |
+------------------------------------------------+--------------------------------------------------------------------+
|  nbund[1,2]                                    |  - number of escorting bundles for chauffeur 1 and 2               |
+------------------------------------------------+--------------------------------------------------------------------+
|  nbundles                                      |  - total number of bundles                                         |
|                                                |  - equals nbund1 + nbund2                                          |
+------------------------------------------------+--------------------------------------------------------------------+
|  nrs1                                          |  - number of ride share bundles for chauffeur 1                    |
+------------------------------------------------+--------------------------------------------------------------------+
|  npe1                                          |  - number of pure escort bundles for chauffeur 1                   |
+------------------------------------------------+--------------------------------------------------------------------+
|  nrs2                                          |  - number of ride share bundles for chauffeur 2                    |
+------------------------------------------------+--------------------------------------------------------------------+
|  npe2                                          |  - number of pure escort bundles for chauffeur 2                   |
+------------------------------------------------+--------------------------------------------------------------------+
|  Description                                   |  - text description of alternative                                 |
+------------------------------------------------+--------------------------------------------------------------------+

The model as currently implemented contains three escortees and two chauffeurs.
Escortees are students under age 16 with a mandatory tour whereas chaperones are all persons in the household over the age of 18.
For households that have more than three possible escortees, the three youngest children are selected for the model.
The two chaperones are selected as the adults of the household with the highest weight according to the following calculation:
:math:`Weight = 100*personType + 10*gender + 1*age(0,1)`
Where *personType* is the person type number from 1 to 5, *gender* is 1 for male and 2 for female, and
*age* is a binary indicator equal to 1 if age is over 25 else 0.

The model is run sequentially three times, once in the outbound direction, once in the inbound direction,
and again in the outbound direction with additional conditions on what happened in the inbound direction.
There are therefore three sets of utility specifications, coefficients, and pre-processor files.
Each of these files is specified in the school_escorting.yaml file along with the number of escortees and number of chaperones.

There is also a constants section in the school_escorting.yaml file which contain two constants.
One which sets the maximum time bin difference to match school and work tours for ride sharing
and another to set the number of minutes per time bin.
In the :ref:`prototype_mtc_extended` example, these are set to 1 and 60 respectively.

After a school escorting alternative is chosen for the inbound and outbound direction, the model will
create the tours and trips associated with the decision.  Pure escort tours are created,
and the mandatory tour start and end times are changed to match the school escort bundle start and end times.
(Outbound tours have their start times matched and inbound tours have their end times matched.)
Escortee drop-off / pick-up order is determined by the distance from home to the school locations.
They are ordered from smallest to largest in the outbound direction, and largest to smallest in the inbound direction.
Trips are created for each half-tour that includes school escorting according to the provided order.

The created pure escort tours are joined to the already created mandatory tour table in the pipeline
and are also saved separately to the pipeline under the table name “school_escort_tours”.
Created school escorting trips are saved to the pipeline under the table name “school_escort_trips”.
By saving these to the pipeline, their data can be queried in downstream models to set correct purposes,
destinations, and schedules to satisfy the school escorting model choice.

There are a host of downstream model changes that are involved when including the school escorting model.
The following list contains the models that are changed in some way when school escorting is included:

 * **Joint tour scheduling:** Joint tours are not allowed to be scheduled over school escort tours.
   This happens automatically by updating the timetable object with the updated mandatory tour times
   and created pure escort tour times after the school escorting model is run.
   There were no code or config changes in this model, but it is still affected by school escorting.
 * **Non-Mandatory tour frequency:**  Pure school escort tours are joined to the tours created in the
   non-mandatory tour frequency model and tour statistics (such as tour_count and tour_num) are re-calculated.
 * **Non-Mandatory tour destination:** Since the primary destination of pure school escort tours is known,
   they are removed from the choosers table and have their destination set according to the destination in\
   school_escort_tours table.  They are also excluded from the estimation data bundle.
 * **Non-Mandatory tour scheduling:** Pure escort tours need to have the non-escorting portion of their tour scheduled.
   This is done by inserting availability conditions in the model specification that ensures the alternative
   chosen for the start of the tour is equal to the alternative start time for outbound tours and the end time
   is equal to the alternative end time for the inbound tours.  There are additional terms that ensure the tour
   does not overlap with subsequent school escorting tours as well.  Beware -- If the availability conditions
   in the school escorting model are not set correctly, the tours created may not be consistent with each other
   and this model will fail.
 * **Tour mode choice:** Availability conditions are set in tour mode choice to prohibit the drive alone mode
   if the tour contains an escortee and the shared-ride 2 mode if the tour contains more than one escortee.
 * **Stop Frequency:** No stops are allowed on half-tours that include school escorting.
   This is enforced by adding availability conditions in the stop frequency model.  After the stop frequency
   model is run, the school escorting trips are merged from the trips created by the stop frequency model
   and a new stop frequency is computed along with updated trip numbers.
 * **Trip purpose, destination, and scheduling:** Trip purpose, destination, and departure times are known
   for school escorting trips.  As such they are removed from their respective chooser tables and the estimation
   data bundles, and set according to the values in the school_escort_trips table residing in the pipeline.
 * **Trip mode choice:** Like in tour mode choice, availability conditions are set to prohibit trip containing
   an escortee to use the drive alone mode or the shared-ride 2 mode for trips with more than one escortee.

Many of the changes discussed in the above list are handled in the code and the user is not required to make any
changes when implementing the school escorting model.  However, it is the users responsibility to include the
changes in the following model configuration files for models downstream of the school escorting model:

+--------------------------------------------------------------------+------------------------------------------------------------------+
| File Name(s)                                                       | Change(s) Needed                                                 |
+====================================================================+==================================================================+
|  - `non_mandatory_tour_scheduling_annotate_tours_preprocessor.csv` |                                                                  |
|  - `tour_scheduling_nonmandatory.csv`                              | - Set availability conditions based on those times               |
|                                                                    | - Do not schedule over other school escort tours                 |
+--------------------------------------------------------------------+------------------------------------------------------------------+
|  - `tour_mode_choice_annotate_choosers_preprocessor.csv`           |  - count number of escortees on tour by parsing the              |
|  - `tour_mode_choice.csv`                                          |  ``escort_participants`` column                                  |
|                                                                    |  - set mode choice availability based on number of escortees     |
|                                                                    |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+
| - `stop_frequency_school.csv`                                      |  Do not allow stops for half-tours that include school escorting |
| - `stop_frequency_work.csv`                                        |                                                                  |
| - `stop_frequency_univ.csv`                                        |                                                                  |
| - `stop_frequency_escort.csv`                                      |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+
|  - `trip_mode_choice_annotate_trips_preprocessor.csv`              |  - count number of escortees on trip by parsing the              |
|  - `trip_mode_choice.csv`                                          |  ``escort_participants`` column                                  |
|                                                                    |  - set mode choice availability based on number of escortees     |
|                                                                    |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+

When not including the school escorting model, all of the escort trips to and from school are counted implicitly in
escort tours determined in the non-mandatory tour frequency model. Thus, when including the school escort model and
accounting for these tours explicitly, extra care should be taken not to double count them in the non-mandatory
tour frequency model. The non-mandatory tour frequency model should be re-evaluated and likely changed to decrease
the number of escort tours generated by that model.  This was not implemented in the :ref:`prototype_mtc_extended`
implementation due to a lack of data surrounding the number of escort tours in the region.


.. automodule:: activitysim.abm.models.school_escorting
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
function.  This function is registered as an Inject step in the example Pipeline.

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
function.  This function is registered as an Inject step in the example Pipeline.

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
function.  This function is registered as an Inject step in the example Pipeline.

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

Joint tour location choice for :ref:`multiple_zone_systems` models uses :ref:`presampling` by default.

The main interface to the model is the :py:func:`~activitysim.abm.models.joint_tour_destination.joint_tour_destination`
function.  This function is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums` for how
to write logsums for estimation.

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
function.  This function is registered as an Inject step in the example Pipeline.

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
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``persons`` | Result Fields: ``non_mandatory_tour_frequency`` | Skims Keys: NA


.. automodule:: activitysim.abm.models.non_mandatory_tour_frequency
   :members:

.. _non_mandatory_tour_destination_choice:

Non-Mandatory Tour Destination Choice
-------------------------------------

The non-mandatory tour destination choice model chooses a destination zone for
non-mandatory tours.  The three step (sample, logsums, final choice) process also used for
mandatory tour destination choice is used for non-mandatory tour destination choice.

Non-mandatory tour location choice for :ref:`multiple_zone_systems` models uses :ref:`presampling` by default.

The main interface to the non-mandatory tour destination choice model is the
:py:func:`~activitysim.abm.models.non_mandatory_destination.non_mandatory_tour_destination`
function.  This function is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums`
for how to write logsums for estimation.

Core Table: ``tours`` | Result Field: ``destination`` | Skims Keys: ``TAZ, alt_dest, MD time period, MD time period``


.. automodule:: activitysim.abm.models.non_mandatory_destination
   :members:


.. _non_mandatory_tour_scheduling:

Non-Mandatory Tour Scheduling
-----------------------------

The non-mandatory tour scheduling model selects a tour departure and duration period (and therefore a start and end
period as well) for each non-mandatory tour.  This model uses person :ref:`time_windows`.  Includes support
for :ref:`representative_logsums`.

The main interface to the non-mandatory tour purpose scheduling model is the
:py:func:`~activitysim.abm.models.non_mandatory_scheduling.non_mandatory_tour_scheduling`
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``TAZ, destination, MD time period, MD time period``


.. automodule:: activitysim.abm.models.non_mandatory_scheduling
   :members:

.. _vehicle_allocation:

Vehicle Allocation
-------------------

The vehicle allocation model selects which vehicle would be used for a tour of given occupancy. The alternatives for the vehicle
allocation model consist of the vehicles owned by the household and an additional non household vehicle option. (Zero-auto
households would be assigned the non-household vehicle option since there are no owned vehicles in the household).
A vehicle is selected for each occupancy level set by the user such that different tour modes that have different occupancies could see different operating
characteristics. The output of the vehicle allocation model is appended to the tour table with column names ``vehicle_occup_{occupancy}`` and the values are
the vehicle type selected.

In :ref:`prototype_mtc_extended`, three occupancy levels are used: 1, 2, and 3.5.  The auto operating cost
for occupancy level 1 is used in the drive alone mode and drive to transit modes. Occupancy levels 2 and 3.5 are used for shared
ride 2 and shared ride 3+ auto operating costs, respectively.  Auto operating costs are selected in the mode choice pre-processors by selecting the allocated
vehicle type data from the vehicles table. If the allocated vehicle type was the non-household vehicle, the auto operating costs uses
the previous default value from :ref:`prototype_mtc`. All trips and atwork subtours use the auto operating cost of the parent tour.  Functionality
was added in tour and atwork subtour mode choice to annotate the tour table and create a ``selected_vehicle`` which denotes the actual vehicle used.
If the tour mode does not include a vehicle, then the ``selected_vehicle`` entry is left blank.

The current implementation does not account for possible use of the household vehicles by other household members.  Thus, it is possible for a
selected vehicle to be used in two separate tours at the same time.


.. automodule:: activitysim.abm.models.vehicle_allocation
   :members:

.. _tour_mode_choice:

Tour Mode Choice
----------------

The mandatory, non-mandatory, and joint tour mode choice model assigns to each tour the "primary" mode that
is used to get from the origin to the primary destination. The tour-based modeling approach requires a reconsideration
of the conventional mode choice structure. Instead of a single mode choice model used in a four-step
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

The main interface to the mandatory, non-mandatory, and joint tour mode model is the
:py:func:`~activitysim.abm.models.tour_mode_choice.tour_mode_choice_simulate` function.  This function is
called in the Inject step ``tour_mode_choice_simulate`` and is registered as an Inject step in the example Pipeline.
See :ref:`writing_logsums` for how to write logsums for estimation.

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
function.  This function is registered as an Inject step in the example Pipeline.

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

At-work subtour location choice for :ref:`multiple_zone_systems` models uses :ref:`presampling` by default.

Core Table: ``tours`` | Result Table: ``destination`` | Skims Keys: ``workplace_taz, alt_dest, MD time period``

The main interface to the at-work subtour destination model is the
:py:func:`~activitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination`
function.  This function is registered as an Inject step in the example Pipeline.
See :ref:`writing_logsums` for how to write logsums for estimation.

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
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``tours`` | Result Field: ``start, end, duration`` | Skims Keys: ``workplace_taz, alt_dest, MD time period, MD time period``

.. automodule:: activitysim.abm.models.atwork_subtour_scheduling
   :members:


.. _atwork_subtour_mode_choice:

At-work Subtour Mode
--------------------

The at-work subtour mode choice model assigns a travel mode to each at-work subtour using the :ref:`tour_mode_choice` model.

The main interface to the at-work subtour mode choice model is the
:py:func:`~activitysim.abm.models.atwork_subtour_mode_choice.atwork_subtour_mode_choice`
function.  This function is called in the Inject step ``atwork_subtour_mode_choice`` and
is registered as an Inject step in the example Pipeline.
See :ref:`writing_logsums` for how to write logsums for estimation.

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
function.  This function is registered as an Inject step in the example Pipeline.

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
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``purpose`` | Skims Keys: NA

.. note::
   Trip purpose and trip destination choice can be run iteratively together via :ref:`trip_purpose_and_destination_model`.


.. automodule:: activitysim.abm.models.trip_purpose
   :members:


.. _trip_destination_choice:

Trip Destination Choice
-----------------------

See :ref:`Trip Destination <component-trip-destination>`.


.. _trip_purpose_and_destination_model:

Trip Purpose and Destination
----------------------------

After running trip purpose and trip destination separately, the two model can be ran together in an iterative fashion on
the remaining failed trips (i.e. trips that cannot be assigned a destination).  Each iteration uses new random numbers.

The main interface to the trip purpose model is the
:py:func:`~activitysim.abm.models.trip_purpose_and_destination.trip_purpose_and_destination`
function.  This function is registered as an Inject step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``purpose, destination`` | Skims Keys: ``origin, (tour primary) destination, dest_taz, trip_period``

.. automodule:: activitysim.abm.models.trip_purpose_and_destination
   :members:


.. _trip_scheduling:

Trip Scheduling (Probablistic)
------------------------------

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
This function is registered as an Inject step in the example Pipeline.

Core Table: ``trips`` | Result Field: ``depart`` | Skims Keys: NA

.. automodule:: activitysim.abm.models.trip_scheduling
   :members:


.. _trip_scheduling_choice:

Trip Scheduling Choice (Logit Choice)
-------------------------------------

This model uses a logit-based formulation to determine potential trip windows for the three
main components of a tour.

-  Outbound Leg: The time from leaving the origin location to the time second to last outbound stop.
-  Main Leg: The time window from the last outbound stop through the main tour destination to the first inbound stop.
-  Inbound Leg: The time window from the first inbound stop to the tour origin location.

Core Table: ``tours`` | Result Field: ``outbound_duration``, ``main_leg_duration``, ``inbound_duration`` | Skims Keys: NA


**Required YAML attributes:**

- ``SPECIFICATION``
    This file defines the logit specification for each chooser segment.
- ``COEFFICIENTS``
    Specification coefficients
- ``PREPROCESSOR``:
    Preprocessor definitions to run on the chooser dataframe (trips) before the model is run

.. _trip_departure_choice:

Trip Departure Choice (Logit Choice)
-------------------------------------

Used in conjuction with Trip Scheduling Choice (Logit Choice), this model chooses departure
time periods consistent with the time windows for the appropriate leg of the trip.

Core Table: ``trips`` | Result Field: ``depart`` | Skims Keys: NA

**Required YAML attributes:**

- ``SPECIFICATION``
    This file defines the logit specification for each chooser segment.
- ``COEFFICIENTS``
    Specification coefficients
- ``PREPROCESSOR``:
    Preprocessor definitions to run on the chooser dataframe (trips) before the model is run

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

The trip mode choice models explanatory variables include household and person variables, level-of-service
between the trip origin and destination according to the time period for the tour leg, urban form
variables, and alternative-specific constants segmented by tour mode.

The main interface to the trip mode choice model is the
:py:func:`~activitysim.abm.models.trip_mode_choice.trip_mode_choice` function.  This function
is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums` for how to write logsums for estimation.

Core Table: ``trips`` | Result Field: ``trip_mode`` | Skims Keys: ``origin, destination, trip_period``

.. automodule:: activitysim.abm.models.trip_mode_choice
   :members:

.. _parking_location_choice:

Parking Location Choice
-----------------------

The parking location choice model selects a parking location for specified trips. While the model does not
require parking location be applied to any specific set of trips, it is usually applied for drive trips to
specific zones (e.g., CBD) in the model.

The model provides provides a filter for both the eligible choosers and eligible parking location zone. The
trips dataframe is the chooser of this model. The zone selection filter is applied to the land use zones
dataframe.

If this model is specified in the pipeline, the `Write Trip Matrices`_ step will using the parking location
choice results to build trip tables in lieu of the trip destination.

The main interface to the trip mode choice model is the
:py:func:`~activitysim.abm.models.parking_location_choice.parking_location_choice` function.  This function
is registered as an Inject step, and it is available from the pipeline.  See :ref:`writing_logsums` for how to write
logsums for estimation.

**Skims**

- ``odt_skims``: Origin to Destination by Time of Day
- ``dot_skims``: Destination to Origin by Time of Day
- ``opt_skims``: Origin to Parking Zone by Time of Day
- ``pdt_skims``: Parking Zone to Destination by Time of Day
- ``od_skims``: Origin to Destination
- ``do_skims``: Destination to Origin
- ``op_skims``: Origin to Parking Zone
- ``pd_skims``: Parking Zone to Destination

Core Table: ``trips``

**Required YAML attributes:**

- ``SPECIFICATION``
    This file defines the logit specification for each chooser segment.
- ``COEFFICIENTS``
    Specification coefficients
- ``PREPROCESSOR``:
    Preprocessor definitions to run on the chooser dataframe (trips) before the model is run
- ``CHOOSER_FILTER_COLUMN_NAME``
    Boolean field on the chooser table defining which choosers are eligible to parking location choice model. If no
    filter is specified, all choosers (trips) are eligible for the model.
- ``CHOOSER_SEGMENT_COLUMN_NAME``
    Column on the chooser table defining the parking segment for the logit model
- ``SEGMENTS``
    List of eligible chooser segments in the logit specification
- ``ALTERNATIVE_FILTER_COLUMN_NAME``
    Boolean field used to filter land use zones as eligible parking location choices. If no filter is specified,
    then all land use zones are considered as viable choices.
- ``ALT_DEST_COL_NAME``
    The column name to append with the parking location choice results. For choosers (trips) ineligible for this
    model, a -1 value will be placed in column.
- ``TRIP_ORIGIN``
    Origin field on the chooser trip table
- ``TRIP_DESTINATION``
    Destination field on the chooser trip table

.. automodule:: activitysim.abm.models.parking_location_choice
   :members:

.. _write_trip_matrices:

Write Trip Matrices
-------------------

Write open matrix (OMX) trip matrices for assignment.  Reads the trips table post preprocessor and run expressions
to code additional data fields, with one data fields for each matrix specified.  The matrices are scaled by a
household level expansion factor, which is the household sample rate by default, which is calculated when
households are read in at the beginning of a model run.  The main interface to write trip
matrices is the :py:func:`~activitysim.abm.models.trip_matrices.write_trip_matrices` function.  This function
is registered as an Inject step in the example Pipeline.

If the `Parking Location Choice`_ model is defined in the pipeline, the parking location zone will be used in
lieu of the destination zone.

Core Table: ``trips`` | Result: ``omx trip matrices`` | Skims Keys: ``origin, destination``

.. automodule:: activitysim.abm.models.trip_matrices
   :members:

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

Estimation
~~~~~~~~~~

See :ref:`estimation` for more information.

.. automodule:: activitysim.abm.models.util.estimation
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
