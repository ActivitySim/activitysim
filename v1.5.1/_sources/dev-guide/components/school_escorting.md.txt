(component-school-escorting)=
# School Escorting

```{eval-rst}
.. currentmodule:: activitysim.abm.models.school_escorting
```
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

```{eval-rst}
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
```

The model as currently implemented contains three escortees and two chauffeurs.
Escortees are students under age 16 with a mandatory tour whereas chaperones are all persons in the household over the age of 18.
For households that have more than three possible escortees, the three youngest children are selected for the model.
The two chaperones are selected as the adults of the household with the highest weight according to the following calculation:
`Weight = 100*personType + 10*gender + 1*age(0,1)`
Where `personType` is the person type number from 1 to 5, `gender` is 1 for male and 2 for female, and
`age` is a binary indicator equal to 1 if age is over 25 else 0.

The model is run sequentially three times, once in the outbound direction, once in the inbound direction,
and again in the outbound direction with additional conditions on what happened in the inbound direction.
There are therefore three sets of utility specifications, coefficients, and pre-processor files.
Each of these files is specified in the school_escorting.yaml file along with the number of escortees and number of chaperones.

There is also a constants section in the school_escorting.yaml file which contain two constants.
One which sets the maximum time bin difference to match school and work tours for ride sharing
and another to set the number of minutes per time bin.
In the [prototype_mtc_extended](prototype_mtc_extended) example, these are set to 1 and 60 respectively.

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

```{eval-rst}
+--------------------------------------------------------------------+------------------------------------------------------------------+
| File Name(s)                                                       | Change(s) Needed                                                 |
+====================================================================+==================================================================+
|  - `non_mandatory_tour_scheduling_annotate_tours_preprocessor.csv` |                                                                  |
|  - `tour_scheduling_nonmandatory.csv`                              | - Set availability conditions based on those times               |
|                                                                    | - Do not schedule over other school escort tours                 |
+--------------------------------------------------------------------+------------------------------------------------------------------+
|  - `tour_mode_choice_annotate_choosers_preprocessor.csv`           |  - count number of escortees on tour by parsing the              |
|  - `tour_mode_choice.csv`                                          |    ``escort_participants`` column                                |
|                                                                    |  - set mode choice availability based on number of escortees     |
|                                                                    |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+
| - `stop_frequency_school.csv`                                      |  Do not allow stops for half-tours that include school escorting |
| - `stop_frequency_work.csv`                                        |                                                                  |
| - `stop_frequency_univ.csv`                                        |                                                                  |
| - `stop_frequency_escort.csv`                                      |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+
|  - `trip_mode_choice_annotate_trips_preprocessor.csv`              |  - count number of escortees on trip by parsing the              |
|  - `trip_mode_choice.csv`                                          |    ``escort_participants`` column                                |
|                                                                    |  - set mode choice availability based on number of escortees     |
|                                                                    |                                                                  |
+--------------------------------------------------------------------+------------------------------------------------------------------+
```

- *Joint tour scheduling:* Joint tours are not allowed to be scheduled over school escort tours.
   This happens automatically by updating the timetable object with the updated mandatory tour times
   and created pure escort tour times after the school escorting model is run.
   There were no code or config changes in this model, but it is still affected by school escorting.
- *Non-Mandatory tour frequency:*  Pure school escort tours are joined to the tours created in the
   non-mandatory tour frequency model and tour statistics (such as tour_count and tour_num) are re-calculated.
- *Non-Mandatory tour destination:* Since the primary destination of pure school escort tours is known,
   they are removed from the choosers table and have their destination set according to the destination in\
   school_escort_tours table.  They are also excluded from the estimation data bundle.
- *Non-Mandatory tour scheduling:* Pure escort tours need to have the non-escorting portion of their tour scheduled.
   This is done by inserting availability conditions in the model specification that ensures the alternative
   chosen for the start of the tour is equal to the alternative start time for outbound tours and the end time
   is equal to the alternative end time for the inbound tours.  There are additional terms that ensure the tour
   does not overlap with subsequent school escorting tours as well.  Beware -- If the availability conditions
   in the school escorting model are not set correctly, the tours created may not be consistent with each other
   and this model will fail.
- *Tour mode choice:* Availability conditions are set in tour mode choice to prohibit the drive alone mode
   if the tour contains an escortee and the shared-ride 2 mode if the tour contains more than one escortee.
- *Stop Frequency:* No stops are allowed on half-tours that include school escorting.
   This is enforced by adding availability conditions in the stop frequency model.  After the stop frequency
   model is run, the school escorting trips are merged from the trips created by the stop frequency model
   and a new stop frequency is computed along with updated trip numbers.
- *Trip purpose, destination, and scheduling:* Trip purpose, destination, and departure times are known
   for school escorting trips.  As such they are removed from their respective chooser tables and the estimation
   data bundles, and set according to the values in the school_escort_trips table residing in the pipeline.
- *Trip mode choice:* Like in tour mode choice, availability conditions are set to prohibit trip containing
   an escortee to use the drive alone mode or the shared-ride 2 mode for trips with more than one escortee.

Many of the changes discussed in the above list are handled in the code and the user is not required to make any
changes when implementing the school escorting model.  However, it is the users responsibility to include the
changes in the following model configuration files for models downstream of the school escorting model:


When not including the school escorting model, all of the escort trips to and from school are counted implicitly in
escort tours determined in the non-mandatory tour frequency model. Thus, when including the school escort model and
accounting for these tours explicitly, extra care should be taken not to double count them in the non-mandatory
tour frequency model. The non-mandatory tour frequency model should be re-evaluated and likely changed to decrease
the number of escort tours generated by that model.  This was not implemented in the [prototype_mtc_extended](prototype_mtc_extended)
implementation due to a lack of data surrounding the number of escort tours in the region.

## Configuration

```{eval-rst}
.. autopydantic_model:: SchoolEscortSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC Extended](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc_extended/configs/school_escorting.yaml)

## Implementation

```{eval-rst}
.. autofunction:: school_escorting
```
