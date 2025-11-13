(component-trip-scheduling-probabilistic)=
# Trip Scheduling (Probabilistic)

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_scheduling
```

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

In order to avoid trip failing, a new probabilistic trip scheduling mode was developed named "relative".
When setting the _scheduling_mode_ option to relative, trips are scheduled relative to the previously scheduled trips.
The first trip still departs when the tour starts and for every subsequent trip, the choices are selected with respect to
the previous trip depart time. Inbound trips are no longer handled in reverse order.  The key to this relative mode is to
index the probabilities based on how much time is remaining on the tour.  For tours that include subtours, the time remaining will
be based on the subtour start time for outbound trips and will resume again for inbound trips after the subtour ends.
By indexing the probabilities based on time remaining and scheduling relative to the previous trip, scheduling trips in relative
mode will not fail.  Note also that relative scheduling mode requires the use of logic
version 2 (see warning about logic versions, below).

An example of trip scheduling in relative mode is included in the [prototype_mwcog](prototype_mwcog) example.  In this example, trip
scheduling probabilities are indexed by the following columns:

  * periods_left_min: the minimum bin for the number of time periods left on the tour.
  * periods_left_max: the maximum bin for the number of time periods left on the tour.  This is the same as periods_left_min until the final time period bin.
  * outbound: whether the trip occurs on the outbound leg of a tour.
  * tour_purpose_grouped: Tour purpose grouped into mandatory and non-mandatory categories
  * half_tour_stops_remaining_grouped: The number of stops remaining on the half tour with the categories of 0 and 1+

Each of these variables are listed as merge columns in the trip_scheduling.yaml file and are declared in the trip scheduling preprocessor.
The variables above attempt to balance the statistics available for probability creation with the amount of segmentation of trip characteristics.

.. warning::

    Earlier versions of ActivitySim contained a logic error in this model, whereby
    the earliest departure time for inbound legs was bounded by the maximum outbound
    departure time, even if there was a scheduling failure for one or more outbound
    leg departures and that bound was NA.  For continuity, this process has been
    retained in this ActivitySim component as *logic_version* 1, and it remains the
    default process if the user does not explicitly specify a logic version in the
    model settings yaml file. The revised logic includes bounding inbound legs only
    when the maximum outbound departure time is well defined.  This version of the
    model can be used by explicitly setting `logic_version: 2` (or greater) in the
    model settings yaml file.  It is strongly recommended that all new model
    development efforts use logic version 2; a future version of ActivitySim may
    make this the default for this component, and/or remove logic version 1 entirely.

The main interface to the trip scheduling model is the
[trip_scheduling](activitysim.abm.models.trip_scheduling.trip_scheduling) function.
This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `trip_scheduling.yaml`
- *Core Table*: `trips`
- *Result Field*: `depart`

## Configuration

```{eval-rst}
.. autopydantic_model:: TripSchedulingSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/camsys/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/trip_scheduling.yaml)
- [Prototype SEMCOG](https://github.com/camsys/activitysim/blob/main/activitysim/examples/prototype_semcog/configs/trip_scheduling.yaml)

## Implementation

```{eval-rst}
.. autofunction:: trip_scheduling
.. autofunction:: set_stop_num
.. autofunction:: update_tour_earliest
.. autofunction:: schedule_trips_in_leg
```
