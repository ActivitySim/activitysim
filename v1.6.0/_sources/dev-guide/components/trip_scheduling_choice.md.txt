(component-trip-scheduling-choice)=
# Trip Scheduling Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_scheduling_choice
```

This model uses a logit-based formulation to determine potential trip windows for the three
main components of a tour.

-  Outbound Leg: The time from leaving the origin location to the time second to last outbound stop.
-  Main Leg: The time window from the last outbound stop through the main tour destination to the first inbound stop.
-  Inbound Leg: The time window from the first inbound stop to the tour origin location.

## Structure

- *Configuration File*: `trip_scheduling_choice.yaml`
- *Core Table*: `tours`
- *Result Field*: `outbound_duration`, `main_leg_duration`, `inbound_duration`

## Configuration

```{eval-rst}
.. autopydantic_model:: TripSchedulingChoiceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_scheduling_choice.yaml)

## Implementation

```{eval-rst}
.. autofunction:: trip_scheduling_choice
.. autofunction:: generate_schedule_alternatives
.. autofunction:: no_stops_patterns
.. autofunction:: stop_one_way_only_patterns
.. autofunction:: stop_two_way_only_patterns
.. autofunction:: get_pattern_index_and_arrays
.. autofunction:: get_spec_for_segment
```
