(component-trip-departure-choice)=
# Trip Departure Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_departure_choice
```

Used in conjuction with Trip Scheduling Choice (Logit Choice), this model chooses departure
time periods consistent with the time windows for the appropriate leg of the trip.

## Structure

- *Configuration File*: `trip_departure_choice.yaml`
- *Core Table*: `trips`
- *Result Field*: `depart`

## Configuration

```{eval-rst}
.. autopydantic_model:: TripDepartureChoiceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_departure_choice.yaml)

## Implementation

```{eval-rst}
.. autofunction:: trip_departure_choice
```
