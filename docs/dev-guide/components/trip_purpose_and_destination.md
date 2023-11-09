(component-trip-purpose-and-destination)=
# Trip Purpose and Destination

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_purpose_and_destination
```

After running trip purpose and trip destination separately, the two model can be ran together in an iterative fashion on
the remaining failed trips (i.e. trips that cannot be assigned a destination).  Each iteration uses new random numbers.

The main interface to the trip purpose model is the
[trip_purpose_and_destination](activitysim.abm.models.trip_purpose_and_destination.trip_purpose_and_destination)
function.  This function is registered as an Inject step in the example Pipeline.


## Structure

- *Core Table*: `trips`
- *Result Field*: `purpose, destination`
- *Skims Keys*: `origin, (tour primary) destination, dest_taz, trip_period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TripPurposeAndDestinationSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/trip_purpose_and_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_purpose_and_destination.yaml)


## Implementation

```{eval-rst}
.. autofunction:: trip_purpose_and_destination
```
