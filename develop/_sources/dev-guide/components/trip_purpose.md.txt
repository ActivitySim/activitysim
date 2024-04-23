(component-trip-purpose)=
# Trip Purpose

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_purpose
```

For trip other than the last trip outbound or inbound, assign a purpose based on an
observed frequency distribution.  The distribution is segmented by tour purpose, tour
direction and person type. Work tours are also segmented by departure or arrival time period.

The main interface to the trip purpose model is the
[trip_purpose](activitysim.abm.models.trip_purpose.trip_purpose)
function.  This function is registered as an Inject step in the example Pipeline.


## Structure


- *Core Table*: `trips`
- *Result Field*: `purpose`

## Configuration

```{eval-rst}
.. autopydantic_model::
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Note
Trip purpose and trip destination choice can be run iteratively together [trip_purpose_and_destination_model](activitysim.abm.models.trip_purpose_and_destination.py)

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/trip_purpose.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_purpose.yaml)


## Implementation

```{eval-rst}
.. autofunction:: trip_purpose
.. autofunction:: choose_intermediate_trip_purpose
.. autofunction:: run_trip_purpose
```
