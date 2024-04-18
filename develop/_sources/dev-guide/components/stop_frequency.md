(component-stop-frequency)=
# Stop Frequency

```{eval-rst}
.. currentmodule:: activitysim.abm.models.stop_frequency
```

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
[stop_frequency](activitysim.abm.models.stop_frequency.stop_frequency)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `stop_frequency.yaml`
- *Core Table*: `tours`
- *Result Field*: `stop_frequency`

## Configuration

```{eval-rst}
.. autopydantic_model:: StopFrequencySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/stop_frequency.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/stop_frequency.yaml)


## Implementation

```{eval-rst}
.. autofunction:: stop_frequency
```
