(component-joint-tour-scheduling)=
# Joint Tour Scheduling

```{eval-rst}
.. currentmodule:: activitysim.abm.models.joint_tour_scheduling
```

The joint tour scheduling model selects a tour departure and duration period (and therefore a start and end
period as well) for each joint tour.  This model uses person [time_windows](time_windows). The primary drivers in the
models are accessibility-based parameters such
as the auto travel time for the departure/arrival hour combination, demographics, and time
pattern characteristics such as the time windows available from previously scheduled tours.
The joint tour scheduling model does not use mode choice logsums.

The main interface to the joint tour purpose scheduling model is the
[joint_tour_scheduling](activitysim.abm.models.joint_tour_scheduling.joint_tour_scheduling)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `joint_tour_scheduling.yaml`
- *Core Table*: `tours`
- *Result Field*: `start, end, duration`
- *Skims Keys*: ` TAZ, destination, MD time period, MD time period`




## Configuration

```{eval-rst}
.. autopydantic_model:: TourSchedulingSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/joint_tour_scheduling.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/joint_tour_scheduling.yaml)

## Implementation

```{eval-rst}
.. autofunction:: joint_tour_scheduling
```
