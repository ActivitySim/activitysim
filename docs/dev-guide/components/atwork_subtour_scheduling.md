(component-atwork-subtour-scheduling)=
# At-work Subtour Scheduling

```{eval-rst}
.. currentmodule:: activitysim.abm.models.atwork_subtour_scheduling
```

The at-work subtours scheduling model selects a tour departure and duration period (and therefore a start and end
period as well) for each at-work subtour.  This model uses person `time_windows`.

This model is the same as the mandatory tour scheduling model except it operates on the at-work tours and
constrains the alternative set to available person `time_windows`.  The at-work subtour scheduling model does not use mode choice logsums.
The at-work subtour frequency model can choose multiple tours so this model must process all first tours and then second
tours since isFirstAtWorkTour is an explanatory variable.

Choosers: at-work tours
Alternatives: alternative departure time and arrival back at origin time pairs WITHIN the work tour departure time and arrival time back at origin AND the person time window. If no time window is available for the tour, make the first and last time periods within the work tour available, make the choice, and log the number of times this occurs.
Dependent tables: skims, person, land use, work tour
Outputs: at-work tour departure time and arrival back at origin time, updated person time windows

The main interface to the at-work subtours scheduling model is the
[atwork_subtour_scheduling](activitysim.abm.models.atwork_subtour_scheduling.atwork_subtour_scheduling)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `tour_scheduling_atwork.yaml`
- *Core Table*: `tours`
- *Result Field*: `start, end, duration`
- *Skims keys*: `workplace_taz, alt_dest, MD time period, MD time period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourSchedulingSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/tour_scheduling_atwork.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/tour_scheduling_atwork.yaml)


## Implementation

```{eval-rst}
.. autofunction:: atwork_subtour_scheduling
```
