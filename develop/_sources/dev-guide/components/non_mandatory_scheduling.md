(component-non-mandatory-scheduling)=
# Non-Mandatory Tour Scheduling

```{eval-rst}
.. currentmodule:: activitysim.abm.models.non_mandatory_scheduling
```
The non-mandatory tour scheduling model selects a tour departure and duration period (and therefore a start and end
period as well) for each non-mandatory tour.  This model uses person [time_windows](time_windows).  Includes support
for [representative_logsums](representative_logsums).

The main interface to the non-mandatory tour purpose scheduling model is the
[non_mandatory_tour_scheduling](activitysim.abm.models.non_mandatory_scheduling.non_mandatory_tour_scheduling)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `non_mandatory_tour_scheduling.yaml`
- *Core Table*: `tours`
- *Result Field*: `start, end, duration`
- *Skims Keys*: `TAZ, destination, MD time period, MD time period`

### Examples

- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/non_mandatory_tour_scheduling.yaml)

## Implementation

```{eval-rst}
.. autofunction:: non_mandatory_tour_scheduling
```
