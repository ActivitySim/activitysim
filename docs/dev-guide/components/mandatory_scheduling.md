(component-mandatory-scheduling)=
# Mandatory Tour Scheduling

```{eval-rst}
.. currentmodule:: activitysim.abm.models.mandatory_scheduling
```

The mandatory tour scheduling model selects a tour departure and duration period (and therefore a
start and end period as well) for each mandatory tour.   The primary drivers in the model are
accessibility-based parameters such as the mode choice logsum for the departure/arrival hour
combination, demographics, and time pattern characteristics such as the time windows available
from previously scheduled tours. This model uses person :ref:`time_windows`.

```{note}
For `prototype_mtc`, the modeled time periods for all submodels are hourly from 3 am to
3 am the next day, and any times before 5 am are shifted to time period 5, and any times
after 11 pm are shifted to time period 23.
```

If ``tour_departure_and_duration_segments.csv`` is included in the configs, then the model
will use these representative start and end time periods when calculating mode choice logsums
instead of the specific start and end combinations for each alternative to reduce runtime.  This
feature, know as ``representative logsums``, takes advantage of the fact that the mode choice logsum,
say, from 6 am to 2 pm is very similar to the logsum from 6 am to 3 pm, and 6 am to 4 pm, and so using
just 6 am to 3 pm (with the idea that 3 pm is the "representative time period") for these alternatives is
sufficient for tour scheduling.  By reusing the 6 am to 3 pm mode choice logsum, ActivitySim saves
significant runtime.

The main interface to the mandatory tour purpose scheduling model is the
[mandatory_tour_scheduling](activitysim.abm.models.mandatory_scheduling.mandatory_tour_scheduling)
function.  This function is registered as an Inject step in the example Pipeline

## Structure

- *Configuration File*: `mandatory_tour_scheduling.yaml`
- *Core Table*: `tours`
- *Result Field*: `start`,`end`,`duration`
- *Skim Keys*: `TAZ`,`workplace_taz`,`school_taz`,`start`,`end`

This model generates only True or False outcomes, and is structured as a binary
logit model.

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/mandatory_tour_scheduling.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/mandatory_tour_scheduling.yaml)

## Implementation

```{eval-rst}
.. autofunction:: mandatory_tour_scheduling
```
