(component-tour-mode-choice)=
# Tour Mode Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.tour_mode_choice
```

The mandatory, non-mandatory, and joint tour mode choice model assigns to each tour the "primary" mode that
is used to get from the origin to the primary destination. The tour-based modeling approach requires a reconsideration
of the conventional mode choice structure. Instead of a single mode choice model used in a four-step
structure, there are two different levels where the mode choice decision is modeled: (a) the
tour mode level (upper-level choice); and, (b) the trip mode level (lower-level choice conditional
upon the upper-level choice).

The mandatory, non-mandatory, and joint tour mode level represents the decisions that apply to the entire tour, and
that will affect the alternatives available for each individual trip or joint trip. These decisions include the choice to use a private
car versus using public transit, walking, or biking; whether carpooling will be considered; and
whether transit will be accessed by car or by foot. Trip-level decisions correspond to details of
the exact mode used for each trip, which may or may not change over the trips in the tour.

The mandatory, non-mandatory, and joint tour mode choice structure is a nested logit model which separates
similar modes into different nests to more accurately model the cross-elasticities between the alternatives. The
eighteen modes are incorporated into the nesting structure specified in the model settings file. The
first level of nesting represents the use a private car, non-motorized
means, or transit. In the second level of nesting, the auto nest is divided into vehicle occupancy
categories, and transit is divided into walk access and drive access nests. The final level splits
the auto nests into free or pay alternatives and the transit nests into the specific line-haul modes.

The primary variables are in-vehicle time, other travel times, cost (the influence of which is derived
from the automobile in-vehicle time coefficient and the persons' modeled value of time),
characteristics of the destination zone, demographics, and the household's level of auto
ownership.

The main interface to the mandatory, non-mandatory, and joint tour mode model is the
[tour_mode_choice_simulate](activitysim.abm.models.tour_mode_choice.tour_mode_choice_simulate) function.  This function is
called in the Inject step [tour_mode_choice_simulate](tour_mode_choice_simulate) and is registered as an Inject step in the example Pipeline.
See [writing_logsums](writing_logsums) for how to write logsums for estimation.

## Structure

- *Configuration File*: `tour_mode_choice.yaml`
- *Core Table*: `tours`
- *Result Field*: `mode`
- *Skims Keys*: `TAZ, destination, start, end`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourModeComponentSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/tour_mode_choice.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/tour_mode_choice.yaml)


## Implementation

```{eval-rst}
.. autofunction:: tour_mode_choice_simulate
.. autofunction:: create_logsum_trips
.. autofunction:: append_tour_leg_trip_mode_choice_logsums
.. autofunction:: get_trip_mc_logsums_for_all_modes
.. autofunction:: get_trip_mc_logsums_for_all_modes
```
