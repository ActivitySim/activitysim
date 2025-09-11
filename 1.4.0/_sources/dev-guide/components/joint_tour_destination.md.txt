(component-joint-tour-destination)=
# Joint Tour Destination

```{eval-rst}
.. currentmodule:: activitysim.abm.models.joint_tour_destination
```

The joint tour destination choice model operate similarly to the usual work and
school location choice model, selecting the primary destination for travel tours. The only
procedural difference between the models is that the usual work and school location choice
model selects the usual location of an activity whether or not the activity is undertaken during the
travel day, while the joint tour destination choice model selects the location for an
activity which has already been generated.

The tour's primary destination is the location of the activity that is assumed to provide the greatest
impetus for engaging in the travel tour. In the household survey, the primary destination was not asked, but
rather inferred from the pattern of stops in a closed loop in the respondents' travel diaries. The
inference was made by weighing multiple criteria including a defined hierarchy of purposes, the
duration of activities, and the distance from the tour origin. The model operates in the reverse
direction, designating the primary purpose and destination and then adding intermediate stops
based on spatial, temporal, and modal characteristics of the inbound and outbound journeys to
the primary destination.

The joint tour destination choice model is made up of three model steps:
  * sample - selects a sample of alternative locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative location.
  * simulate - starts with the table created above and chooses a final location, this time with the mode choice logsum included.

Joint tour location choice for [multiple_zone_systems](multiple_zone_systems) models uses [presampling](presampling) by default.

The main interface to the model is the [joint_tour_destination](activitysim.abm.models.joint_tour_destination.joint_tour_destination)
function.  This function is registered as an Inject step in the example Pipeline.  See [writing_logsums](writing_logsums) for how
to write logsums for estimation.

## Structure

- *Configuration File*: `joint_tour_destination.yaml`
- *Core Table*: `tours`
- *Result Field*: `destination`
- *Skims Keys*: `TAZ, alt_dest, MD time period`


## Configuration

```{eval-rst}
.. autopydantic_model:: TourLocationComponentSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/joint_tour_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/joint_tour_destination.yaml)

## Implementation

```{eval-rst}
.. autofunction:: joint_tour_destination
```
