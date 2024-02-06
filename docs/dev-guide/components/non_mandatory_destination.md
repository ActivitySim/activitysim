(component-non-mandatory-destination)=
# Non-Mandatory Destination Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.non_mandatory_destination
```

The non-mandatory tour destination choice model chooses a destination zone for
non-mandatory tours.  The three step (sample, logsums, final choice) process also used for
mandatory tour destination choice is used for non-mandatory tour destination choice.

Non-mandatory tour location choice for [multiple_zone_systems](multiple_zone_systems) models uses [presampling](presampling) by default.

The main interface to the non-mandatory tour destination choice model is the
[non_mandatory_tour_destination](activitysim.abm.models.non_mandatory_destination.non_mandatory_tour_destination)
function.  This function is registered as an Inject step in the example Pipeline.  See :ref:`writing_logsums`
for how to write logsums for estimation.

## Structure

- *Configuration File*: `non_mandatory_tour_destination.yaml`
- *Core Table*: `tours`
- *Result Field*: `destination`
- *Skims Keys*: `TAZ, alt_dest, MD time period, MD time period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourLocationComponentSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/non_mandatory_tour_destination.yaml)

## Implementation

```{eval-rst}
.. autofunction:: non_mandatory_tour_destination
```
