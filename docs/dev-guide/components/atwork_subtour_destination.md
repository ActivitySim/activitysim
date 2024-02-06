(component-atwork-subtour-destination)=
# At-work Subtours Destination Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.atwork_subtour_destination
```

The at-work subtours destination choice model is made up of three model steps:

  * sample - selects a sample of alternative locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative location.
  * simulate - starts with the table created above and chooses a final location, this time with the mode choice logsum included.

At-work subtour location choice for [multiple_zone_systems](multiple_zone_systems) models uses [presampling](presampling) by default.

The main interface to the at-work subtour destination model is the
[atwork_subtour_destination](ctivitysim.abm.models.atwork_subtour_destination.atwork_subtour_destination)
function.  This function is registered as an Inject step in the example Pipeline.
[writing_logsums](writing_logsums) for how to write logsums for estimation.

## Structure

- *Configuration File*: `atwork_subtour_destination.yaml`
- *Core Table*: `tours`
- *Result Field*: `destination`
- *Skims keys*: `workplace_taz, alt_dest, MD time period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourLocationComponentSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/atwork_subtour_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/atwork_subtour_destination.yaml)


## Implementation

```{eval-rst}
.. autofunction:: atwork_subtour_destination
```
