(component-location_choice)=
# Work Location

```{eval-rst}
.. currentmodule:: activitysim.abm.models.location_choice.workplace_location
```

The usual work location choice models assign a usual work location for the primary
mandatory activity of each employed person in the
synthetic population. The models are composed of a set of accessibility-based parameters
(including one-way distance between home and primary destination and the tour mode choice
logsum - the expected maximum utility in the mode choice model which is given by the
logarithm of the sum of exponentials in the denominator of the logit formula) and size terms,
which describe the quantity of work opportunities in each possible destination.

The work location model is made up of four steps:
  * sample - selects a sample of alternative work locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative work location.
  * simulate - starts with the table created above and chooses a final work location, this time with the mode choice logsum included.
  * shadow prices - compare modeled zonal destinations to target zonal size terms and calculate updated shadow prices.

These steps are repeated until shadow pricing convergence criteria are satisfied or a max number of iterations is reached.  See [shadow_pricing](shadow_pricing).

Work location choice for [multiple_zone_systems](multiple_zone_systems) models uses [presampling](presampling) by default.

The main interfaces to the model is the [workplace_location](activitysim.abm.models.location_choice.workplace_location) function.
This function is registered as an Inject step in the example Pipeline.  See [writing_logsums](writing_logsums) for how to write logsums for estimation.

## Structure

- *Configuration File*: `workplace_location.yaml`
- *Core Table*: `persons`
- *Result Field*: `workplace_taz`
- *School Location - Skims Keys*: `TAZ, alt_dest, AM time period, PM time period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourLocationComponentSettings
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/workplace_location.yaml)
- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/workplace_location.yaml)


## Implementation

```{eval-rst}
.. autofunction:: workplace_location
```
