(component-location_choice)=
# School Location

```{eval-rst}
.. currentmodule:: activitysim.abm.models.location_choice.school_location
```

The usual school location choice models assign a usual school location for the primary
mandatory activity of each child and university student in the
synthetic population. The models are composed of a set of accessibility-based parameters
(including one-way distance between home and primary destination and the tour mode choice
logsum - the expected maximum utility in the mode choice model which is given by the
logarithm of the sum of exponentials in the denominator of the logit formula) and size terms,
which describe the quantity of grade-school or university opportunities in each possible
destination.

The school location model is made up of four steps:
  * sampling - selects a sample of alternative school locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative school location.
  * simulate - starts with the table created above and chooses a final school location, this time with the mode choice logsum included.
  * shadow prices - compare modeled zonal destinations to target zonal size terms and calculate updated shadow prices.

These steps are repeated until shadow pricing convergence criteria are satisfied or a max number of iterations is reached. See [shadow_pricing](shadow_pricing).

School location choice for [multiple_zone_systems](multiple_zone_systems) models uses [presampling](presampling) by default.

The main interfaces to the model is the [school_location](activitysim.abm.models.location_choice.school_location) function.
This function is registered as an Inject step in the example Pipeline. [writing_logsums](writing_logsums) for how to write logsums for estimation.

## Structure

- *Configuration File*: `school_location.yaml`
- *Core Table*: `persons`
- *Result Field*: `school_taz`
- *School Location - Skims Keys*: `TAZ, alt_dest, AM time period, MD time period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourLocationComponentSettings
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/school_location.yaml)
- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/school_location.yaml)


## Implementation

```{eval-rst}
.. autofunction:: school_location
```
