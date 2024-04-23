(component-trip-mode-choice)=
# Trip Mode Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_mode_choice
```

The trip mode choice model assigns a travel mode for each trip on a given tour. It
operates similarly to the tour mode choice model, but only certain trip modes are available for
each tour mode. The correspondence rules are defined according to the following principles:

  * Pay trip modes are only available for pay tour modes (for example, drive-alone pay is only available at the trip mode level if drive-alone pay is selected as a tour mode).
  * The auto occupancy of the tour mode is determined by the maximum occupancy across all auto trips that make up the tour. Therefore, the auto occupancy for the tour mode is the maximum auto occupancy for any trip on the tour.
  * Transit tours can include auto shared-ride trips for particular legs. Therefore, 'casual carpool', wherein travelers share a ride to work and take transit back to the tour origin, is explicitly allowed in the tour/trip mode choice model structure.
  * The walk mode is allowed for any trip.
  * The availability of transit line-haul submodes on transit tours depends on the skimming and tour mode choice hierarchy. Free shared-ride modes are also available in walk-transit tours, albeit with a low probability. Paid shared-ride modes are not allowed on transit tours because no stated preference data is available on the sensitivity of transit riders to automobile value tolls, and no observed data is available to verify the number of people shifting into paid shared-ride trips on transit tours.

The trip mode choice models explanatory variables include household and person variables, level-of-service
between the trip origin and destination according to the time period for the tour leg, urban form
variables, and alternative-specific constants segmented by tour mode.

The main interface to the trip mode choice model is the
[trip_mode_choice](activitysim.abm.models.trip_mode_choice.trip_mode_choice) function.  This function
is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `trip_mode_choice.yaml`
- *Result Field*: `trip_mode`
- *Skim Keys*: `origin, destination, trip_period`

## Configuration

```{eval-rst}
.. autopydantic_model:: TripModeChoiceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/trip_mode_choice.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_mode_choice.yaml)

## Implementation

```{eval-rst}
.. autofunction:: trip_mode_choice
```
