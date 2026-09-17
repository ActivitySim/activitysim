(component-parking-location-choice)=
# Parking Location Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.parking_location_choice
```

The parking location choice model selects a parking location for specified trips. While the model does not
require parking location be applied to any specific set of trips, it is usually applied for drive trips to
specific zones (e.g., CBD) in the model.

The model provides provides a filter for both the eligible choosers and eligible parking location zone. The
trips dataframe is the chooser of this model. The zone selection filter is applied to the land use zones
dataframe.

If this model is specified in the pipeline, the `Write Trip Matrices`_ step will using the parking location
choice results to build trip tables in lieu of the trip destination.

The main interface to the trip mode choice model is the
[parking_location_choice](activitysim.abm.models.parking_location_choice.parking_location_choice) function.  This function
is registered as an Inject step, and it is available from the pipeline.

## Structure

- *Configuration File*: `parking_location_choice.yaml`
- *Core Table*: `trips`
- *Result*: `omx trip matrices`
- *Skims*: `odt_skims: Origin to Destination by Time of Day`, `dot_skims: Destination to Origin by Time of Day`,
`opt_skims: Origin to Parking Zone by Time of Day`, `pdt_skims: Parking Zone to Destination by Time of Day`,
`od_skims: Origin to Destination`, `do_skims: Destination to Origin`, `op_skims: Origin to Parking Zone`,
`pd_skims: Parking Zone to Destination`

#### Required YAML attributes:

- `SPECIFICATION`:
    This file defines the logit specification for each chooser segment.
- `COEFFICIENTS`:
    Specification coefficients
- `PREPROCESSOR`:
    Preprocessor definitions to run on the chooser dataframe (trips) before the model is run
- `CHOOSER_FILTER_COLUMN_NAME`:
    Boolean field on the chooser table defining which choosers are eligible to parking location choice model. If no
    filter is specified, all choosers (trips) are eligible for the model.
- `CHOOSER_SEGMENT_COLUMN_NAME`:
    Column on the chooser table defining the parking segment for the logit model
- `SEGMENTS`:
    List of eligible chooser segments in the logit specification
- `ALTERNATIVE_FILTER_COLUMN_NAME`:
    Boolean field used to filter land use zones as eligible parking location choices. If no filter is specified,
    then all land use zones are considered as viable choices.
- `ALT_DEST_COL_NAME`:
    The column name to append with the parking location choice results. For choosers (trips) ineligible for this
    model, a -1 value will be placed in column.
- `TRIP_ORIGIN`:
    Origin field on the chooser trip table
- `TRIP_DESTINATION`:
    Destination field on the chooser trip table


## Configuration

```{eval-rst}
.. autopydantic_model:: ParkingLocationSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/parking_location_choice.yaml)

## Implementation

```{eval-rst}
.. autofunction:: parking_location
.. autofunction:: wrap_skims
.. autofunction:: parking_destination_simulate
```
