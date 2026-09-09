(component-write-trip-matrices)=
# Write Trip Matrices

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_matrices
```

Write open matrix (OMX) trip matrices for assignment.  Reads the trips table post preprocessor and run expressions
to code additional data fields, with one data fields for each matrix specified.  The matrices are scaled by a
household level expansion factor, which is the household sample rate by default, which is calculated when
households are read in at the beginning of a model run.  The main interface to write trip
matrices is the [write_trip_matrices](activitysim.abm.models.trip_matrices.write_trip_matrices) function.
This function is registered as an Inject step in the example Pipeline.

## Structure

- *Core Table*: `trips`
- *Result*: `omx trip matrices`
- *Skims Keys*: `origin, destination`

This model generates only True or False outcomes, and is structured as a binary
logit model.


## Park-and-Ride Legs

Each entry in `MATRICES[].tables` can specify `origin` and `destination`
columns; they default to `origin` and `destination`. This allows a trip using
{ref}`component-park-and-ride-lot-choice` to contribute drive and transit legs
through `pnr_zone_id`. Entries with the same `name` within one output file
accumulate into the same OMX matrix.

For example, the following fragment routes outbound drive access to the lot
and outbound transit from the lot to the destination:

```yaml
MATRICES:
  - file_name: trip_matrices.omx
    tables:
      - name: auto
        data_field: pnr_outbound
        origin: origin
        destination: pnr_zone_id
      - name: transit
        data_field: pnr_outbound
        origin: pnr_zone_id
        destination: destination
```

Before writing matrices, use the `preprocessor` to map the final tour lot onto
the trips table by `tour_id` and create the data fields to aggregate. In this
example, `pnr_outbound` is a count or weight for the applicable outbound
park-and-ride trips and zero for other trips. It must check actual mode use
and a valid lot, since a candidate lot alone does not imply park-and-ride use.
Create corresponding inbound fields and entries for destination-side origin
to lot (transit), then lot to home-side destination (auto). Implementations
with intermediate stops must define which trips use each portion of the path.

These column names are illustrative, not automatically generated fields. The
writer does not split trips automatically. Supply endpoint columns in a
consistent zone system suitable for the output domain; for two-zone outputs,
land-use `TAZ` mappings are used when conversion is needed. Household expansion
weights apply to these entries as they do to other matrices.

## Configuration

```{eval-rst}
.. autopydantic_model:: WriteTripMatricesSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/write_trip_matrices.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/write_trip_matrices.yaml)

## Implementation

```{eval-rst}
.. autofunction:: write_trip_matrices
```
