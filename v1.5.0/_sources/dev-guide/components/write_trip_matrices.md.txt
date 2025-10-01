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
