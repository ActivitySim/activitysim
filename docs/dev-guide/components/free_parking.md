(component-free-parking)=
# Free Parking Eligibility

```{eval-rst}
.. currentmodule:: activitysim.abm.models.free_parking
```

The Free Parking Eligibility model predicts the availability of free parking at a person's
workplace.  It is applied for people who work in zones that have parking charges, which are
generally located in the Central Business Districts. The purpose of the model is to adequately
reflect the cost of driving to work in subsequent models, particularly in mode choice.

## Structure

- *Configuration File*: `free_parking.yaml`
- *Core Table*: `persons`
- *Result Field*: `free_parking_at_work`

This model generates only True or False outcomes, and is structured as a binary
logit model.


## Configuration

```{eval-rst}
.. autopydantic_model:: FreeParkingSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/free_parking.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/free_parking.yaml)

## Implementation

```{eval-rst}
.. autofunction:: free_parking
```
