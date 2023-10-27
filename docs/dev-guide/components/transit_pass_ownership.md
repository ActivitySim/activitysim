(component-transit-pass-ownership)=
# Transit Pass Ownership


```{eval-rst}
.. currentmodule:: activitysim.abm.models.transit_pass_ownership
```

The transit fare discount is defined as persons who purchase or are
provided a transit pass.  The transit fare discount consists of two submodels - this
transit pass ownership model and a person :ref:`transit_pass_subsidy` model. The
result of this model can be used to condition downstream models such as the tour and trip
mode choice models via fare discount adjustments.

The main interface to the transit pass ownership model is the
:py:func:`~activitysim.abm.models.transit_pass_ownership` function.  This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `transit_pass_ownership.yaml`
- *Core Table*: `persons`
- *Result Field*: `transit_pass_ownership`
- *Skim Keys*: NA




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
.. autofunction:: transit_pass_ownership
```
