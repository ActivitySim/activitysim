(component-transit-pass-subsidy)=
# Transit Pass Subsidy

```{eval-rst}
.. currentmodule:: activitysim.abm.models.transit_pass_subsidy
```

The transit pass subsidy model is a component of the transit fare discount model, which models persons who purchase or are
provided a transit pass.  The transit fare discount consists of two submodels - this
transit pass subsidy model and a person [transit_pass_ownership](transit_pass_ownership) model.  The
result of this model can be used to condition downstream models such as the
person [transit_pass_ownership](transit_pass_ownership) model and the tour and trip mode choice models
via fare discount adjustments.

The main interface to the transit pass subsidy model is the
[transit_pass_subsidy](activitysim.abm.models.transit_pass_subsidy) function.  This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `transit_pass_subsidy.yaml`
- *Core Table*: `persons`
- *Result Field*: `transit_pass_subsidy`

## Configuration

```{eval-rst}
.. autopydantic_model:: TransitPassSubsidySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/transit_pass_subsidy.yaml)
- [Prototype SEMCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_semcog/configs/transit_pass_subsidy.yaml)

## Implementation

```{eval-rst}
.. autofunction:: transit_pass_subsidy
```
