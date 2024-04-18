(component-auto-ownership)=
# Auto Ownership

```{eval-rst}
.. currentmodule:: activitysim.abm.models.auto_ownership
```

The auto ownership model selects a number of autos for each household in the simulation.
The primary model components are household demographics, zonal density, and accessibility.

## Structure

- *Configuration File*: `auto_ownership.yaml`
- *Core Table*: `households`
- *Result Field*: `auto_owenership`

This model is typically structured as multinomial logit model.

## Configuration

```{eval-rst}
.. autopydantic_model:: AutoOwnershipSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/auto_ownership.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/auto_ownership.yaml)

## Implementation

```{eval-rst}
.. autofunction:: auto_ownership_simulate
```
