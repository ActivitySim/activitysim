(component-explicit_telecommute)=
# Explicit Telecommute

```{eval-rst}
.. currentmodule:: activitysim.abm.models.explicit_telecommute
```

Telecommuting is defined as workers who work from home instead of going
to work. It only applies to workers with a regular workplace outside of home.
The ActivitySim telecommute model consists of two long term submodels - a person [work_from_home](work_from_home) model and a
person [telecommute_frequency](telecommute_frequency) model. The work from home model predicts if a worker works exclusively from home, 
whereas the telecommute frequency model predicts number of days in a week a worker telecommuters, if they do not exclusively work from home. 
However, neither of them predicts whether a worker is telecommuting or not on the simulation day. 
This explicit telecommute model extends the previous two models to predicts for all workers whether they
are telecommuting on the simulation day.

An simple implementation of the explicit telecommute model can be based on the telecommute frequency.
For example, if a worker telecommutes 4 days a week, then there is a 80% probabilty for them to telecommute
on the simulation day.

The main interface to the explicit telecommute model is the
[explicit_telecommute](activitysim.abm.models.explicit_telecommute) function.  This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `explicit_telecommute.yaml`
- *Core Table*: `persons`
- *Result Table*: `is_telecommuting`


## Configuration

```{eval-rst}
.. autopydantic_model:: ExplicitTelecommuteSetting
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Example SANDAG ABM3](https://github.com/ActivitySim/sandag-abm3-example/tree/main/configs/resident/explicit_telecommute.yaml)


## Implementation

```{eval-rst}
.. autofunction:: explicit_telecommute
```
