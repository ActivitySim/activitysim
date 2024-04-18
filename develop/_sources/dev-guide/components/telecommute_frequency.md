(component-telecommute-frequency)=
# Telecommute Frequency

```{eval-rst}
.. currentmodule:: activitysim.abm.models.telecommute_frequency
```

Telecommuting is defined as workers who work from home instead of going to work. It only applies to
workers with a regular workplace outside of home. The telecommute model consists of two
submodels - a person [work_from_home](work_from_home) model and this person telecommute frequency model.

For all workers that work out of the home, the telecommute models predicts the
level of telecommuting. The model alternatives are the frequency of telecommuting in
days per week (0 days, 1 day, 2 to 3 days, 4+ days).

The main interface to the work from home model is the
[telecommute_frequency](activitysim.abm.models.telecommute_frequency) function.  This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `telecommute_frequency.yaml`
- *Core Table*: `persons`
- *Result Field*: `telecommute_frequency`


## Configuration

```{eval-rst}
.. autopydantic_model:: TelecommuteFrequencySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/telecommute_frequency.yaml)
- [Prototype SEMCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_semcog/configs/telecommute_frequency.yaml)

## Implementation

```{eval-rst}
.. autofunction:: telecommute_frequency
```
