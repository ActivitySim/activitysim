(component-work_from_home)=
# Work from Home

```{eval-rst}
.. currentmodule:: activitysim.abm.models.work_from_home
```

Telecommuting is defined as workers who work from home instead of going
to work. It only applies to workers with a regular workplace outside of home.
The telecommute model consists of two submodels - this work from home model and a
person [telecommute_frequency](telecommute_frequency) model. This model predicts for all workers whether they
usually work from home.

The work from home model includes the ability to adjust a work from home alternative
constant to attempt to realize a work from home percent for what-if type analysis.
This iterative single process procedure takes as input a number of iterations, a filter on
the choosers to use for the calculation, a target work from home percent, a tolerance percent
for convergence, and the name of the coefficient to adjust.  An example setup is provided and
the coefficient adjustment at each iteration is:
``new_coefficient = log( target_percent / current_percent ) + current_coefficient``.

The main interface to the work from home model is the
[work_from_home](activitysim.abm.models.work_from_home) function.  This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `work_from_home.yaml`
- *Core Table*: `persons`
- *Result Table*: `work_from_home`


## Configuration

```{eval-rst}
.. autopydantic_model:: WorkFromHomeSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype SEMCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_semcog/configs/work_from_home.yaml)
- [Prototype MWCOG](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mwcog/configs/work_from_home.yaml)


## Implementation

```{eval-rst}
.. autofunction:: work_from_home
```
