(component-atwork-subtour-mode-choice)=
# At-work Subtour Mode

```{eval-rst}
.. currentmodule:: activitysim.abm.models.atwork_subtour_mode_choice
```

The at-work subtour mode choice model assigns a travel mode to each at-work subtour using the `tour_mode_choice` model.

The main interface to the at-work subtour mode choice model is the
[atwork_subtour_mode_choice](activitysim.abm.models.atwork_subtour_mode_choice.atwork_subtour_mode_choice)
function.  This function is called in the Inject step `atwork_subtour_mode_choice` and
is registered as an Inject step in the example Pipeline.
[writing_logsums](writing_logsums) for how to write logsums for estimation.

## Structure

- *Configuration File*: `tour_mode_choice.yaml`
- *Core Table*: `tour`
- *Result Field*: `tour_mode`
- *Skims keys*: `workplace_taz, destination, start, end`

## Configuration

```{eval-rst}
.. autopydantic_model:: TourModeComponentSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/tour_mode_choice.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/tour_mode_choice.yaml)


## Implementation

```{eval-rst}
.. autofunction:: atwork_subtour_mode_choice
```
