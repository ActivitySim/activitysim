(component-atwork-subtour-frequency)=
# At-work Subtours Frequency

```{eval-rst}
.. currentmodule:: activitysim.abm.models.atwork_subtour_frequency
```

The at-work subtour frequency model selects the number of at-work subtours made for each work tour.
It also creates at-work subtours by adding them to the tours table in the data pipeline.
These at-work sub-tours are travel tours taken during the workday with their origin at the work
location, rather than from home. Explanatory variables include employment status,
income, auto ownership, the frequency of other tours, characteristics of the parent work tour, and
characteristics of the workplace zone.

Choosers: work tours
Alternatives: none, 1 eating out tour, 1 business tour, 1 maintenance tour, 2 business tours, 1 eating out tour + 1 business tour
Dependent tables: household, person, accessibility
Outputs: work tour subtour frequency choice, at-work tours table (with only tour origin zone at this point)

The main interface to the at-work subtours frequency model is the
[atwork_subtour_frequency](activitysim.abm.models.atwork_subtour_frequency.atwork_subtour_frequency)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `atwork_subtour_frequency.yaml`
- *Core Table*: `tours`
- *Result Field*: `atwork_subtour_frequency`

## Configuration

```{eval-rst}
.. autopydantic_model:: AtworkSubtourFrequencySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/atwork_subtour_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/atwork_subtour_destination.yaml)


## Implementation

```{eval-rst}
.. autofunction:: atwork_subtour_frequency
```
