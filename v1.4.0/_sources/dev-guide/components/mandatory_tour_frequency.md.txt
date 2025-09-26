(component-mandatory-tour-frequency)=
# Mandatory Tour Frequency

```{eval-rst}
.. currentmodule:: activitysim.abm.models.mandatory_tour_frequency
```

The individual mandatory tour frequency model predicts the number of work and school tours
taken by each person with a mandatory DAP. The primary drivers of mandatory tour frequency
are demographics, accessibility-based parameters such as drive time to work, and household
automobile ownership.  It also creates mandatory tours in the data pipeline.

The main interface to the mandatory tour purpose frequency model is the
[mandatory_tour_frequency](activitysim.abm.models.mandatory_tour_frequency.mandatory_tour_frequency)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `mandatory_tour_frequency.yaml`
- *Core Table*: `persons`
- *Result Field*: `mandatory_tour_frequency`

This model generates only True or False outcomes, and is structured as a binary
logit model.


## Configuration

```{eval-rst}
.. autopydantic_model:: MandatoryTourFrequencySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/mandatory_tour_frequency.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/mandatory_tour_frequency.yaml)

## Implementation

```{eval-rst}
.. autofunction:: mandatory_tour_frequency
```
