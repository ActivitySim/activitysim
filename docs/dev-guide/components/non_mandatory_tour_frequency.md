(component-non-mandatory-tour-frequency)=
# Non-Mandatory Tour Frequency

```{eval-rst}
.. currentmodule:: activitysim.abm.models.non_mandatory_tour_frequency
```

The non-mandatory tour frequency model selects the number of non-mandatory tours made by each person on the simulation day.
It also adds non-mandatory tours to the tours in the data pipeline. The individual non-mandatory tour frequency model
operates in two stages:

  * A choice is made using a random utility model between combinations of tours containing zero, one, and two or more escort tours, and between zero and one or more tours of each other purpose.
  * Up to two additional tours of each purpose are added according to fixed extension probabilities.

The main interface to the non-mandatory tour purpose frequency model is the
[non_mandatory_tour_frequency](activitysim.abm.models.non_mandatory_tour_frequency.non_mandatory_tour_frequency)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `non_mandatory_tour_frequency.yaml`
- *Core Table*: `persons`
- *Result Field*: `non_mandatory_tour_frequency`

## Configuration

```{eval-rst}
.. autopydantic_model:: NonMandatoryTourFrequencySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/non_mandatory_tour_frequency.yaml)

## Implementation

```{eval-rst}
.. autofunction:: non_mandatory_tour_frequency
```
