(component-joint-tour-composition)=
# Joint Tour Composition

```{eval-rst}
.. currentmodule:: activitysim.abm.models.joint_tour_composition
```
In the joint tour party composition model, the makeup of the travel party (adults, children, or
mixed - adults and children) is determined for each joint tour.  The party composition determines the
general makeup of the party of participants in each joint tour in order to allow the micro-simulation
to faithfully represent the prevalence of adult-only, children-only, and mixed joint travel tours
for each purpose while permitting simplicity in the subsequent person participation model.

The main interface to the joint tour composition model is the
[joint_tour_composition](activitysim.abm.models.joint_tour_composition.joint_tour_composition)
function.  This function is registered as an Inject step in the example Pipeline.


## Structure

- *Configuration File*: `joint_tour_composition.yaml`
- *Core Table*: `tours`
- *Result Field*: `composition`


## Configuration

```{eval-rst}
.. autopydantic_model:: JointTourCompositionSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/free_parking.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/joint_tour_composition.yaml)

## Implementation

```{eval-rst}
.. autofunction:: joint_tour_composition
```
