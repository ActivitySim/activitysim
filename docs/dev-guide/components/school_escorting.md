(component-school-escorting)=
# School Escorting

```{eval-rst}
.. currentmodule:: activitysim.abm.models.school_escorting
```
The school escort model determines whether children are dropped-off at or picked-up from school,
simultaneously with the chaperone responsible for chauffeuring the children,
which children are bundled together on half-tours, and the type of tour (pure escort versus rideshare).
The model is run after work and school locations have been chosen for all household members,
and after work and school tours have been generated and scheduled.
The model labels household members of driving age as potential ‘chauffeurs’ and children with school tours as potential ‘escortees’.
The model then attempts to match potential chauffeurs with potential escortees in a choice model whose alternatives
consist of ‘bundles’ of escortees with a chauffeur for each half tour.

School escorting is a household level decision – each household will choose an alternative from the ``school_escorting_alts.csv`` file,
with the first alternative being no escorting. This file contains the following columns:


| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |


## Structure

- *Configuration File*: `free_parking.yaml`
- *Core Table*: `persons`
- *Result Field*: `free_parking_at_work`
| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |



## Configuration

```{eval-rst}
.. autopydantic_model:: SchoolEscortingSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/free_parking.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/free_parking.yaml)

## Implementation

```{eval-rst}
.. autofunction:: school_escorting
```
