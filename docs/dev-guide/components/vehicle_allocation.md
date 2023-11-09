(component-vehicle_allocation)=
# Vehicle Allocation

```{eval-rst}
.. currentmodule:: activitysim.abm.models.vehicle_allocation
```

The vehicle allocation model selects which vehicle would be used for a tour of given occupancy. The alternatives for the vehicle
allocation model consist of the vehicles owned by the household and an additional non household vehicle option. (Zero-auto
households would be assigned the non-household vehicle option since there are no owned vehicles in the household).
A vehicle is selected for each occupancy level set by the user such that different tour modes that have different occupancies could see different operating
characteristics. The output of the vehicle allocation model is appended to the tour table with column names [vehicle_occup_{occupancy}](vehicle_occup_{occupancy}) and the values are
the vehicle type selected.

In [prototype_mtc_extended](prototype_mtc_extended), three occupancy levels are used: 1, 2, and 3.5.  The auto operating cost
for occupancy level 1 is used in the drive alone mode and drive to transit modes. Occupancy levels 2 and 3.5 are used for shared
ride 2 and shared ride 3+ auto operating costs, respectively.  Auto operating costs are selected in the mode choice pre-processors by selecting the allocated
vehicle type data from the vehicles table. If the allocated vehicle type was the non-household vehicle, the auto operating costs uses
the previous default value from [prototype_mtc](prototype_mtc). All trips and atwork subtours use the auto operating cost of the parent tour.  Functionality
was added in tour and atwork subtour mode choice to annotate the tour table and create a ``selected_vehicle`` which denotes the actual vehicle used.
If the tour mode does not include a vehicle, then the ``selected_vehicle`` entry is left blank.

The current implementation does not account for possible use of the household vehicles by other household members.  Thus, it is possible for a
selected vehicle to be used in two separate tours at the same time.

## Structure

- *Configuration File*: `vehicle_allocation.yaml`
- *Result Field*: `vehicle_occup_{occupancy}`

## Configuration

```{eval-rst}
.. autopydantic_model:: VehicleAllocationSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC Extended](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc_extended/configs/vehicle_type_choice.yaml)

## Implementation

```{eval-rst}
.. autofunction:: vehicle_allocation
.. autofunction:: annotate_vehicle_allocation
.. autofunction:: get_skim_dict
```
