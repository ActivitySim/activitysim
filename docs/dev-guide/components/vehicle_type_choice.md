(component-vehicle-type-choice)=
# Vehicle Type Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.vehicle_type_choice
```

The vehicle type choice model selects a vehicle type for each household vehicle. A vehicle type
is a combination of the vehicle's body type, age, and fuel type.  For example, a 13 year old
gas powered van would have a vehicle type of *van_13_gas*.

There are two vehicle type choice model structures implemented:

1. Simultaneous choice of body type, age, and fuel type.
2. Simultaneous choice of body type and age, with fuel type assigned from a probability distribution.

## Structure

- *Configuration File*: `vehicle_type_choice.yaml`

Input vehicle type data included in [prototype_mtc_extended](prototype_mtc_extended) came from a variety of sources. The number of vehicle makes, models, MPG, and
electric vehicle range was sourced from the Enivornmental Protection Agency (EPA).  Additional data on vehicle costs were derived from the
National Household Travel Survey. Auto operating costs in the vehicle type data file were a sum of fuel costs and maintenance costs.
Fuel costs were calculated from MPG assuming a $3.00 cost for a gallon of gas. When MPG was not available to calculate fuel costs,
the closest year, vehicle type, or body type available was used. Maintenance costs were taken from AAA's
[2017 driving cost study](https://exchange.aaa.com/wp-content/uploads/2017/08/17-0013_Your-Driving-Costs-Brochure-2017-FNL-CX-1.pdf).
Size categories within body types were averaged, e.g. car was an average of AAA's small, medium, and large sedan categories.
Motorcycles were assigned the small sedan maintenance costs since they were not included in AAA's report.
Maintenance costs were not varied by vehicle year. (According to
`data from the U.S. [Bureau of Labor Statistics](https://www.bls.gov/opub/btn/volume-3/pdf/americans-aging-autos.pdf),
there was no consistent relationship between vehicle age and maintenance costs.)

Using the above methodology, the average auto operating costs of vehicles output from :ref:`prototype_mtc_extended` was 18.4 cents.
This value is very close to the auto operating cost of 18.3 cents used in [prototype_mtc](prototype_mtc).
Non-household vehicles in prototype_mtc_extended use the auto operating cost of 18.3 cents used in prototype_mtc.
Users are encouraged to make their own assumptions and calculate auto operating costs as they see fit.

The distribution of fuel type probabilities included in [prototype_mtc_extended](prototype_mtc_extended) are computed directly from the National Household Travel Survey data
and include the entire US. Therefore, there is "lumpiness" in probabilities due to poor statistics in the data for some vehicle types.
The user is encouraged to adjust the probabilities to their modeling region and "smooth" them for more consistent results.

Further discussion of output results and model sensitivities can be found [here](https://github.com/ActivitySim/activitysim/wiki/Project-Meeting-2022.05.05).


## Configuration

```{eval-rst}
.. autopydantic_model:: VehicleTypeChoiceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc_extended/configs/vehicle_type_choice.yaml)


## Implementation

```{eval-rst}
.. autofunction:: vehicle_type_choice
.. autofunction:: append_probabilistic_vehtype_type_choices
.. autofunction:: annotate_vehicle_type_choice_households
.. autofunction:: annotate_vehicle_type_choice_persons
.. autofunction:: annotate_vehicle_type_choice_vehicles
.. autofunction:: get_combinatorial_vehicle_alternatives
.. autofunction:: construct_model_alternatives
.. autofunction:: get_vehicle_type_data
.. autofunction:: iterate_vehicle_type_choice
```
