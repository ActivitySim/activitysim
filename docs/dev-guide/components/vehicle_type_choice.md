(component-vehicle-type-choice)=
# Vehicle Type Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.vehicle_type_choice
```

The vehicle type choice model selects a vehicle type for each household vehicle. A vehicle type
is a combination of the vehicle's body type, age, and fuel type.  For example, a 13 year old
gas powered van would have a vehicle type of *van_13_gas*.

There are two vehicle type choice model structures implemented:

1. Simultaneous choice of body type, age, and fuel type.
2. Simultaneous choice of body type and age, with fuel type assigned from a probability distribution.

## Structure

- *Configuration File*: `vehicle_type_choice.yaml`
   The *vehicle_type_choice.yaml* file contains the following model specific options:
  - ``SPEC``: Filename for input utility expressions
  - ``COEFS``: Filename for input utility expression coefficients
  - `LOGIT_TYPE``: Specifies whether you are using a nested or multinomial logit structure
  - ``combinatorial_alts``: Specifies the alternatives for the choice model.
  Has sub-categories of ``body_type``, ``age``, and ``fuel_type``.
  -  ``PROBS_SPEC``: Filename for input fuel type probabilities. Supplying probabilities
  corresponds to implementation structure 2 above, and not supplying probabilities would correspond to implementation structure 1.
  If provided, the ``fuel_type`` category in ``combinatorial_alts``
  will be excluded from the model alternatives such that only body type and age are selected.  Input ``PROBS_SPEC`` table will have an index
  column named *vehicle_type* which is a combination of body type and age in the form ``{body type}_{age}``.  Subsequent column names
  specify the fuel type that will be added and the column values are the probabilities of that fuel type.
  The vehicle type model will select a fuel type for each vehicle based on the provided probabilities.
  - ``VEHICLE_TYPE_DATA_FILE``: Filename for input vehicle type data. Must have columns ``body_type``, ``fuel_type``, and ``vehicle_year``.
  Vehicle ``age`` is computed using the ``FLEET_YEAR`` option. Data for every alternative specified in the ``combinatorial_alts`` option must be included
  in the file. Vehicle type data file will be joined to the alternatives and can be used in the utility expressions if ``PROBS_SPEC`` is not provided.
  If ``PROBS_SPEC`` is provided, the vehicle type data will be joined after a vehicle type is decided so the data can be used in downstream models.
  - ``COLS_TO_INCLUDE_IN_VEHICLE_TABLE``: List of columns from the vehicle type data file to include in the vehicle table that can be used in downstream models.
  Examples of data that might be needed is vehicle range for the :ref:`vehicle_allocation` model, auto operating costs to use in tour and trip mode choice,
  and emissions data for post-model-run analysis.
  - ``FLEET_YEAR``: Integer specifying the fleet year to be used in the model run. This is used to compute ``age`` in the
  vehicle type data table where ``age = (1 + FLEET_YEAR - vehicle_year)``. Computing age on the fly with the ``FLEET_YEAR`` variable allows the
  user flexibility to compile and share a single vehicle type data file containing all years and simply change the ``FLEET_YEAR`` to run
  different scenario years.
  - Optional additional settings that work the same in other models are constants, expression preprocessor, and annotate tables.
- *Core Table*: 
- *Result Field*: 

Input vehicle type data included in :ref:`prototype_mtc_extended` came from a variety of sources. The number of vehicle makes, models, MPG, and
electric vehicle range was sourced from the Enivornmental Protection Agency (EPA).  Additional data on vehicle costs were derived from the
National Household Travel Survey. Auto operating costs in the vehicle type data file were a sum of fuel costs and maintenance costs.
Fuel costs were calculated from MPG assuming a $3.00 cost for a gallon of gas. When MPG was not available to calculate fuel costs,
the closest year, vehicle type, or body type available was used. Maintenance costs were taken from AAA's
[2017 driving cost study](https://exchange.aaa.com/wp-content/uploads/2017/08/17-0013_Your-Driving-Costs-Brochure-2017-FNL-CX-1.pdf).
Size categories within body types were averaged, e.g. car was an average of AAA's small, medium, and large sedan categories.
Motorcycles were assigned the small sedan maintenance costs since they were not included in AAA's report.
Maintenance costs were not varied by vehicle year. (According to
`data from the U.S. Bureau of Labor Statistics <https://www.bls.gov/opub/btn/volume-3/pdf/americans-aging-autos.pdf>`_,
there was no consistent relationship between vehicle age and maintenance costs.)

Using the above methodology, the average auto operating costs of vehicles output from :ref:`prototype_mtc_extended` was 18.4 cents.
This value is very close to the auto operating cost of 18.3 cents used in :ref:`prototype_mtc`.
Non-household vehicles in prototype_mtc_extended use the auto operating cost of 18.3 cents used in prototype_mtc.
Users are encouraged to make their own assumptions and calculate auto operating costs as they see fit.

The distribution of fuel type probabilities included in :ref:`prototype_mtc_extended` are computed directly from the National Household Travel Survey data
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
```
