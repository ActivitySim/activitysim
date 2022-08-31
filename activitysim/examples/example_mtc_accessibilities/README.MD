
### MTC Extended Accessibilities Example

The MTC example with the following additional models:
* Disaggregate Accessibilities model

## Background
Destination choice accessibility measures are based on the logsum values calculated in the discrete choice model. These values reflect the expected utility of a choice, and provide a useful means of approximating the relative accessibility of each choice. These accessibility measures are used to ensure that upper-level models are sensitive to the choices in lower-level models.

## Purpose
This disaggregate accessibility model expands the static implementation that has only four income segments, to allow for user-defined population variables. Users can now estimate disaggregate accessibility measures for any combination of features in the persons, households, and tours tables (e.g., sex, employment, education, household size, etc.). 

## Architecture
The disaggregate accessibility model is composed of four primary steps: 

1. Synthesize user-defined proto-population
2. Run destination choice models on proto-population to calculate logsums
   - Fixed work location choice model
   - Fixed school location choice model
   - Non-mandatory destination choice model
3. Merge extracted values onto the full population data and store results
4. Initialize disaggregate accessibilities in main model run

To keep the model lightweight with minimal redundant code, it leverages existing functions and logsum calculations in ActivitySim. Only one python file (`activitysim/abm/models/disaggregate_accessibility.py`) is added to the code base and no other files are modified.

The model is intended to run in tandem with a primary ActivitySim model, utilizing the primary model's input data and config files. This avoids having to create a duplicate set of config files for the accessibility model. However, this means that the model interacts and modifies tables in the ActivitySim pipeline. To avoid corrupting the actual model pipeline, the disaggregate accessibility model runs in a separate ActivitySim process. A completely separate model pipeline is initialized, extracts the accessibility values, and then closes the pipeline. 

## Usage
There is only one config file necessary to run the disaggregate accessibility model `./configs/disaggregate_accessibility.yaml`, all other downstream models configs (i.e., fixed work/school location and non-mandatory tour destination) will be shadowed from the primary model. 

The model can be called as a standalone python script, using `./disaggregate_accessibility_model.py` or as an injected model step in the settings.yaml model list:

`- compute_disaggregate_accessibility`\
`- initialize_disaggregate_accessibility`\
*** IMPORTANT: these steps must be run first before any other model ***


### Config file

---todo