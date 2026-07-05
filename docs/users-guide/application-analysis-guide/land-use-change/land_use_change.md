# Land Use Change Guide

## Introduction

Many contemporary urban planners are encouraging developers to build denser housing, particularly around transit stops. Naturally, planners will want to gauge what the impact of such a development would be on their jurisdiction's transportation system, particularly regarding metrics such as VMT (and subsequently greenhouse gas emission) and transit boardings (and subsequently farebox revenue). To demonstrate this, we will be analyzing a hypothetical development in the San Diego Region that will add 2000 households and 1000 retail jobs near a light rail station. These will be added to the existing population and jobs in the region. The guide will show how to make changes to the ActivitySim inputs, how to run the test, and how to calculate some of the key metrics such as VMT and changes in mode share.

**NOTE: The example provided is a hypothetical project that demonstrates how one would use ActivitySim to model the effects of a land use change and does not necessarily reflect any real planned developments.**

## Setting Up the Scenario

Three input files need to be changed in order to run this test: the files defining the synthetic population (households.csv and persons.csv) and the land use file (land_use.csv). Because Activity-based models use synthetic populations, those input files will need to be updated to reflect the increase in the number of households within the study area. There are multiple ways that this could be done. The ActivitySim consortium maintains the PopulationSim population synthesis software, which includes a `repop` mode that can be used to add households to an existing synthetic population. This demonstration will show how to do this, though there are many ways one could add the additional households. In addition to PopulationSim's `repop` mode, one could shift existing population from elsewhere, or even generating a wholly new synthetic population.

While updating the land use file may seem very straightforward, it is very easy to overlook some necessary changes that could result in the model understating the impact of the change. A modeler doesn't need to just edit the household and employment fields in the study area--they also need to edit any field derived from those fields. The following fields land use fields in the SANDAG ABM3 example reflect the increase in the number of multi-family households (descriptions of the fields can be found at SANDAG's [ABM3 Documentation](https://sandag.github.io/ABM/inputs.html#land-use)):
- pop
- hhp
- hs
- hs_mf
- hh
- hh_mf
- i1
- i2
- i3
- i4
- i5
- i6
- i7
- i8
- i9
- i10
- duden
- popden
- dudenbin
- PopEmpDenPerMi

Further, the following fields would need to be updated to reflect the increase in retail employment:
- emp_ret
- emp_total
- empden
- PopEmpDenPerMi

### Instructions

1. The first thing to do would be to update the new synthetic population. This can be done using PopulationSim's repop mode, which adds additional population on top of existing PopulationSim outputs (a pipeline file is needed). A more detailed explanation of PopulationSim's repop mode can be found in [PopulationSim's documentation](https://activitysim.github.io/populationsim/application_configuration.html#configuring-settings-file-for-repop-mode), but this guide will briefly provide some examples of how PopulationSim can be configured for this particular scenario.

First, the `run_list` in the settings file (settings.yaml) needs to be adjusted to have PopulationSim run the repop steps:
```
run_list:
  steps:
    - input_pre_processor.repop
    - repop_setup_data_structures
    - initial_seed_balancing.final=true;repop
    - integerize_final_seed_weights.repop
    - repop_balancing
    # expand_households options are append or replace
    - expand_households.repop;append
    - summarize.repop
    - write_synthetic_population.repop
    - write_tables.repop
```

Next, `repop_control_file_name: repop_controls.csv` should be added to the settings file. This tells PopulationSim which file to configure what the control totals will be within the configs directory (configs\repop_controls.csv). The following configuration will help control for characteristic of the population within a TOD area. For example, TOD is more likely to attract smaller households who are more likely to be workers, more likely to be held by younger adults, and less likely to have children than the general population.
| target     | geography | seed_table | importance | control_field | expression                                         |
|------------|-----------|------------|------------|---------------|----------------------------------------------------|
| num_hh     | mgra      | households | 1000000000 | Total_HH      | (households.WGTP > 0) & (households.WGTP < np.inf) |
| HHSize_1   | mgra      | households | 250000     | HHSize_1      | households.NP == 1                                 |
| HHSize_2   | mgra      | households | 250000     | HHSize_2      | households.NP == 2                                 |
| HHWork_0   | mgra      | households | 100000     | HHWork_0      | households.workers == 0                            |
| HHWork_1   | mgra      | households | 100000     | HHWork_1      | households.workers == 1                            |
| HHWork_2   | mgra      | households | 100000     | HHWork_2      | households.workers == 2                            |
| HHChild_0  | mgra      | households | 100000     | HHChild_0     | households.HUPAC == 4                              |
| Age_18to24 | mgra      | persons    | 100000     | Age_18to24    | (persons.AGEP >= 18) & (persons.AGEP <= 24)        |
| Age_25to34 | mgra      | persons    | 100000     | Age_25to34    | (persons.AGEP >= 25) & (persons.AGEP <= 34)        |
| Age_35to44 | mgra      | persons    | 100000     | Age_35to44    | (persons.AGEP >= 35) & (persons.AGEP <= 44)        |
| Age_45to54 | mgra      | persons    | 100000     | Age_45to54    | (persons.AGEP >= 45) & (persons.AGEP <= 54)        |

The totals in each of the zones are then defined in the control total file, which is defined in settings.yaml within the `input_table_list` setting. Within that setting, there are two subsettings: `filename` which defines the name of the file within the PopulationSim run's data directory (in this case data\repop_control_totals.csv), and `tablename`, which defines what the table will be called in memory.
```
input_table_list:
  - filename : repop_control_totals.csv
    tablename: mgra_control_data
```

These values can be set to add a population to the study area that is characteristic of a typical transit-oriented development.
| mgra | Total_HH | HHSize_1 | HHSize_2 | HHWork_0 | HHWork_1 | HHWork_2 | HHChild_0 | Age_18to24 | Age_25to34 | Age_35to44 | Age_45to54 |
|------|----------|----------|----------|----------|----------|----------|-----------|------------|------------|------------|------------|
| 579  | 500      | 250      | 150      | 50       | 240      | 120      | 400       | 150        | 200        | 200        | 150        |
| 4502 | 500      | 250      | 150      | 50       | 240      | 120      | 400       | 150        | 200        | 200        | 150        |

The output synthetic population files, output\synthetic_households.csv and output\synthetic_persons.csv, then need to be placed in the `data` directory for the ActivitySim run and renamed households.csv and persons.csv, respectively.

2. The land use file now needs to be updated to reflect the updated population. This will be demonstrated using code blocks, that have the ActivitySim input files read in as Pandas DataFrames as follows:
```
households = pd.DataFrame(r"data-full\households.csv")
persons = pd.DataFrame(r"data-full\persons.csv")
land_use = pd.DataFrame(r"data-full\land_use.csv")
```
These lines of code update the household and population values within the land use file.
```
land_use["hh"] = households.groupby("home_zone_id").count()["household_id"]
persons["home_zone_id"] = persons["household_id"].map(households.set_index("household_id")["home_zone_id"])
land_use["pop"] = persons.groupby("home_zone_id").count()["person_id"]
del persons["home_zone_id"]
```
Further variables will need to be edited as well (a full list of all variables is shown at the beginning of the section on setting up the scenario). For example, the SANDAG example contains variables on the number of housing units in each MAZ, including those that are vacant (so this will be higher than the number of households). There are also variables for population density. It may initially seem that one could just calculate them by dividing the updated population by the total area. While some models may use population density calculated in that way, the population density variables in the SANDAG model are actually the population within a buffer and are calculated via a preprocessing step that's external to ActivitySim (which should be rerun in this particular scenario). One should pay close attention to how each of the variables are defined to reduce the risk of a misunderstanding causing incorrect model results.

3. The land use data needs to again be adjusted for the retail jobs. This is overall more straightforward than adjusting the population, as ActivitySim models don't typically have a set of synthetic set of establishments. As this particular scenario add 1000 additional retail jobs to the TOD development, the values of the retail and total employment fields simply need to be updated for the zones within the study area:
```
# Adjust employment
land_use = land_use.set_index("MAZ")
land_use.loc[579, "emp_ret"] += 500
land_use.loc[579, "emp_tot"] += 500
land_use.loc[4502, "emp_ret"] += 500
land_use.loc[4502, "emp_tot"] += 500
land_use = land_use.reset_index() # Not necessary, but helpful if further operations use the MAZ field this could prevent an error
```
It should be noted that the same caveat applies to fields derived from employment data, such as employment density or any aggregated fields that may be present. One should be careful to update all fields that are relevant to the total employment.

The following lines of code then write the updated ActivitySim inputs to file (the `index = False` keyword argument prevents a superfluous index column from being written):
```
households.to_csv(r"data-full\households.csv", index = False)
persons.to_csv(r"data-full\persons.csv", index = False)
land_use.to_csv(r"data-full\land_use.csv", index = False)
```

## Running the Test

To run the test, run the following command line argument:
```
uv run activitysim run -c configs\common -c configs\resident -d data_full -o output --ext extensions
```

## Analyzing the Results

The following code blocks demonstrate how to calculate key metrics from the model outputs. They all assume that the ActivitySim output files will be read in as a data frame where the name will be the same as the file name but without the prefix or the file extension (e.g. final_trips.csv will be read as trips).

### Vehicle Miles Traveled
While the true modeled VMT requires assignment to be run, one can get a reasonable estimate via the ActivitySim outputs. The output trips table in the SANDAG ABM3 example actually includes fields called `distance` and `weightTrip`, which are created in the preprocessor for writing the outputs (write_trip_matrices_annotate_trips_preprocessor.csv). The `distance` field is created by [reading in the distance skim value](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L5) and the `weightTrip` field is a weight that [factors in the occupancy](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L7). The following lines of code compute the VMT using those particular fields:
```
auto_modes = ["DRIVEALONE", "SHARED2", "SHARED3", "TNC_SINGLE", "TNC_SHARED", "TAXI"]
auto_trips = trips[["trip_mode", "distance", "weightTrip"]].query("trip_mode in @auto_modes")
vmt = (auto_trips["distance"] * auto_trips["weightTrip"]()).sum()
```
Now, not every ActivitySim implementation will have such a field in their outputs, so the calculation may not be as simple. If the distance field isn't added to the outputs, one will need to read in the skims in order to perform the calculation. One will also need to remember to factor in the occupancy, as an individual who is carpooling has less of an impact on VMT than a person who is driving alone.

### Mode Share
Calculating the mode share of ActivitySim is fairly straightforward as the modes are reported in the output. However, one needs to ask the questions of *which* mode share they'd like to know. For example, the simplest is the regional mode share, which can just be directly calculated from the trips file:
```
mode_share = trips["trip_mode"].value_counts(normalize = True)
```
This will return the percentage of trips that use each mode. However, a single localized development won't move the needle much, so it may be hard to tell if there was an impact. The following metric computes the *tour* mode share to work of households living in zones close to the transit stop, which should show a much larger difference from the baseline (there are no households in the study area in the baseline run so the baseline mode share would be undefined).
```
station_area = [579, 4502, 8524, 7714, 12170, 12171, 5455, 8457, 846, 8232, 7831, 12172, 12173, 12174, 12175, 12176, 12177, 12178]
station_area_tours = tours[["home_maz", "tour_mode", "tour_purpose"]].query("origin in @station_area and tour_purpose == 'work'")
tour_mode_share_to_work = station_area_tours["tour_mode"].value_counts(normalize = True)
```

### Auto Ownership
The calculation of the reigional auto ownership rates is very straightforward, as that variable is reported directly in the households table:
```
auto_ownership = households["auto_ownership"].value_counts(normalize = True)
```
However, auto ownership has the same issue with mode share where the TOD development will barely move the needle on the regional auto ownership rates. Therefore, a similar calculation would need to be done:
```
station_area = [579, 4502, 8524, 7714, 12170, 12171, 5455, 8457, 846, 8232, 7831, 12172, 12173, 12174, 12175, 12176, 12177, 12178]
station_area_households = households.query("household_id in @station_area")
station_area_auto_ownership = station_area_households["auto_ownership"].value_counts(normalize = True)
```

## Summary
This guide demonstrated how to set up, run, and analyze a TOD scenario in ActivitySim. Users were shown how to configure PopulationSim in repop mode, perform QA on the updated synthetic population, run the scenario, and analyze some key metrics. Some alternate ways that scenarios could be set up were mentioned, such as adding the population through a script or changing utility coefficients to better reflect different behavior when living in such a development. Nuances on the metrics were described, such has how impacts from localized projects may not be as noticable in metrics that describe network performance at a regional level.