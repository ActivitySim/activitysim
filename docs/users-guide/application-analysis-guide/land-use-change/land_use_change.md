# Land Use Change Guide

## Introduction

Many contemporary urban planners are encouraging developers to build denser housing, particularly around transit stops. Naturally, planners will want to gauge what the impact of such a development would be on their jurisdiction's transportation system, particularly regarding metrics such as VMT (and subsequently greenhouse gas emission) and transit boardings (and subsequently farebox revenue). To demonstrate this, we will be analyzing a hypothetical development in the San Diego Region. The particular development will add 2000 households and 1000 retail jobs in the vicinity of the Grossmont Station on the Green and Orange Lines of San Diego's light rail system, where there is an existing auto-oriented shopping mall. The guide will show how to make changes to the ActivitySim inputs, how to run the test, and how to calculate some of the key metrics such as VMT and changes in mode share.

**NOTE: The example provided is a hypothetical project that demonstrates how one would use ActivitySim to model the effects of a land use change and does not necessarily reflect any real planned developments.**

## Setting Up the Test

Three input files need to be changed in order to run this test: the land use file (landuse.csv) and the files defining the synthetic population (households.csv and persons.csv). While updating the land use file may seem very straightforward, it is very easy to overlook some necessary changes that could result in the model understating the impact of the change. A modeler doesn't need to just edit the household and employment fields in the study area--they also need to edit any field derived from those fields. For example, every new household will have at least one person in it, so the population field will need to be updated as well (along with population density if that is present). If the total population is to be kept the same, households will need to be removed outside of the study area as well.

Because Activity-based models use synthetic populations, those input files will need to be updated to reflect the different distribution in the population. There are multiple ways that this could be done. The ActivitySim consortium maintains the PopulationSim population synthesis software, which includes a `repop` mode that can be used to add households to an existing synthetic population. This demonstration will show how to do this, though any user is welcome to add the additional households in whatever way works best for them (such as through a script).

Before setting up this test, one should run the [SANDAG ABM3 Example model](https://github.com/activitysim/sandag-abm3-example), as the inputs used in it will be copied for this test.

### Instructions
1. Create a new directory for your test (suggested name: example-land-use). Copy the `data` and `configs` folders from your completed SANDAG ABM3 Example run into this directory. Additionally, create an empty directory called `output`.

2. Next, add the new households. This can be done using PopulationSim's repop mode. To do that, copy an existing setup to a new location. Then, edit the model steps within the `run_list` settings file to be the following steps:
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

Next, add the setting `repop_control_file_name: repop_controls.csv` to the settings file. This tells PopulationSim which file to configure what the control totals will be within the configs directory (configs\repop_controls.csv). Now, open that file and edit it to match the following table:
| target     | geography | seed_table | importance | control_field | expression                                          |
|------------|-----------|------------|------------|---------------|-----------------------------------------------------|
| num_hh     | mgra      | households | 1000000000 | Total_HH      | (households.WGTP > 0) & (households.WGTP < np.inf)  |
| HHSize_1   | mgra      | households | 250000     | HHSize_1      | households.NP == 1                                  |
| HHSize_2   | mgra      | households | 250000     | HHSize_2      | households.NP == 2                                  |
| HHWork_0   | mgra      | households | 100000     | HHWork_0      | households.workers == 0                             |
| HHWork_1   | mgra      | households | 100000     | HHWork_1      | households.workers == 1                             |
| HHWork_2   | mgra      | households | 100000     | HHWork_2      | households.workers == 2                             |
| HHChild_0  | mgra      | households | 100000     | HHChild_0     | households.HUPAC == 4                               |
| Age_18to24 | mgra      | persons    | 100000     | Age_18to24    | (persons.AGEP >= 18) & (persons.AGEP <= 24)         |
| Age_25to34 | mgra      | persons    | 100000     | Age_25to34    | (persons.AGEP >= 25) & (persons.AGEP <= 34)         |
| Age_35to44 | mgra      | persons    | 100000     | Age_35to44    | (persons.AGEP >= 35) & (persons.AGEP <= 44)         |
| Age_45to54 | mgra      | persons    | 100000     | Age_45to54    | (persons.AGEP >= 45) & (persons.AGEP <= 54)         |

Many of the fields that are controlled for can be defined to be characteristic of the population within a TOD area. For example, TOD is more likely to attract smaller households who are more likely to be workers, more likely to be held by younger adults, and less likely to have children than the general population. This can be defined in the control totals table, which resides within the data folder. The exact name of the file is defined as the `mgra_control_data` within the `input_table_list` setting:
```
input_table_list:
  - filename : repop_control_totals.csv
    tablename: mgra_control_data
```

Setting that CSV file to the following values should result in the addition of 1000 households that are characteristic of a TOD in the San Diego area:
| mgra | Total_HH | HHSize_1 | HHSize_2 | HHWork_0 | HHWork_1 | HHWork_2 | HHChild_0 | Age_18to24 | Age_25to34 | Age_35to44 | Age_45to54 |
|------|----------|----------|----------|----------|----------|----------|-----------|------------|------------|------------|------------|
| 1    | 250      | 125      | 75       | 25       | 120      | 60       | 200       | 75         | 100        | 100        | 75         |
| 2    | 250      | 125      | 75       | 25       | 120      | 60       | 200       | 75         | 100        | 100        | 75         |
| 3    | 250      | 125      | 75       | 25       | 120      | 60       | 200       | 75         | 100        | 100        | 75         |
| 4    | 250      | 125      | 75       | 25       | 120      | 60       | 200       | 75         | 100        | 100        | 75         |

3. The land use data needs to be readjusted to increase the number of households within the study area.
```
land_use["hh"] = households.groupby("home_zone_id").count()["household_id"]
persons["home_zone_id"] = persons["household_id"].map(households.set_index("household_id")["home_zone_id"])
land_use["pop"] = persons.groupby("home_zone_id").count()["person_id"]
del persons["home_zone_id"]
```

4. The land use data needs to again be adjusted for the retail jobs. This example will show those jobs being taken from other areas, though if the total number of jobs don't need to stay constant, no values outside the study area would need to change.
```
# Adjust employment
outside_mask = ~land_use["MAZ"].isin(STUDY_AREA_ZONES)
inside_mask = land_use["MAZ"].isin(STUDY_AREA_ZONES)

# Randomly select which outside zones lose retail jobs (weighted by current emp_ret)
outside = land_use.loc[outside_mask].copy()
outside_ret = outside.loc[outside["emp_ret"] > 0]
weights = outside_ret["emp_ret"] / outside_ret["emp_ret"].sum()
jobs_to_remove = pd.Series(
    np.random.choice(outside_ret.index, size=N_NEW_RETJOBS, p=weights),
).value_counts()
land_use.loc[jobs_to_remove.index, "emp_ret"] -= jobs_to_remove.values

# Randomly distribute those jobs into study area zones
jobs_to_add = pd.Series(
    np.random.choice(land_use.loc[inside_mask].index, size=N_NEW_RETJOBS),
).value_counts()
land_use.loc[jobs_to_add.index, "emp_ret"] += jobs_to_add.values
```

## Running the Test
To run the test, run the following command line argument:
```

```

## Analyzing the Results