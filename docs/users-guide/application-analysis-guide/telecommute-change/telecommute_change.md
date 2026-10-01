# Telecommuting Change

Telecommuting has been on planners' minds since the start of the COVID-19 Pandemic in 2020, as the rapid increase in telecommuting at the onset of the pandemic resulted in drastic reductions in VMT and transit boardings, the latter of which hasn't recovered in some places. At the present many employers are pushing for their employees to return to the office, and thus planners would like to test the impacts of such on the transportation network. This guide will demonstrate how to set up an increased return-to-office scenario in ActivitySim.

## Setting up the Scenario

ActivitySim has a telecommute frequency model, which is run fairly early on in the model stream as whether or not one telecommutes has a major impact on their travel behavior in a given day. If one were to look at the telecommute frequency results for the SANDAG ABM3 Example model (which was estimated and calibrated using a combination of 2016 and 2022 data), they would see a share along the lines of this (note that 5 days/week is not an option as those workers are determined in the work from home model):

| Alternative               | Share of Workers |
| ------------------------- | ---------------- |
| No Telecommuting          |            75.8% |
| Telecommuting 1 Day/Week  |             4.9% |
| Telecommuting 2 Days/Week |             5.9% |
| Telecommuting 3 Days/Week |             5.9% |
| Telecommuting 4 Days/Week |             7.4% |

Because the output of the share, the model outputs would need to be calibrated. The telecommute frequency in the SANDAG ABM3 Model has three calibration coefficients: one for telecommuting 1 day per week, one for telecommuting 2-3 days per week, and one for telecommuting 4 days per week (These values are set in the file configs\resident\telecommute_frequency_coeffs.csv). This means that the 2 days per week and 3 days per week categories need to be added together when calibrating:

| Alternative                 | Share of Workers |
| --------------------------- | ---------------- |
| No Telecommuting            |            75.8% |
| Telecommuting 1 Day/Week    |             4.9% |
| Telecommuting 2-3 Days/Week |            11.8% |
| Telecommuting 4 Days/Week   |             7.4% |

The guide will calibrate the telecommute frequencies to the following shares:
| Alternative                 | Share of Workers |
| --------------------------- | ---------------- |
| No Telecommuting            |            90.0% |
| Telecommuting 1 Day/Week    |             7.0% |
| Telecommuting 2-3 Days/Week |             2.0% |
| Telecommuting 4 Days/Week   |             1.0% |

The coefficients then need to be adjusted based on the natural logarithm of the ratio of the target share to the modeled share. It should be noted that the shares won't match at first, so the process needs to be run iteratively. The next iteration's coefficient $$c_{n+1}$$ on a given iteration $$n$$ is calculated as follows:

$$c_{n+1}=c_{n}+\alpha\ln{(\frac{\hat{s}}{s_{n}})}$$

Where $$\hat{s}$$ is the target share, $$s_{n}$$ is the share for calibration iteration $$n$$, and $$\alpha$$ is an optional factor to control how quickly convergence is reached (in this example it will be set to 1). This means that the coefficients for the first calibration iteration would be as follows:

| Alternative                 | Target Share | Modeled Share | Adjustment | Old Coefficient | New Coefficient |
| --------------------------- | ------------ | ------------- | ---------- | --------------- | --------------- |
| No Telecommuting            |        90.0% |         75.8% |            |                 |             |
| Telecommuting 1 Day/Week    |         7.0% |          4.9% |      0.357 |          -2.549 |          -2.192 |
| Telecommuting 2-3 Days/Week |         2.0% |         11.8% |     -1.775 |          -1.534 |          -3.309 |
| Telecommuting 4 Days/Week   |         1.0% |          7.4% |     -2.001 |          -1.948 |          -3.949 |

The user would then need to run ActivitySim with the following values set in configs\resident\telecommute_frequency_coeffs.csv:

| coefficient_name | value  | constrain |
| ---------------- | ------ | --------- |
| asc_1day         | -2.192 | F         |
| asc_23day        | -3.309 | F         |
| asc_4day         | -3.949 | F         |

As the telecommute frequency model is relatively early in the model stream, one can save time while calibrating by skipping the remaining model steps. This can be done by commenting out all everything from `cdap_simulate` to `parking_location` along with `write_trip_matrices` in the `models` list in configs\resident\settings_mp.yaml along with commenting out the blocks defining the output `tours` and `trips` tables in the `output_tables` section of configs\resident\settings.yaml. To speed up each calibration even further, one can set `resume_after: free_parking` in settings_mp.yaml to skip all of the prior steps and read them in from the saved pipeline (assuming `cleanup_pipeline_after_run` is set to be `False`).

After enough calibration iterations have been run so that the model shares are sufficiently close enough to the target shares, the user can run the full model to estimate the impact of increased returning to the office on the transportation system and people's overall travel behavior.

## Running the Test

To run the test, run the following command line argument:
```
uv run activitysim run -c configs\common -c configs\resident -d data_full -o output --ext extensions
```

## Analyzing the Results

The following code blocks demonstrate how to calculate key metrics from the model outputs. They all assume that the ActivitySim output files will be read in as a data frame where the name will be the same as the file name but without the prefix or the file extension (e.g. final_trips.csv will be read as trips).

### Daily Activity Pattern
One model result that one would expect to change from a decrease in teleworking is the daily activity pattern. One would expect the use of the "Mandatory" activity pattern to increase and the "Nonmandatory" and "Home" patterns to decrease. For both the baseline and the test runs, one can calculate the share of people choosing each day pattern with the following block of code:
```
dap_share = persons["cdap_activity"].value_counts(normalize = True)
```
However, one may want to only look at the workers, as the activity pattern of nonworkers should be generally the same between the two scenarios (though intra-household interactions may change the patterns of some of the non-workers):
```
worker_dap_share = persons.query("is_worker")["cdap_activity"].value_counts(normalize = True)
```

### Trips by Purpose for Workers
A change in telecommute frequency would likely result in a change in the number of tours by purpose for workers. If a worker is teleworking, that gives them more flexibility in their ability to make nonmandatory travel, so one could expect trips within those purposes to increase when comparing to a baseline run. The following calculates the number of trips by purpose for workers:
```
trips["is_worker"] = persons.set_index("person_id")["is_worker"].reindex(trips["person_id"])
trips_by_workers = trips.query("is_worker")
worker_trips_by_purpose = trips_by_workers["purpose"].value_counts()
```

### Time of Day Distribution
Decreased telecommuting should have a strong impact on the time of day distribution. It was observed in multiple places that the AM travel peak effectively disappeared in the years immediately following the onset of the COVID-19 pandemic, so one could reasonably guess that 9-5 workers largely returning to the office would result in that peak reemerging. The time of the trip is stored in the `depart` field of the trips file and is coded in a half-hour bin starting at 3 AM, with time period 1 being 3-3:30 AM, time period 2 being 3:30-4 AM, and so forth. One can group the number of trips by time period and sort them in order. Comparing the `trips_by_time_period` series between the baseline run and the test run will allow the analyst to see if the AM peak returned.
```
trips_by_time_period = trips["depart"].value_counts(normalize = True).sort_index()
```

### Average Distance to Work
The work location model is run before the telecommute frequency, so one should expect it should not change. However, one may want to check its results to ensure that it doesn't change. The following code block calculates that:
```
avg_dist_to_work = persons["distance_to_work"].mean()
```

### Vehicle Miles Traveled
While the true modeled VMT requires assignment to be run, one can get a reasonable estimate via the ActivitySim outputs. The output trips table in the SANDAG ABM3 example actually includes fields called `distance` and `weightTrip`, which are created in the preprocessor for writing the outputs (write_trip_matrices_annotate_trips_preprocessor.csv). The `distance` field is created by [reading in the distance skim value](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L5) and the `weightTrip` field is a weight that [factors in the occupancy](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L7). The following lines of code compute the VMT using those particular fields:
```
auto_modes = ["DRIVEALONE", "SHARED2", "SHARED3", "TNC_SINGLE", "TNC_SHARED", "TAXI"]
auto_trips = trips[["trip_mode", "distance", "weightTrip"]].query("trip_mode in @auto_modes")
vmt = (auto_trips["distance"] * auto_trips["weightTrip"]()).sum()
```

### Number of Transit Trips
As previously mentioned, teleworking has generally seen a decrease in transit usage. Analysts may want to estimate the impact of return to office on the number of transit trips for purposes such as revenue forecasting. The following code block demonstrates how to calculate the number of transit trips for a given run. One can compare the value of the `transit_trips` series between the test run and a baseline run to get an estimate in the increase, though results from transit assignment would be needed for more detailed calculation, such as boardings on specific lines.
```
transit_modes = [
  "WALK_LOC", "WALK_PRM", "WALK_MIX",
  "PNR_LOC", "PNR_PRM", "PNR_MIX",
  "KNR_LOC", "KNR_PRM", "KNR_MIX",
  "TNC_LOC", "TNC_PRM", "TNC_MIX"
]
transit_trips = len(trips.query("trip_mode in @transit_modes"))
```

## Summary

The impacts of returning to the office are something that is on the mind of many transportation planners as we move into the late 2020s. Many employers want to have their employees work from an office more which will result in an increase in travel to work. This will result in an increased strain on the transportation network that needs to be planned for. ActivitySim is capable of modeling an decrease in telecommuting with its Telecommute Frequency model component, though it takes more effort than just simply changing a few model inputs here and there. Because how often people telecommute is an output of that model, it must be calibrated to match the desired telecommuting shares, which will involve iteratively running the component and adjusting calibration coefficients until the results are sufficiently close to the desired targets. After that is done, the full modeling system can be run to estimate the impact of people telecommuting fewer days, which based on the structure of the model will impact things such as the daily activity pattern and number of trips taken by transit, but not where people choose to work as that model component is typically run earlier.