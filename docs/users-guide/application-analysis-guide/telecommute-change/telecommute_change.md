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