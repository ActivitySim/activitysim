# Network Change

The most influential policy choice that transportation planners can make is changing the network itself. Evaluating the impact of building additional roadways were why travel demand models were initially developed in the first place. As transportation planning has expanded to be reflective of other modes, models of people's daily travel behavior needed to be adjusted so they could model the affects of changes to the transit or nonmotorized networks as well. ActivitySim itself does not directly model network changes at the present. In fact, ActivitySim doesn't actually take network files as inputs. This doesn't mean that ActivitySim isn't sensitive to network changes. ActivitySim models the demand for travel, so it uses the skim matrices as those contain the information that travelers use when making their decisions. In most cases travelers aren't concerned about the specific route when deciding if and where to travel. They're more concerned with things such as how long it will take them to get there or what the cost will be, and this information is stored in the skim matrices.

This guide will demonstrate how to model and analyze the addition of a BRT line using ActivitySim. The line will run from the San Diego suburb of La Mesa to the neighborhood of Ocean Beach via the Hillcrest neighborhood. Please note that the network change itself needs to be made in whichever assignment software that is being used, which will not be discussed in this guide as many ActivitySim users use different assignment software, and network editing needs to be done differently in each. SANDAG uses custom software that was developed by ESRI for editing and managing their model networks.

## Setting up the Scenario

1. In the software system that manages the model networks, add the BRT line to the transit network.

2. Run transit assignment and skimming with the updated network.

3. Move the updated transit skim files into the `data` folder for the ActivitySim run.

If one were to make a change to the roadway network instead of the transit network, they would need to run highway assignment and skimming instead of transit assignment and skimming. Further, for any network change that is more temporary (such as a closure of a bridge or transit line), it is possible to freeze the results of the longer-term choice models and only model how people will change their shorter-term behavior. To do so, one must first have a completed run for the desired scenario year, and save all of the pipeline results (`cleanup_pipeline_after_run` must be set to `False` in the ActivitySim settings file). Then, in the settings file, set `resume_after` to be the last model step that you wish to freeze. In the SANDAG ABM3 example, if one were to want to only run the short-term choices, they would set `resume_after: telecommute_frequency` in settings_mp.yaml which would only run the model steps from `cdap_simulate` on, reading the upstream results from the saved pipeline.

## Running the Test

To run the test, run the following command line argument:
```
uv run activitysim run -c configs\common -c configs\resident -d data_full -o output --ext extensions
```

## Analyzing the Results

The following code blocks demonstrate how to calculate key metrics from the model outputs. They all assume that the ActivitySim output files will be read in as a data frame where the name will be the same as the file name but without the prefix or the file extension (e.g. final_trips.csv will be read as trips).

### Mode Share by Purpose

The following code calculates the tour mode share by tour purpose:
```
tour_mode_share_by_purpose = tours[["tour_mode", "primary_purpose"]].value_counts().reset_index().pivot(index = "tour_mode", columns = "primary_purpose", values = "count").fillna(0)
for col in tour_mode_share_by_purpose.columns:
    tour_mode_share_by_purpose[col] /= tour_mode_share_by_purpose[col].sum()
```
To see a stronger contrast with the baseline scenario, one can filter the tours if the origin and/or destination is along the BRT line's corridor. Which specific zones are along the corridor would be determined via a GIS exercise.

### Share of Trips by Time Period

ActivitySim codes the time of day with 48 half-hour periods starting at 3 AM. This means that period 1 is from 3-3:30 AM, period 2 is 3:30-4 AM, and so forth. Many planners are interested in measuring how many people shift the time of their travel from peak periods to off-peak periods (known as peak spreading), which puts less strain on the transportation system as a whole. The following code creates a field that categorizes trips into categories for each hour of the AM peak or the rest of the day based on the departure period, and then groups the number of trips by that category. By comparing the results of the test run to the baseline run, the analyst can directly caluclate how many trips moved away from the AM peak.
```
trips["AM Peak Hour"] = np.where(
    (trips["depart"] == 7) | (trips["depart"] == 8),
    "6-7 AM",
    np.where(
        (trips["depart"] == 9) | (trips["depart"] == 10),
        "7-8 AM",
        np.where(
            (trips["depart"] == 11) | (trips["depart"] == 12),
            "8-9 AM",
            "Rest of Day"
        )
    )
)
trips_by_am_peak_hour = trips["AM Peak Hour"].value_counts()
```

### Congestion Impact
There are many ways one could measure the effects of traffic congestion. Most of these are directly the result of highway assignment, of which ActivitySim does not presently have a model component for. For example, ActivitySim does not report any network-level results, so one would not be able to calculate link volumes directly from ActivitySim outputs.

### Vehicle Miles Traveled
While the true modeled VMT requires assignment to be run, one can get a reasonable estimate via the ActivitySim outputs. The output trips table in the SANDAG ABM3 example actually includes fields called `distance` and `weightTrip`, which are created in the preprocessor for writing the outputs (write_trip_matrices_annotate_trips_preprocessor.csv). The `distance` field is created by [reading in the distance skim value](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L5) and the `weightTrip` field is a weight that [factors in the occupancy](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/write_trip_matrices_annotate_trips_preprocessor.csv#L7). The following lines of code compute the VMT using those particular fields:
```
auto_modes = ["DRIVEALONE", "SHARED2", "SHARED3", "TNC_SINGLE", "TNC_SHARED", "TAXI"]
auto_trips = trips[["trip_mode", "distance", "weightTrip"]].query("trip_mode in @auto_modes")
vmt = (auto_trips["distance"] * auto_trips["weightTrip"]()).sum()
```
Now, not every ActivitySim implementation will have such a field in their outputs, so the calculation may not be as simple. If the distance field isn't added to the outputs, one will need to read in the skims in order to perform the calculation. One will also need to remember to factor in the occupancy, as an individual who is carpooling has less of an impact on VMT than a person who is driving alone.

Further, if one wants to normalize the VMT by capita, they simply need to divide the VMT value by the number of persons:
```
vmt_per_capita = vmt / len(persons)
```

## Summary
Network changes are the most important and noticible transportation policy change that planning agencies will make. While ActivitySim does not directly model how people will adjust the route changes they make in response to the network change, it can model how they adjust their overall travel behavior such as whether or not transit use increases if a transit line serving their needs is added to the network. Setting up the scenario doesn't require much within ActivitySim configuration other than updating the appropriate skim files. However, those changes in the skim files could have a significant change on the outputs, many relevant summaries of which were demonstrated here.