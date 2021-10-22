# SANDAG CrossBorder ActivitySim Implementation

## To run
1. Install ActivitySim from the `develop` branch of the [SANDAG fork](https://github.com/SANDAG/activitysim/tree/xborder)
2. (optional) Configure the preprocessor settings in **configs/preprocessing.yaml**
3. (optional) Run the preprocessor: `python cross_border_model.py -p`
      - Only necessary if converting CTRAMP inputs to ActivitySim format. Only needs to be run once.
4. (optional) Configure the border crossing wait time updater settings
      - ActivitySim settings (e.g. number of processes, household sample size) in **configs/wait_time_mode.yaml**
      - Preprocessor settings (e.g. number of iterations) in **configs/settings.yaml**
5. (optional) Update the border crossing wait times: `python cross_border_model.py -w`
      - If land use data does not yet have wait time columns (see below), you'll have to first run in preprocessing mode in order to generate the first set of wait times.
6. (optional) Configure the main ActivitySim settings in **configs/settings.yaml**
      - Settings you'll likely want to tweak: `household_sample_size`, `chunk_size`, `num_processes`
7. Run ActivitySim `python cross_border_model.py -a`

## Generate Estimation Data Bundles
ActivitySim models can be re-estimated from estimation data bundles, which inject survey observations (i.e. choices) into model data input tables. To create the bundles, simply set `enable: True` in **configs/estimation.yaml**, and run ActivitySim as you would normally. 


## Helpful tips:
- You can execute any or all of the above processes at once by using multiple flags: `python cross_border_model.py -p -w -a`
- Run with IPython to for easier debugging: `ipython -i cross_border_model.py -- -a` and then use `%debug` magic command if/when an error is thrown.
- Each time you run with wait time mode activated (`-w`), the wait time columns in the land use table get updated. If you run preprocessing mode at the same time (`-p -w`), the first iteration of wait times will be based on CTRAMP inputs. If you run without preprocessing mode enabled, the first iteration of wait times will be based on whatever was already in the land use file.

## Required inputs:

### ActivitySim
 - **land use** (e.g. "mazs_xborder.csv")
    - custom columns (created by preprocessor):
       - `poe_id`: port of entry (PoE) associated with each MAZ, will be null for most rows.
       - `external_TAZ`: each PoE is associated with one internal and one external MAZ/TAZ, will be null except for internal MAZs with non-null PoE IDs.
       - `original_MAZ`: the internal MAZs associated with each external MAZ/TAZ, will be null except for external MAZs.
       - `<lane_type>_wait_<period>`: e.g. "sentri\_wait\_15", an extra 192 columns of lane-period-poe-specific wait times, only populated for MAZs associated with a PoE.
 - **tours** (e.g. "tours_xborder.csv")
    - crossing type, lane type, tour type, tour category, # participants, household ID, person ID
    - hardcoded one person per tour
 - **households** (e.g. "households_xborder.csv")
   - list of successive integers from 0 to the number of tours (one household per tour)
 - **persons** (e.g. "persons_xborder.csv")
    - list of successive integers from 0 to the number of tours (one household per tour), and household_id
    - hardcoded one person per household
 - **skims**
    - traffic (e.g. "traffic_skims_xborder_\<TOD\>.omx")
    - transit (e.g. "transit_skims.omx")
    - transit access (e.g. "maz_tap_walk.csv", "taps.csv", and "tap_lines.csv")
    - microzone (e.g. "maz_maz_walk.csv")

### Preprocessor
The preprocessor is mainly designed to convert old CTRAMP-formatted input/survey data into the formats needed by ActivitySim. 
| CTRAMP input filenames | ActivitySim inputs created| ActivitySim location | Description | Preprocessor operation |
|---|---|---|---|---|
| <ul><li>mgra13_based_input2016.csv</li><li>crossBorder_supercolonia.csv</li><li>crossBorder_pointOfEntryWaitTime</li></ul>  | <ul><li>mazs_xborder.csv</ul></li>  | `data/` | land use data | Append PoE IDs, PoE wait times, and colonia population accessibility. Add external MAZ rows corresponding to the external TAZs that appear in the traffic skims|
| <ul><li>crossBorder_tourEntryAndReturn.csv</ul></li> | <ul><li>tour_scheduling_probs.csv</li><li>tour_departure_and_duration_alternatives.csv</li></ul> | `configs/` |tour scheduling probability lookup table| convert CTRAMP 40-period probabilities and alts into ActivtySim's 48-period format |
|<ul><li>microMgraEquivMinutes.csv </li></ul>| <ul><li>maz_maz_walk.csv</li></ul>| `data/` | microzone walk access skims (i.e. MAZ to MAZ)| rename columns, augment with external MAZs access|
|<ul><li>microMgraTapEquivMinutes.csv</li></ul>| <ul><li>maz_tap_walk.csv</li></ul>| `data/` |transit access skims (i.e. MAZ to TAP)| rename/remove columns, augment with external MAZ access |
| <ul><li>transit_skims.omx </li></ul>| <ul><li>transit_skims_xborder.omx</li></ul>| `data/` | transit skims (i.e. TAP to TAP)|  |
| <ul><li>traffic_skims_\<TOD\>.omx </li></ul>| <ul><li>traffic_skims_xborder_\<TOD\>.omx</li></ul> | `data/` | traffic skims (TAZ to TAZ) by time of day [EA, AM, MD, PM, EV] | reindex omx matrices for 3d lookups |
| <ul><li>crossBorder_stopFrequency.csv | <ul><li>stop_frequency_alternatives.csv</li><li>stop_frequency_coefficients_\<purpose\>.csv</li><li>stop_frequency_\<purpose\>.csv</li></ul> | `configs/` | stop frequency probability lookup | create stop frequency alts, segmented model specs, and coefficients files from CTRAMP inputs |
|<ul><li>crossBorder_stopPurpose.csv</li></ul>|<ul><li>trip_purpose_probs.csv</li></ul>| `configs/` | trip purpose probability lookup table| rename columns, delete cargo probs|
| <ul><li>crossBorder_outboundStopDuration.csv</li><li>crossBorder_inboundStopDuration.csv</li></ul>  | <li><ul>trip_scheduling_probs.csv</li></ul> | `configs/` | trip scheduling probability lookup table | combine inbound and outbound alts/probs |
|<ul><li>crossBorder_tourPurpose_control.csv</li></ul> | <ul><li>tour_purpose_probs_by_poe.csv </li></ul>| `configs/` | tour purpose reassignment probability lookup table | rename cols, drop cargo probs |

### Wait Time Updating
 - **land use** table with PoE wait time columns (e.g. "mazs_xborder.csv"). If the land use table doesn't have these columns yet, you'll have to run in preprocessing mode to generate them.
