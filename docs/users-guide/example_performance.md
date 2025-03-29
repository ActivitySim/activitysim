(example-performance)=
# Example Model Performance Benchmarking

This page provides performance metrics for the MTC and SANDAG example models,
run during spring 2024 using a pre-release version of ActivitySim 1.3.
Runtimes are reported in minutes and are broken down by model component.

Note that performance is highly dependent on the hardware on which the model
is run, and your results may vary.

## MTC Example

Full-scale testing of the canonical 1-zone MTC example model was conducted
on a full-scale dataset, containing 1454 TAZs, and a 100% sample of households
in the MTC (San Francisco) modeling region.

### Multiprocessing, Sharrow Enabled

The following performance metrics were collected for the MTC example model,
running on a Windows server with 24 cores and 500GB of RAM, using the following
configuration:

- `sharrow: require`
- `multiprocess: True`

See more complete results and discussion
[here](https://github.com/ActivitySim/activitysim-prototype-mtc/issues/12).


| model name                         | 4 cores | 8 cores | 12 cores | 20 cores |
|------------------------------------|---------|---------|----------|----------|
| mp_setup_skims                     | 1       | 1       | 1        | 1        |
| input_checker                      | 0.4     | 0.4     | 0.4      | 0.4      |
| initialize_proto_population        | 0.1     | 1       | 0.1      | 0.1      |
| mp_da_apportion                    | 0.3     | 0.2     | 0.2      | 0.2      |
| compute_disaggregate_accessibility | 2.5     | 1.6     | 1.2      | 0.9      |
| mp_da_coalesce                     | 0.2     | 0.4     | 0.2      | 0.3      |
| initialize landuse                 | 0       | 0       | 0        | 0        |
| initialize households              | 3.2     | 3.2     | 3.1      | 3.2      |
| mp_accessibility_apportion         | 1       | 1.8     | 2.4      | 4        |
| compute_accessibility              | 0.3     | 0.3     | 0.2      | 0.3      |
| mp_accessibility_coalesce          | 0.6     | 0.6     | 0.6      | 0.6      |
| mp_simulate_apportion              | 0.7     | 0.8     | 0.9      | 1.2      |
| school location                    | 8       | 5.2     | 4.8      | 5.3      |
| workplace_location                 | 16.2    | 8.9     | 7.3      | 6.8      |
| auto_ownership_simulate            | 0.3     | 0.2     | 0.1      | 0.1      |
| vehicle_type_choice                | 3.2     | 1.6     | 1.3      | 1.2      |
| free_parking                       | 0.4     | 0.2     | 0.2      | 0.1      |
| cdap_simulate                      | 1.2     | 0.7     | 0.5      | 0.5      |
| mandatory_tour_frequency           | 0.6     | 0.3     | 0.2      | 0.2      |
| mandatory_tour_scheduling          | 20.9    | 10.7    | 8.4      | 7.4      |
| school_escorting                   | 18.9    | 12.6    | 11.8     | 14.2     |
| joint_tour_frequency               | 0.3     | 0.2     | 0.1      | 0.1      |
| joint_tour_composition             | 0.1     | 0       | 0        | 0        |
| joint_tour_participation           | 0.7     | 0.4     | 0.3      | 0.3      |
| joint_tour_destination             | 1       | 0.5     | 0.4      | 64       |
| joint_tour_scheduling              | 0.3     | 0.2     | 0.2      | 0.1      |
| non_mandatory_tour_frequency       | 2.8     | 1.4     | 1.1      | 1        |
| non_mandatory_tour_destination     | 13.5    | 6.8     | 5.3      | 4.8      |
| non_mandatory_tour_scheduling      | 9.8     | 4.9     | 3.9      | 3.6      |
| vehicle allocation                 | 11.1    | 7.4     | 6.1      | 7.6      |
| tour mode choice simulate          | 3       | 1.5     | 1.2      | 1        |
| atwork_subtour_frequency           | 0.4     | 0.2     | 0.2      | 0.1      |
| atwork subtour destination         | 2.6     | 1.3     | 1        | 0.9      |
| atwork_subtour_scheduling          | 0.6     | 0.3     | 0.2      | 0.2      |
| atwork subtour mode choice         | 0.4     | 0.2     | 0.2      | 0.2      |
| stop_frequency                     | 5.7     | 2.8     | 2        | 1.4      |
| trip_purpose                       | 0.9     | 0.4     | 0.3      | 0.3      |
| trip_destination                   | 55.5    | 26      | 19.6     | 16.9     |
| trip_purpose_and_destination       | 4.9     | 4.1     | 4.9      | 6.7      |
| trip_scheduling                    | 13.2    | 6.6     | 5.1      | 4.2      |
| trip_mode_choice                   | 6       | 3       | 2.4      | 2        |
| mp_simulate_coalesce               | 151     | 14.7    | 14.7     | 14.7     |
| write_data_dictionary              | 13.5    | 13.3    | 13.3     | 13.3     |
| write_trip_matrices                | 5.7     | 5.6     | 5.6      | 6        |
| write tables                       | 3.3     | 3.2     | 3.2      | 364      |
| Runtime Total                      | 253.8   | 160.6   | 141.4    | 144      |

### Multiprocessing, Sharrow Disabled

The following performance metrics were collected for the MTC example model,
running on a Windows server with 24 cores and 500GB of RAM, using the following
configuration:

- `sharrow: false`
- `multiprocess: True`

See more complete results and discussion
[here](https://github.com/ActivitySim/activitysim-prototype-mtc/issues/13).


| model name                         | 4 cores | 8cores | 12cores | 16 cores | 20 cores | 24 cores |
|------------------------------------|---------|--------|---------|----------|----------|----------|
| mp_setup_skims                     | 1.3     | 1.2    | 1.1     | 161      | 1.2      | 1.2      |
| input_checker                      | 0.4     | 0.4    | 0.3     | 0.3      | 0.4      | 0.4      |
| initialize_proto_population        | 0.1     | 0.1    | 0.1     | 0.1      | 0.1      | 0.1      |
| mp_da_apportion                    | 0.2     | 0.2    | 0.2     | 0.2      | 0.2      | 0.2      |
| compute_disaßdregate_accessibility | 8.3     | 4.6    | 3.2     | 2.7      | 2.5      | 2        |
| mp_da_coalesce                     | 0.2     | 0.2    | 0.2     | 0.2      | 0.3      | 0.3      |
| initialize landuse                 | 0       | 0      | 0       | 0        | 0        | 0        |
| initialize households              | 3       | 3.1    | 2.9     | 2.9      | 3.1      | 3.2      |
| mp_accessibility_apportion         | 1       | 1.8    | 2.5     | 3.1      | 3.9      | 4.7      |
| compute_accessibility              | 0.3     | 0.3    | 0.2     | 0.2      | 0.3      | 0.3      |
| mp_accessibility_coalesce          | 0.6     | 0.6    | 0.5     | 0.5      | 0.6      | 0.7      |
| mp_simulate_apportion              | 0.6     | 0.8    | 0.8     | 1        | 1.1      | 1.2      |
| school location                    | 24.3    | 14.8   | 11.3    | 10.2     | 9.8      | 11.3     |
| workplace_location                 | 59      | 33.7   | 24.5    | 21.4     | 19.3     | 19.4     |
| auto_ownership_simulate            | 0.3     | 0.2    | 0.1     | 0.1      | 0.1      | 81       |
| vehicle_type_choice                | 34.1    | 19.2   | 13.6    | 11.3     | 1061     | 9.9      |
| free_parking                       | 0.4     | 0.2    | 0.2     | 0.1      | 0.1      | 0.1      |
| cdap_simulate                      | 1.3     | 0.7    | 0.6     | 0.5      | 0.5      | 0.6      |
| mandatory_tour_frequency           | 0.6     | 0.3    | 0.3     | 0.2      | 0.2      | 0.2      |
| mandatory_tour_scheduling          | 44.7    | 24.8   | 18.2    | 15.8     | 14.6     | 13.7     |
| school_escorting                   | 22.6    | 11.6   | 8.6     | 72       | 664      | 6.2      |
| joint_tour_frequency               | 0.3     | 0.2    | 0.1     | 0.1      | 0.1      | 0.1      |
| joint_tour_composition             | 0.1     | 0      | 0       | 0        | 0        | 0        |
| joint_tour_participation           | 0.7     | 0.4    | 0.3     | 0.3      | 0.3      | 0.3      |
| joint_tour_destination             | 2.5     | 1.2    | 0.9     | 0.8      | 0.8      | 0.7      |
| joint_tour_scheduling              | 0.4     | 0.3    | 0.2     | 0.2      | 0.2      | 0.4      |
| non_mandatory_tour_frequency       | 12.3    | 6.4    | 4.8     | 4.1      | 3.6      | 3.4      |
| non_mandatory_tour_destination     | 53      | 27.5   | 20.7    | 17.5     | 15.9     | 14.4     |
| non_mandatory_tour_scheduling      | 13.5    | 7.1    | 5.4     | 4.7      | 4.4      | 461      |
| vehicle allocation                 | 6.5     | 3.2    | 2.4     | 2.1      | 1.9      | 1.8      |
| tour mode choice simulate          | 5.3     | 2.7    | 22      | 1.8      | 1.7      | 1.7      |
| atwork_subtour_frequency           | 0.4     | 0.2    | 0.2     | 0.1      | 0.1      | 0.1      |
| atwork subtour destination         | 9.4     | 5.1    | 3.9     | 3.3      | 2.9      | 2.6      |
| atwork_subtour_scheduling          | 0.9     | 0.5    | 0.4     | 0.4      | 0.3      | 0.4      |
| atwork subtour mode choice         | 0.6     | 0.3    | 0.2     | 0.2      | 0.2      | 0.2      |
| stop_frequency                     | 5.4     | 2.8    | 2       | 1.6      | 164      | 1.3      |
| tri p_purpose                      | 0.9     | 0.5    | 0.3     | 0.3      | 0.3      | 0.2      |
| trip_destination                   | 163.9   | 86.5   | 64.6    | 55.1     | 49.8     | 45.3     |
| trip_purpose_and_destination       | 0.5     | 0.3    | 0.2     | 0.2      | 0.2      | 64       |
| trip_scheduling                    | 13.1    | 6.8    | 5.1     | 4.6      | 4        | 3.6      |
| trip_mode_choice                   | 11.9    | 6      | 4.3     | 3.9      | 3.4      | 3.2      |
| mp_simulate_coalesce               | 15.1    | 14.6   | 14.6    | 14.8     | 14.7     | 14.7     |
| write_data_dictionary              | 13.5    | 13.2   | 13.2    | 13.3     | 13.3     | 13.3     |
| write_trip_matrices                | 5.6     | 5.2    | 5.2     | 5.9      | 5.8      | 5.6      |
| write tables                       | 2.5     | 2.5    | 2.5     | 2.6      | 2.6      | 2.6      |
| Runtime Total                      | 546.2   | 318.9  | 248     | 228      | 209.3    | 206.7    |

## SANDAG Example

Full-scale testing of the canonical 2-zone SANDAG example model was conducted
on a full-scale dataset, with 24333 MAZs, and a 100% sample of households
representing the entire SANDAG modeling region.

### Multiprocessing, Sharrow Enabled

The following performance metrics were collected for the SANDAG example model,
running on a Windows server with 24 cores and 500GB of RAM, using the following
configuration:

- `sharrow: require`
- `multiprocess: True`

See more complete results and discussion
[here](https://github.com/ActivitySim/sandag-abm3-example/issues/22).


| model name                            | 4 cores | 8 cores | 12 cores | 16 cores | 20 cores | 24 cores |
|---------------------------------------|---------|---------|----------|----------|----------|----------|
| mp_setup_skims                        | 10.7    | 10.8    | 11.3     | 10.9     | 10.4     | 10.3     |
| initialize_proto_population           | 0.5     | 0.5     | 0.5      | 0.5      | 0.5      | 0.5      |
| mp_da_apportion                       | 0.2     | 0.2     | 0.3      | 0.3      | 0.3      | 0.3      |
| compute_disaggregate_accessibility    | 7       | 4.5     | 4        | 3.5      | 2.9      | 2.7      |
| mp_da_coalesce                        | 0.3     | 0.4     | 0.5      | 0.3      | 0.3      | 0.3      |
| initialize landuse                    | 0.1     | 0.1     | 0.1      | 0.1      | 0.1      | 0.1      |
| initialize households                 | 5.2     | 5.2     | 5.4      | 5.4      | 5.1      | 5.1      |
| mp_accessibility_apportion            | 1.2     | 1.9     | 2.9      | 3.7      | 4.5      | 5.4      |
| comp ute_accessibility                | 4       | 2.4     | 1.9      | 1.6      | 1.3      | 1.2      |
| mp_accessibility_coalesce             | 0.6     | 0.6     | 0.6      | 0.6      | 0.6      | 0.6      |
| mp_households_ap portion              | 0.7     | 1.1     | 1.2      | 1.5      | 1.6      | 1.9      |
| av_ownership                          | 0.3     | 0.2     | 0.2      | 0.2      | 0.1      | 0.1      |
| auto_ownership_simulate               | 0.4     | 0.2     | 0.2      | 0.1      | 0.1      | 0.1      |
| work from home                        | 0.2     | 0.1     | 0.1      | 0.1      | 0        | 0        |
| external worker identification        | 2.3     | 2.2     | 2.2      | 2.1      | 1.8      | 1.7      |
| external_workplace_location           | 1.1     | 1.7     | 2.4      | 3        | 3.8      | 4.7      |
| school location                       | 12.1    | 7.5     | 6.8      | 6.6      | 6.6      | 6.9      |
| workplace_location                    | 15.1    | 8.1     | 6.5      | 6        | 5.8      | 5.7      |
| transit_pass_subsidy                  | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| transit_pass_ownership                | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| vehicle_type_choice                   | 2.4     | 1.3     | 1        | 1        | 1        | 0.9      |
| adjust_auto_operating_cost            | 0.2     | 0.2     | 0.2      | 0.2      | 0.2      | 0.2      |
| transponder_ownership                 | 0.3     | 0.2     | 0.2      | 0.2      | 0.2      | 0.2      |
| free_parking                          | 0.4     | 0.3     | 0.3      | 0.2      | 0.3      | 0.3      |
| telecommute_frequency                 | 0.4     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| cdap_simulate                         | 1.1     | 0.8     | 0.7      | 0.8      | 0.8      | 0.9      |
| mandatory_tour_frequency              | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| mandatory_tour_scheduling             | 20.5    | 11.1    | 8.3      | 7.8      | 7.3      | 7.7      |
| school_escorting                      | 5.9     | 3.1     | 2.4      | 2.2      | 2.1      | 1.9      |
| joint_tour_frequency_composition      | 0.6     | 0.4     | 0.3      | 0.3      | 0.3      | 0.3      |
| external_joint_tour_identification    | 1.5     | 1.2     | 1.1      | 1.2      | 1.2      | 1        |
| joint_tour_participation              | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| joint_tour_destination                | 2.6     | 1.5     | 1.2      | 1.2      | 1.2      | 1.2      |
| external_joint_tour_destination       | 0.4     | 0.4     | 0.4      | 0.5      | 0.5      | 0.5      |
| joint_tour_scheduling                 | 1.4     | 0.8     | 0.7      | 0.6      | 0.7      | 0.8      |
| non_mandatory_tour_frequency          | 1.4     | 0.8     | 0.6      | 0.6      | 0.6      | 0.6      |
| external_non_mandatory_identification | 2.6     | 2.2     | 2.2      | 2.5      | 2.8      | 2.9      |
| non_mandatory_tour_destination        | 18.9    | 10.2    | 7.8      | 7.3      | 7        | 6.4      |
| external_non_mandatory_destination    | 0.6     | 0.5     | 0.5      | 0.5      | 0.6      | 0.7      |
| non_mandatory_tour_scheduling         | 26.5    | 15.1    | 11.6     | 11.6     | 11.4     | 11.2     |
| vehicle allocation                    | 5.9     | 4.3     | 3.8      | 4.2      | 4.3      | 4.7      |
| tour mode choice simulate             | 2.9     | 1.7     | 1.4      | 1.5      | 1.4      | 1.5      |
| atwork subtour_frequency              | 0.4     | 0.3     | 0.2      | 0.3      | 0.2      | 0.3      |
| atwork subtour destination            | 3.3     | 1.9     | 1.4      | 1.4      | 1.3      | 1.2      |
| atwork subtour_scheduling             | 1       | 0.6     | 0.5      | 0.5      | 0.5      | 0.5      |
| atwork subtour mode choice            | 0.5     | 0.4     | 0.4      | 0.4      | 0.4      | 0.4      |
| stop_frequency                        | 3       | 1.6     | 1.2      | 1.1      | 0.9      | 0.9      |
| trip_purpose                          | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| trip_destination                      | 85.2    | 46      | 33.8     | 32       | 28       | 26.7     |
| trip_purpose_and_desnnat•on           | 0.3     | 0.6     | 0.6      | 0.4      | 0.3      | 0.4      |
| trip_scheduling                       | 0.6     | 0.5     | 0.4      | 0.4      | 0.4      | 0.4      |
| trip_mode_choice                      | 5.5     | 3.2     | 2.5      | 2.3      | 2.1      | 2.1      |
| parking_location                      | 2.8     | 1.6     | 1.2      | 1.1      | 1.1      | 1        |
| mp_households_coalesce                | 7.7     | 7.8     | 7.8      | 7.6      | 7.6      | 7.7      |
| write_data_dictionary                 | 6.6     | 6.7     | 6.6      | 6.6      | 6.5      | 6.6      |
| track_skim_usage                      | 0.4     | 0.4     | 0.4      | 0.4      | 0.4      | 0.4      |
| write_trip_matrices                   | 18.8    | 20.3    | 19.8     | 19.3     | 18.9     | 19.1     |
| write tables                          | 8.1     | 7.9     | 7.8      | 7.7      | 7.3      | 7.5      |
| Total Runtime                         | 309.5   | 211.4   | 186.2    | 182.4    | 177.3    | 180      |

### Multiprocessing, Sharrow Disabled

The following performance metrics were collected for the SANDAG example model,
running on a Windows server with 24 cores and 500GB of RAM (same as the results above),
using the following configuration:

- `sharrow: false`
- `multiprocess: True`

See more complete results and discussion
[here](https://github.com/ActivitySim/sandag-abm3-example/issues/9).


| model name                            | 4 cores | 8 cores | 12 cores | 16 cores | 20 cores | 24 cores |
|---------------------------------------|---------|---------|----------|----------|----------|----------|
| mp_setup_skims                        | 11.2    | 10.4    | 10.2     | 10.4     | 10.4     | 10.2     |
| initialize_proto_population           | 0.5     | 0.5     | 0.5      | 0.5      | 0.5      | 0.5      |
| mp_da_apportion                       | 0.2     | 0.2     | 0.2      | 0.2      | 0.3      | 0.3      |
| compute_disaggregate_accessibility    | 13.7    | 7.7     | 5.9      | 5.1      | 4.4      | 3.8      |
| m p_da_coa lesce                      | 0.4     | 0.4     | 0.4      | 0.4      | 0.3      | 0.4      |
| initialize landuse                    | 0.4     | 0.4     | 0.4      | 0.4      | 0.4      | 0.4      |
| initialize households                 | 4.2     | 4.2     | 4.2      | 4.6      | 4.3      | 4.2      |
| m p_accessibility_apportion           | 1.1     | 1.8     | 2.8      | 3.5      | 4.4      | 4.9      |
| compute_accessibility                 | 7       | 4       | 3.1      | 2.5      | 2.1      | 1.8      |
| mp accessibility coalesce             | 0.7     | 0.6     | 0.7      | 0.6      | 0.6      | 0.6      |
| mp_households_apportion               | 0.7     | 1       | 1.2      | 1.4      | 1.6      | 1.8      |
| av_ownership                          | 0.6     | 0.5     | 0.5      | 0.5      | 0.5      | 0.5      |
| auto_ownership_simulate               | 0.4     | 0.2     | 0.1      | 0.1      | 0.1      | 0.1      |
| work from home                        | 0.2     | 0.1     | 0.1      | 0.1      | 0        | 0        |
| external worker identification        | 1.1     | 0.9     | 0.9      | 0.8      | 0.7      | 0.6      |
| external_workplace_location           | 1.1     | 1.7     | 2.3      | 3.1      | 3.7      | 4.2      |
| school location                       | 28.5    | 16.2    | 13       | 16.6     | 11.3     | 11       |
| workplace_location                    | 45.2    | 24.9    | 19       | 25.4     | 14.7     | 13.5     |
| transit_pass_subsidy                  | 0.4     | 0.2     | 0.2      | 0.4      | 0.2      | 0.1      |
| transit_pass_ownership                | 0.3     | 0.2     | 0.2      | 0.4      | 0.2      | 0.1      |
| vehicle_type_choice                   | 19.2    | 10.6    | 7.9      | 10.4     | 6        | 5.2      |
| adjust_auto_operating_cost            | 0       | 0       | 0.1      | 0.1      | 0.1      | 0.1      |
| transponder_ownership                 | 0.2     | 0.1     | 0.1      | 0.1      | 0.1      | 0.1      |
| free_parking                          | 0.3     | 0.2     | 0.2      | 0.1      | 0.1      | 0.1      |
| telecommute_frequency                 | 0.3     | 0.2     | 0.1      | 0.1      | 0.1      | 0.1      |
| cdap_simulate                         | 1       | 0.7     | 0.6      | 0.7      | 0.7      | 0.7      |
| mandatory_tour_frequency              | 0.4     | 0.2     | 0.2      | 0.2      | 0.2      | 0.1      |
| mandatory_tour_scheduling             | 40.1    | 23      | 17.9     | 17.1     | 14.2     | 12.5     |
| school_escorting                      | 9.5     | 5.2     | 3.9      | 3.4      | 3        | 2.6      |
| joint_tour_frequency_composition      | 0.9     | 0.6     | 0.4      | 0.4      | 0.4      | 0.4      |
| external_joint_tour_identification    | 0.9     | 0.7     | 0.6      | 0.6      | 0.6      | 0.5      |
| joint_tour_participation              | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| joint_tour_destination                | 6.2     | 3.4     | 2.5      | 2.1      | 2        | 1.9      |
| external_joint_tour_destination       | 0.5     | 0.4     | 0.4      | 0.5      | 0.5      | 0.6      |
| joint_tour_scheduling                 | 3.4     | 1.9     | 1.5      | 1.3      | 1.2      | 1        |
| non_mandatory_tour_frequency          | 5.4     | 3.1     | 2.3      | 2        | 1.8      | 1.5      |
| external_non_mandatory_identification | 1.7     | 1.3     | 1.2      | 1.3      | 1.4      | 1.3      |
| non mandatory_tour_destination        | 54.7    | 30.2    | 22.6     | 19.1     | 17.3     | 15.4     |
| external_non_mandatory_destination    | 0.6     | 0.6     | 0.5      | 0.6      | 0.6      | 0.8      |
| non_mandatory_tour_scheduling         | 96.6    | 78.1    | 78.5     | 84.6     | 94.6     | 100.7    |
| vehicle allocation                    | 3.4     | 1.9     | 1.5      | 1.3      | 1.2      | 1.1      |
| tour mode choice simulate             | 6.3     | 4.3     | 4.3      | 4.7      | 4.6      | 4.9      |
| atwork subtour_frequency              | 0.3     | 0.3     | 0.2      | 0.3      | 0.3      | 0.3      |
| atwork subtour destination            | 8.2     | 4.4     | 3.3      | 3        | 2.6      | 2.5      |
| atwork subtour_scheduling             | 1.4     | 0.8     | 0.7      | 0.7      | 0.6      | 0.6      |
| atwork subtour mode choice            | 0.7     | 0.6     | 0.6      | 0.7      | 0.6      | 0.7      |
| stop_frequency                        | 2.9     | 1.6     | 1.2      | 1        | 0.9      | 0.9      |
| trip_p urpose                         | 0.5     | 0.3     | 0.3      | 0.3      | 0.3      | 0.3      |
| trip_destination                      | 233.2   | 130.9   | 100.1    | 88.9     | 78.9     | 74.5     |
| trip_purpose_and_destination          | 0.2     | 0.3     | 0.3      | 0.4      | 0.3      | 0.3      |
| trip_scheduling                       | 0.6     | 0.5     | 0.4      | 0.4      | 0.4      | 0.4      |
| trip_mode_choice                      | 10.3    | 5.9     | 4.4      | 4        | 3.3      | 3.2      |
| parking_location                      | 2       | 1.3     | 1        | 0.9      | 0.8      | 0.8      |
| mp_households_coalesce                | 7.6     | 7.5     | 7.5      | 7.7      | 7.6      | 7.6      |
| write_data_dictionary                 | 6.5     | 6.5     | 6.5      | 6.6      | 6.5      | 6.6      |
| track_skim_usage                      | 0.3     | 0.3     | 0.3      | 0.4      | 0.3      | 0.3      |
| write_trip_matrices                   | 25.5    | 25.5    | 25.6     | 25.9     | 25.7     | 25.7     |
| write tables                          | 6.8     | 6.8     | 6.8      | 72       | 6.8      | 6.8      |
| Total Runtime                         | 690.2   | 448.7   | 387.2    | 394.3    | 361.9    | 363.5    |
