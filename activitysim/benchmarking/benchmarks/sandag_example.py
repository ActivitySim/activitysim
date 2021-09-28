COMPONENT_NAMES = [
    # "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    # "write_data_dictionary",
    # "track_skim_usage",
    "write_trip_matrices",
    # "write_tables",
]
BENCHMARK_SETTINGS = {
    "households_sample_size": 48_769,
}
SKIM_CACHE = True
PRELOAD_INJECTABLES = ("skim_dict",)
REPEAT = 1
NUMBER = 1
TIMEOUT = 36000.0  # ten hours
