library(tidyverse)
library(ggplot2)

trips_df <- read_csv("./output/trips_after_parking_choice.csv")

walk_dist_df <- read_csv("./data/ped_distance_maz_maz.csv")

landuse_df <- read_csv("./data/land_use.csv")

landuse_df <- landuse_df %>%
  select(maz, maz_county_based, TAZ_ORIGINAL)

work_df <- trips_df %>%
  rename(parking_maz_target = parking_mgra) %>%
  rename(parking_maz_simulated = parking_zone) %>%
  select(trip_id, household_id, person_id, person_num, stop_id, tour_purpose, orig_purpose, dest_purpose, orig_maz = origin, dest_maz = destination, 
         activity_duration_in_hours, parking_maz_target, parking_maz_simulated) %>%
  left_join(landuse_df, by = c("orig_maz" = "maz")) %>%
  rename(orig_maz_county_based = maz_county_based, orig_taz = TAZ_ORIGINAL) %>%
  left_join(landuse_df, by = c("dest_maz" = "maz")) %>%
  rename(dest_maz_county_based = maz_county_based, dest_taz = TAZ_ORIGINAL) %>%
  left_join(landuse_df, by = c("parking_maz_target" = "maz")) %>%
  rename(parking_maz_county_based_target = maz_county_based, parking_taz_target = TAZ_ORIGINAL) %>%
  left_join(landuse_df, by = c("parking_maz_simulated" = "maz")) %>%
  rename(parking_maz_county_based_simulated = maz_county_based, parking_taz_simulated = TAZ_ORIGINAL) %>%
  left_join(walk_dist_df, by = c("parking_maz_simulated" = "OMAZ", "dest_maz" = "DMAZ")) %>%
  rename(parking_distance_simulated = DISTWALK) %>%
  left_join(walk_dist_df, by = c("parking_maz_target" = "OMAZ", "dest_maz" = "DMAZ")) %>%
  rename(parking_distance_target = DISTWALK) %>%
  left_join(walk_dist_df, by = c("parking_maz_target" = "OMAZ", "parking_maz_simulated" = "DMAZ")) %>%
  rename(distance_target_simulation = DISTWALK) %>%
  select(trip_id, household_id, person_id, stop_id, tour_purpose, orig_purpose, dest_purpose,
         orig_maz, dest_maz, orig_maz_county_based, dest_maz_county_based, orig_taz, dest_taz,
         parking_maz_target, parking_maz_simulated, parking_taz_target, parking_taz_simulated,
         parking_distance_target, parking_distance_simulated, distance_target_simulation)


write.csv(work_df, "./output/parking-location-choice-results.csv", row.names = F)


summary_df <- work_df %>%
  group_by(dest_purpose) %>%
  summarise(
    trips_count = n(),
    min_distance_target = round(min(parking_distance_target, na.rm = T), 3),
    min_distance_simulated = round(min(parking_distance_simulated, na.rm = T), 3),
    max_distance_target = round(max(parking_distance_target, na.rm = T), 3),
    max_distance_simulated = round(max(parking_distance_simulated, na.rm = T), 3),
    mean_distance_target = round(mean(parking_distance_target, na.rm = T), 3),
    mean_distance_simulated = round(mean(parking_distance_simulated, na.rm = T), 3),
    median_distance_target = round(median(parking_distance_target, na.rm = T), 3),
    median_distance_simulated = round(median(parking_distance_simulated, na.rm = T), 3)
  )

#ggplot(work_df, aes(x=distance_target_simulation)) + geom_histogram(binwidth=0.025, color="black", fill="white")

#ggplot(work_df, aes(x=parking_distance_target)) + geom_histogram(binwidth=0.05, color="black", fill="white")

#ggplot(work_df, aes(x=parking_distance_simulated)) + geom_histogram(binwidth=0.05, color="black", fill="white")



