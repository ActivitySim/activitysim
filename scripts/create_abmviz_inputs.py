
# create abmviz input files (partially complete)
# Ben Stabler, ben.stabler@rsginc.com, 07/31/18

import pandas as pd

pipeline_filename = 'output/pipeline.h5'
trips_filename = 'output/ABMVIZ_trips.csv'
animatedmap_filename = 'output/ABMVIZ_3DAnimatedMapData.csv'
barchartandmap_filename = 'output/ABMVIZ_BarChartAndMapData.csv'
barchart_filename = 'output/ABMVIZ_BarChartData.csv'
radarchart_filename = 'output/ABMVIZ_RadarChartsData.csv'
timeuse_filename = 'output/ABMVIZ_TimeUseData.csv'
treemap_filename = 'output/ABMVIZ_TreeMapData.csv'

# get pipeline trips table and add other required fields
pipeline = pd.io.pytables.HDFStore(pipeline_filename)
trips = pipeline['/trips/trip_mode_choice']
households = pipeline['/households/joint_tour_frequency']
households = households.set_index("hhno", drop=False)
tours = pipeline['/tours/tour_mode_choice_simulate']
trips['home_taz'] = households.loc[trips['household_id']]['TAZ'].tolist()
trips['tour_start'] = tours.loc[trips['tour_id']]['start'].tolist()
trips['tour_end'] = tours.loc[trips['tour_id']]['end'].tolist()

trips['in_or_out'] = 0
# trips['in_or_out'][trips['outbound'] == True] = 10

trips['inbound'] = ~trips['outbound']
trips = trips.sort_values(['tour_id', 'inbound', 'trip_num'])

trips['origin_purpose'] = 'Home'
trips['destination_purpose'] = trips['purpose']
trips['origin_purpose_start'] = 1
trips['destination_purpose_start'] = trips['depart']
trips['origin_purpose_end'] = 1
trips['destination_purpose_end'] = 1

trips.to_csv(trips_filename)

# create 3D animated map file

# create remainder of the day at home table
remainder = trips.groupby(['person_id']).max()[['home_taz', 'depart']]
remainder = pd.crosstab(remainder["home_taz"], remainder["depart"])

# loop by period and add trips to DayPop table


# /* Person location by PERIOD of the day based on trips */
# 		INSERT INTO DAYPOP_TEMP (TAZ, PER, PERSONS) SELECT ORIG_TAZ, @hrStr AS PER, COUNT(*) AS PERSONS
# 		FROM TRIPS
# 		WHERE ORIG_PURPOSE_START_PERIOD < (@hr+1) AND DEPART_PERIOD > (@hr-1)
# 		GROUP BY ORIG_TAZ

# DECLARE @minPERIOD AS INT
# DECLARE @maxPERIOD AS INT
# DECLARE @hr AS INT
# DECLARE @hrStr AS VARCHAR(6)

# SET @minPERIOD = 1
# SET @maxPERIOD = 48

# CREATE TABLE DAYPOP (TAZ INT, PER VARCHAR(6), PERSONS INT, PERSONSNOTATHOME INT)
# CREATE TABLE DAYPOP_TEMP (TAZ INT, PER VARCHAR(6), PERSONS INT, PERSONSNOTATHOME INT)

# create bar chart and map file

# create bar chart file

# create radar chart file

# create timeuse file

# create treemap file
