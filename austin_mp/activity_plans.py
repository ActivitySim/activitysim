# # ActivitySim activity Plans for BEAM

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import os
import orca
import time
import geopandas as gpd
import random
import shapely
from h3 import h3
from shapely.geometry import Point, Polygon
from urbansim.utils import misc
from platform import python_version


output_bucket = 'austin-activitysim/base/'
output_table_map = {
    'beam_activity_plans': 'plans',
    'trips/trip_mode_choice': 'trips',
    'tours/trip_mode_choice': 'tours'}

print('Creating Activity Plans')
start = time.time()



# ## Loading data
#Importing ActivitySim results
hdf = pd.HDFStore(path = 'output/pipeline.h5', mode = 'a')

# Loading trips and persons
trips = hdf['/trips/trip_mode_choice'].sort_values(['person_id','depart'])
persons = hdf['/persons/trip_mode_choice']
skims = pd.read_csv(
    'https://beam-outputs.s3.amazonaws.com/output/austin/'
    'austin-prod-200k-skims-with-h3-index-final__2020-04-18_09-44-24_wga/'
    'ITERS/it.0/0.skimsOD.UrbanSim.Full.csv.gz')
#This should be the H3 geopandas file
# zones_shp = gpd.read_file("data/zone_files/Transportation_Analysis_Zones.shp")

def zones(skims):
    """
    Returns a GeoPandasDataframe with the H3 hexbins information 
    """
    zone_ids = skims.origTaz.unique()
    
    #Get boundaries of the H3 hexbins
    polygon_shapes = []
    for zone in zone_ids:
        boundary_points = h3.h3_to_geo_boundary(h3_address=zone, geo_json=True)
        shape = Polygon(boundary_points)
        polygon_shapes.append(shape)

    #Organize information in a GeoPandas dataframe to merge with blocks
    h3_zones = gpd.GeoDataFrame(zone_ids, geometry = polygon_shapes, crs = "EPSG:4326")
    h3_zones.columns = ['h3_id', 'geometry']
    h3_zones['area'] = h3_zones.geometry.area
    h3_zones['TAZ'] = list(range(1, len(zone_ids)+1))
    return h3_zones.set_index('TAZ')

zones_shp = zones(skims)

#Closes boundary loops
zones_shp.geometry = zones_shp.geometry.buffer(0)
zones_shp.reset_index(inplace=True)

#Adding tables to orca
orca.add_table('trips', trips)
orca.add_table('persons', persons);


def random_points_in_polygon(number, polygon):
    '''
    Generate n number of points within a polygon 
    Input: 
    -number: n number of points to be generated
    - polygon: geopandas polygon 
    Return: 
    - List of shapely points
    source: https://gis.stackexchange.com/questions/294394/randomly-sample-from-geopandas-dataframe-in-python
    '''
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points  # returns list of shapely point


def sample_geoseries(geoseries, size, overestimate=2):
    '''
    Generate at most "size" number of points within a polygon 
    Input: 
    - size: n number of points to be generated
    - geoseries: geopandas polygon 
    - overestimate = int to multiply the size. It will account for points that may fall outside the polygon
    Return: 
    - List points
    source: https://gis.stackexchange.com/questions/294394/randomly-sample-from-geopandas-dataframe-in-python
    '''
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area
    overestimate = 2
    samples = np.random.uniform((min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2))
    multipoint = shapely.geometry.MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)
#     while samples.shape[0] < size:
#         # emergency catch in case by bad luck we didn't get enough within the polygon
#         samples = np.concatenate([samples, random_points_in_polygon(polygon, size, overestimate=overestimate)])
    return samples[np.random.choice(len(samples), size)]

#Generates random points in all zones
rand_point_zones = {}
for zone in zones_shp.TAZ:
    size = 500
    polygon = zones_shp[zones_shp.TAZ == zone].geometry
    points = sample_geoseries(polygon, size, overestimate=2)
    rand_point_zones[zone]=points


# Generates random X,Y coordinates for each origin 
# Note: destination will be the otigin of the next activity. 
@orca.injectable()
def ods(trips):
    df = trips.to_frame(columns = ['origin'])
    origs = []
    for i, row in enumerate(df.itertuples(), 0):
        origs.append(random.choice(rand_point_zones[row.origin]))
    return np.array(origs)

origs = orca.get_injectable('ods')


# Columns to trips table

@orca.column('trips')
def x():
    return pd.Series(origs[:,0], index= trips.index)

@orca.column('trips')
def y():
    return pd.Series(origs[:,1], index= trips.index)


# USE THIS CHUNK IF HOME COORDS ARE HIGHER RESOLUTION
# THAN THE ZONES (i.e. from a parcel model)

# @orca.column('trips')
# def orig_x():
#     return pd.Series(origs[:,0], index= trips.index)

# @orca.column('trips')
# def orig_y():
#     return pd.Series(origs[:,1], index= trips.index)

# @orca.column('trips')
# def home_x(persons, trips):
#     return misc.reindex(persons.home_x, trips.person_id)

# @orca.column('trips')
# def home_y(persons, trips):
#     return misc.reindex(persons.home_y, trips.person_id)

# @orca.column('trips')
# def origin_purpose(trips):
#     return trips.purpose.shift(periods = 1).fillna('Home')

# @orca.column('trips')
# def x(trips):
#     return trips.orig_x.where(trips.origin_purpose != 'Home', trips.home_x)

# @orca.column('trips')
# def y(trips):
#     return trips.orig_y.where(trips.origin_purpose != 'Home', trips.home_y)

@orca.column('trips')
def departure_time(trips):
    '''
    Generates a randon departure time within a given hour. 
    This function makes sure sequential trips within the same hour are ordered
    
    Input:
    - trips: ActivitySim trips output table
    
    Output:
    - trips table with randomized departure time witin an hour. 
      Table is sorted by person_id and departure time
    '''
    df = trips.to_frame(columns=['person_id','depart']).reset_index()
    
    #Generate randon decimal part to be added to the hour
    df['frac'] = np.random.rand(len(df),)

    #Making sure trips within the hour are sequential
    ordered = df.sort_values(by=['person_id','depart','frac']).reset_index(drop = True)
    df = df.sort_values(by=['person_id','depart',]).reset_index(drop = True)
    df['fractional'] = ordered.frac
    
    #Adding fractional to int hour
    df['depart'] = np.round(df['depart'] + df['fractional'],3)
    df.set_index('trip_id', inplace = True)
    return df.depart


@orca.table('activity_plan')
def activity_plan(trips, persons):
    
    #trips.sort_values(['person_id','depart' ], inplace= True)
    cols = ['person_id','departure_time', 'purpose', 'origin', 'destination', 'trip_mode', 
            'x', 'y']
    trips = trips.to_frame(columns = cols).sort_values(['person_id','departure_time']).reset_index()
    
    #Adding a new row for each unique person_id
    #this row will represent the returning trip
    return_trip = pd.DataFrame(trips.groupby('person_id').agg({'x':'first', 'y':'first'}), 
             index=trips.person_id.unique())
    
    trips = trips.append(return_trip)
    trips.reset_index(inplace = True)
    trips.person_id.fillna(trips['index'], inplace=True)

    #Creating the Plan Element activity Index 
    # Activities have odd PlantElementIndex, and legs (actual trips) will have even index. 
    trips['PlanElementIndex'] = trips.groupby('person_id').cumcount() * 2 + 1
    trips = trips.sort_values(['person_id','departure_time' ]).reset_index(drop = True)

    #Shifting type one row down
    trips['ActivityType'] = trips.purpose.shift(periods = 1).fillna('Home')
    trips['ActivityElement'] = 'activity'

    #Creating leg (Trips between activities)
    legs = pd.DataFrame({'PlanElementIndex':trips.PlanElementIndex - 1, 
                         'person_id' : trips.person_id})
    legs = legs[legs.PlanElementIndex != 0]

    #Adding the legs to the main table
    trips = trips.append(legs).sort_values(['person_id', 'PlanElementIndex'])
    trips.ActivityElement.fillna('leg', inplace = True)

    
    return trips[['person_id','PlanElementIndex', 'ActivityElement',
                 'ActivityType', 'x', 'y', 'departure_time']]

@orca.step()
def generate_activity_plans():
    activity_plan = orca.get_table('activity_plan').to_frame()
    activity_plan.to_hdf('output/pipeline.h5', key='beam_activity_plans')
    hdf.close()

orca.run(['generate_activity_plans'])

input_store = pd.HDFStore('../data/model_data.h5')
for tablename in output_table_map.keys():
    output_tablename = output_table_map[tablename]
    input_store[output_tablename] = hdf[tablename]

for tablename in input_store.keys(): 
    tablename = tablename.replace('/', '')
    table = input_store[tablename]
    table.to_csv(os.path.join('s3://', output_bucket, tablename + '.csv')

end = time.time()
print("Run time for Activity Plans = {} seconds.".format(end - start))
