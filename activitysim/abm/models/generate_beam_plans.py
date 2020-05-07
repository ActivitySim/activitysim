import numpy as np
import pandas as pd
import random
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import geopandas as gpd

from activitysim.core import pipeline
from activitysim.core import inject


def random_points_in_polygon(number, polygon):
    '''
    Generate n number of points within a polygon
    Input:
    -number: n number of points to be generated
    - polygon: geopandas polygon
    Return:
    - List of shapely points
    source: https://gis.stackexchange.com/questions/294394/
        randomly-sample-from-geopandas-dataframe-in-python
    '''
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i = 0
    while i < number:
        point = Point(
            random.uniform(min_x, max_x), random.uniform(min_y, max_y))
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
    - overestimate = int to multiply the size. It will account for
        points that may fall outside the polygon
    Return:
    - List points
    source: https://gis.stackexchange.com/questions/294394/
        randomly-sample-from-geopandas-dataframe-in-python
    '''
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area
    overestimate = 2
    samples = np.random.uniform(
        (min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2))
    multipoint = MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)
    return samples[np.random.choice(len(samples), size)]


def get_trip_coords(trips, zones, persons, size=500):

    # Generates random points within each zone
    rand_point_zones = {}
    for zone in zones.TAZ:
        size = 500
        polygon = zones[zones.TAZ == zone].geometry
        points = sample_geoseries(polygon, size, overestimate=2)
        rand_point_zones[zone] = points

    # Assign semi-random (within zone) coords to trips
    df = trips[['origin']]
    origins = []
    for i, row in enumerate(df.itertuples(), 0):
        origins.append(random.choice(rand_point_zones[row.origin]))

    origins = np.array(origins)

    trips['origin_x'] = pd.Series(origins[:, 0], index=trips.index)
    trips['origin_y'] = pd.Series(origins[:, 1], index=trips.index)

    # retain home coords from urbansim data bc they will typically be
    # higher resolution than zone, so we don't need the semi-random coords
    trips = pd.merge(
        trips, persons[['home_x', 'home_y']],
        left_on='person_id', right_index=True)
    trips['origin_purpose'] = trips.purpose.shift(periods=1).fillna('Home')
    trips['x'] = trips.origin_x.where(
        trips.origin_purpose != 'Home', trips.home_x)
    trips['y'] = trips.origin_y.where(
        trips.origin_purpose != 'Home', trips.home_y)

    return trips


def generate_departure_times(trips):
    df = trips[['person_id', 'depart']].reset_index()
    df['frac'] = np.random.rand(len(df),)

    # Making sure trips within the hour are sequential
    ordered = df.sort_values(
        by=['person_id', 'depart', 'frac']).reset_index(drop=True)
    df = df.sort_values(by=['person_id', 'depart']).reset_index(drop=True)
    df['fractional'] = ordered.frac

    # Adding fractional to int hour
    df['depart'] = np.round(df['depart'] + df['fractional'], 3)
    df.set_index('trip_id', inplace=True)
    return df.depart


@inject.step()
def generate_beam_plans():

    # Importing ActivitySim results
    trips = pipeline.get_table('trips')
    persons = pipeline.get_table('persons')
    zones = pipeline.get_table('land_use')

    # re-cast zones as a geodataframe
    zones['geometry'] = zones['geometry'].apply(wkt.loads)
    zones = gpd.GeoDataFrame(zones, geometry='geometry', crs="EPSG:4326")
    zones.geometry = zones.geometry.buffer(0)
    zones.reset_index(inplace=True)

    # augment trips table
    trips = get_trip_coords(trips, zones, persons)
    trips['departure_time'] = generate_departure_times(trips)

    # trim trips table
    cols = [
        'person_id', 'departure_time', 'purpose', 'origin',
        'destination', 'trip_mode', 'x', 'y']
    trips = trips[cols].sort_values(
        ['person_id', 'departure_time']).reset_index()

    # Adding a new row for each unique person_id
    # this row will represent the returning trip
    return_trip = pd.DataFrame(
        trips.groupby('person_id').agg({'x': 'first', 'y': 'first'}),
        index=trips.person_id.unique())

    trips = trips.append(return_trip)
    trips.reset_index(inplace=True)
    trips.person_id.fillna(trips['index'], inplace=True)

    # Creating the Plan Element activity Index
    # Activities have odd indices and legs (actual trips) will be even
    trips['PlanElementIndex'] = trips.groupby('person_id').cumcount() * 2 + 1
    trips = trips.sort_values(
        ['person_id', 'departure_time']).reset_index(drop=True)

    # Shifting type one row down
    trips['ActivityType'] = trips.purpose.shift(periods=1).fillna('Home')
    trips['ActivityElement'] = 'activity'

    # Creating legs (trips between activities)
    legs = pd.DataFrame({
        'PlanElementIndex': trips.PlanElementIndex - 1,
        'person_id': trips.person_id})
    legs = legs[legs.PlanElementIndex != 0]

    # Adding the legs to the main table
    trips = trips.append(legs).sort_values(['person_id', 'PlanElementIndex'])
    trips.ActivityElement.fillna('leg', inplace=True)

    # save back to pipeline
    pipeline.replace_table("beam_plans", trips[[
        'person_id', 'PlanElementIndex', 'ActivityElement', 'ActivityType',
        'x', 'y', 'departure_time']])
