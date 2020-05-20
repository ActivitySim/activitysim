import os
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import orca
from shapely.geometry import Polygon
from h3 import h3
from urbansim.utils import misc
import requests
import openmatrix as omx
import logging

from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger(__name__)


def get_zone_geoms_from_h3(h3_ids):
    polygon_shapes = []
    for zone in h3_ids:
        boundary_points = h3.h3_to_geo_boundary(h3_address=zone, geo_json=True)
        shape = Polygon(boundary_points)
        polygon_shapes.append(shape)

    return polygon_shapes


# ** 1. CREATE NEW TABLES **

# Zones
@orca.table('zones', cache=True)
def zones():
    """
    if loading zones from shapefile, coordinates must be
    referenced to WGS84 (EPSG:4326) projection.
    """
    usim_zone_geoms = config.setting('usim_zone_geoms')

    if usim_zone_geoms == 'shp':
        fname = config.setting('usim_zone_shapefile')
        if fname is None:
            raise RuntimeError(
                "Trying to create intermediate zones table from shapefile "
                "but 'usim_zone_shapefile' not specified in settings.yaml")
        filepath = config.data_file_path(fname)
        zones = gpd.read_file(filepath, crs="EPSG:4326")
        zones['area'] = zones.geometry.area
        zones.reset_index(inplace=True, drop=True)
        zones.index.name = 'TAZ'

    elif usim_zone_geoms == 'h3':
        try:
            h3_zone_ids = inject.get_injectable('h3_zone_ids')
        except KeyError:
            raise RuntimeError(
                "Trying to create intermediate zones table from h3 IDs "
                "but the 'h3_zone_ids' injectable is not defined")
        zone_geoms = get_zone_geoms_from_h3(h3_zone_ids)
        h3_zones = gpd.GeoDataFrame(
            h3_zone_ids, geometry=zone_geoms, crs="EPSG:4326")
        h3_zones.columns = ['h3_id', 'geometry']
        h3_zones['area'] = h3_zones.geometry.area
        h3_zones['TAZ'] = list(range(1, len(h3_zone_ids) + 1))
        return h3_zones.set_index('TAZ')


# Schools
@orca.table(cache=True)
def schools(blocks):

    base_url = 'https://educationdata.urban.org/api/v1/' + \
        '{topic}/{source}/{endpoint}/{year}/?{filters}'

    county_codes = blocks.index.str.slice(0, 5).unique()

    school_tables = []
    for county in county_codes:
        enroll_filters = 'county_code={0}'.format(county)
        enroll_url = base_url.format(
            topic='schools', source='ccd', endpoint='directory',
            year='2015', filters=enroll_filters)

        enroll_result = requests.get(enroll_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        school_tables.append(enroll)

    enrollment = pd.concat(school_tables, axis=0)
    enrollment = enrollment[[
        'ncessch', 'county_code', 'latitude',
        'longitude', 'enrollment']].set_index('ncessch')
    return enrollment.dropna()


# Colleges
@orca.table(cache=True)
def colleges(blocks):

    base_url = 'https://educationdata.urban.org/api/v1/' + \
        '{topic}/{source}/{endpoint}/{year}/?{filters}'
    county_codes = blocks.index.str.slice(0, 5).unique()

    colleges_list = []
    for county in county_codes:
        college_filters = 'county_fips={0}'.format(county)
        college_url = base_url.format(
            topic='college-university', source='ipeds', endpoint='directory',
            year='2015', filters=college_filters)

        college_result = requests.get(college_url)
        college = pd.DataFrame(college_result.json()['results'])
        colleges_list.append(college)

    colleges = pd.concat(colleges_list)
    colleges = colleges[[
        'unitid', 'inst_name', 'longitude', 'latitude']].set_index('unitid')
    colleges.rename(columns={'longitude': 'x', 'latitude': 'y'}, inplace=True)
    return colleges


# ** 2. CREATE NEW VARIABLES/COLUMNS **

# Block Variables

# NOTE: AREAS OF BLOCKS BASED ON RESIDENTS AND EMPLOYEES PER BLOCK.
# PROPER LAND USE DATA SHOULD BE PROCURED FROM THE MPO

@orca.column('blocks', cache = True)
def TAZ(blocks, zones):

    # Tranform blocks to a Geopandas dataframe
    blocks_df = blocks.to_frame(columns=['x', 'y'])
    zones_df =  zones.to_frame(columns=['geometry', 'area'])
    h3_gpd = gpd.GeoDataFrame(zones_df, crs='EPSG:4326')

    blocks_df = gpd.GeoDataFrame(
        blocks_df, geometry=gpd.points_from_xy(blocks_df.x, blocks_df.y),
        crs="EPSG:4326")

    # Spatial join 
    blocks_df = gpd.sjoin(blocks_df, h3_gpd, how='left', op = 'intersects')

    # Drop duplicates and keep the one with the smallest H3 area
    blocks_df = blocks_df.sort_values('area')
    blocks_df.drop_duplicates(subset = ['x', 'y'], keep = 'first', inplace = True) 
    
    # Buffer unassigned blocks until they reach a hexbin. 
    null_blocks = blocks_df[blocks_df.index_right.isnull()].drop(columns = ['index_right','area'])

    result_list = []
    for index, block in null_blocks.iterrows():
        buff_size = 0.0001
        matched = False
        geo_block = gpd.GeoDataFrame(block, crs='EPSG:4326').T
        while matched == False:
            geo_block.geometry = geo_block.geometry.buffer(buff_size)
            result = gpd.sjoin(geo_block, h3_gpd, how = 'left', op = 'intersects')
            matched = ~result.index_right.isnull()[0]
            buff_size = buff_size + 0.0001
        result_list.append(result.iloc[0:1])

    null_blocks = pd.concat(result_list)
    
    # Concatenate newly assigned blocks to the main blocks table 
    blocks_df = blocks_df.dropna()
    blocks_df = pd.concat([blocks_df, null_blocks], axis = 0)
    
    return blocks_df.index_right


@orca.column('blocks')
def CI_employment(jobs, blocks):
    job = jobs.to_frame()
    job = job[job.sector_id.isin([11, 3133, 42, 4445, 4849, 52, 54, 7172])]
    s = job.groupby('block_id')['sector_id'].count()

    # to avoid division by zero best to have a relative greater number,
    # so that dividing by this number results in a small value
    return s.reindex(blocks.index).fillna(0.01) 


@orca.column('blocks')
def CIACRE(blocks):
    total_pop = blocks.residential_unit_capacity + blocks.employment_capacity
    ci_pct = blocks.CI_employment/total_pop
    ci_acres = (ci_pct * blocks.square_meters_land)/4046.86 #1m2 = 4046.86acres
    return ci_acres.fillna(0.01)


@orca.column('blocks')
def RESACRE(blocks):
    total_pop = blocks.residential_unit_capacity + blocks.employment_capacity
    res_pct = blocks.residential_unit_capacity/total_pop
    res_acres = (res_pct * blocks.square_meters_land)/4046.86 #1m2 = 4046.86acres
    return res_acres.fillna(0.01)


# School Variables

@orca.column('schools', cache = True)
def TAZ(schools, zones):

    #Tranform blocks to a Geopandas dataframe
    zones_df =  zones.to_frame(columns=['geometry', 'area'])
    h3_gpd = gpd.GeoDataFrame(zones_df, crs='EPSG:4326')

    school_gpd = schools.to_frame(columns = ['ncessch','longitude', 'latitude'])
    school_gpd = gpd.GeoDataFrame(
        school_gpd,
        geometry=gpd.points_from_xy(school_gpd.longitude, school_gpd.latitude),
        crs="EPSG:4326")
    # Spatial join 
    school_gdf = gpd.sjoin(school_gpd, h3_gpd, how = 'left', op = 'intersects')

    #Drop duplicates and keep the one with the smallest H3 area
    school_gdf = school_gdf.sort_values('area')
    school_gdf.reset_index(inplace = True)
    school_gdf.drop_duplicates(subset = ['ncessch'], keep = 'first', inplace = True) 
    
    #Buffer unassigned blocks until they reach a hexbin. 
    null_schools = school_gdf[school_gdf.index_right.isnull()].drop(columns = ['index_right','area'])

    result_list = []
    for index, school in null_schools.iterrows():
        buff_size = 0.0001
        matched = False
        geo_school = gpd.GeoDataFrame(school, crs='EPSG:4326').T
        while matched == False:
            geo_school.geometry = geo_school.geometry.buffer(buff_size)
            result = gpd.sjoin(geo_school, h3_gpd, how = 'left', op = 'intersects')
            matched = ~result.index_right.isnull().iloc[0]
            buff_size = buff_size + 0.0001
        result_list.append(result.iloc[0:1])

    null_school = pd.concat(result_list)

    # Concatenate newly assigned blocks to the main blocks table 
    school_gdf = school_gdf.dropna()
    school_all = pd.concat([school_gdf, null_school], axis = 0)
    school_all.set_index('ncessch', inplace = True)
    return school_all.index_right


# Colleges Variables

@orca.column('colleges')
def full_time_enrollment():
    base_url = 'https://educationdata.urban.org/api/v1/{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}'
    levels = ['undergraduate','graduate']

    enroll_list = []
    for level in levels: 
        base_url = base_url.format(t='college-university', so='ipeds', e='fall-enrollment', 
                                   y='2015', l = level,f='ftpt=1', s = 'sex=99', 
                                   r = 'race=99' , cl = 'class_level=99',ds = 'degree_seeking=99',
                                   fips = 'fips=48')

        enroll_result = requests.get(base_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(columns = {'enrollment_fall':level})
        enroll.set_index('unitid', inplace = True)
        enroll_list.append(enroll)

    full_time = pd.concat(enroll_list, axis = 1)
    full_time['full_time'] = full_time['undergraduate'] + full_time['graduate']
    s = full_time.full_time
    return s


@orca.column('colleges')
def part_time_enrollment():
    base_url = 'https://educationdata.urban.org/api/v1/{t}/{so}/{e}/{y}/{l}/?{f}&{s}&{r}&{cl}&{ds}&{fips}'
    levels = ['undergraduate','graduate']

    enroll_list = []
    for level in levels: 
        base_url = base_url.format(t='college-university', so='ipeds', e='fall-enrollment', 
                                   y='2015', l = level,f='ftpt=2', s = 'sex=99', 
                                   r = 'race=99' , cl = 'class_level=99',ds = 'degree_seeking=99',
                                   fips = 'fips=48')

        enroll_result = requests.get(base_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        enroll = enroll[['unitid', 'enrollment_fall']].rename(columns = {'enrollment_fall':level})
        enroll.set_index('unitid', inplace = True)
        enroll_list.append(enroll)

    part_time = pd.concat(enroll_list, axis = 1)
    part_time['part_time'] = part_time['undergraduate'] + part_time['graduate']
    s = part_time.part_time
    return s


@orca.column('colleges', cache=True)
def TAZ(colleges, zones):
    #Tranform blocks to a Geopandas dataframe
    colleges_df = colleges.to_frame(columns = ['x', 'y'])
    zones_df =  zones.to_frame(columns = ['geometry', 'area'])
    h3_gpd = gpd.GeoDataFrame(zones_df, crs="EPSG:4326")
    
    colleges_df = gpd.GeoDataFrame(
        colleges_df, geometry=gpd.points_from_xy(colleges_df.x, colleges_df.y),
        crs="EPSG:4326")

    # Spatial join 
    colleges_df = gpd.sjoin(colleges_df, h3_gpd, how = 'left', op = 'intersects')

    #Drop duplicates and keep the one with the smallest H3 area
    colleges_df = colleges_df.sort_values('area')
    colleges_df.drop_duplicates(subset = ['x', 'y'], keep = 'first', inplace = True) 
    
    return colleges_df.index_right


# Households Variables

@orca.column('usim_households')
def TAZ(blocks, usim_households):
    return misc.reindex(blocks.TAZ, usim_households.block_id)


@orca.column('usim_households')
def HHT(usim_households):
    s = usim_households.persons
    return s.where(s == 1, 4)


# Persons Variables

@orca.column('usim_persons')
def TAZ(usim_households, usim_persons):
    return misc.reindex(usim_households.TAZ, usim_persons.household_id)


@orca.column('usim_persons')
def ptype(usim_persons):

    # Filters for person type segmentation
    # https://activitysim.github.io/activitysim/abmexample.html#setup
    age_mask_1 = usim_persons.age >= 18
    age_mask_2 = usim_persons.age.between(18, 64, inclusive=True)
    age_mask_3 = usim_persons.age >= 65
    work_mask = usim_persons.worker == 1
    student_mask = usim_persons.student == 1

    # Series for each person segmentation
    type_1 = ((age_mask_1) & (work_mask) & (~student_mask)) * 1  # Full time
    type_4 = ((age_mask_2) & (~work_mask) & (~student_mask)) * 4
    type_5 = ((age_mask_3) & (~work_mask) & (~student_mask)) * 5
    type_3 = ((age_mask_1) & (student_mask)) * 3
    type_6 = (usim_persons.age.between(16, 17, inclusive=True)) * 6
    type_7 = (usim_persons.age.between(6, 16, inclusive=True)) * 7
    type_8 = (usim_persons.age.between(0, 5, inclusive=True)) * 8
    type_list = [
        type_1, type_3, type_4, type_5, type_6, type_7, type_8]

    # Colapsing all series into one series
    for x in type_list:
        type_1.where(type_1 != 0, x, inplace=True)

    return type_1


@orca.column('usim_persons')
def pemploy(usim_persons):
    pemploy_1 = ((usim_persons.worker == 1) & (usim_persons.age >= 16)) * 1
    pemploy_3 = ((usim_persons.worker == 0) & (usim_persons.age >= 16)) * 3
    pemploy_4 = (usim_persons.age < 16) * 4

    # Colapsing all series into one series
    type_list = [pemploy_1, pemploy_3, pemploy_4]
    for x in type_list:
        pemploy_1.where(pemploy_1 != 0, x, inplace=True)

    return pemploy_1


@orca.column('usim_persons')
def pstudent(usim_persons):
    pstudent_1 = (usim_persons.age <= 18) * 1
    pstudent_2 = ((usim_persons.student == 1) & (usim_persons.age > 18)) * 2
    pstudent_3 = (usim_persons.student == 0) * 3

    # Colapsing all series into one series
    type_list = [pstudent_1, pstudent_2, pstudent_3]
    for x in type_list:
        pstudent_1.where(pstudent_1 != 0, x, inplace=True)

    return pstudent_1


# Jobs Variables

@orca.column('jobs')
def TAZ(blocks, jobs):
    return misc.reindex(blocks.TAZ, jobs.block_id)


@orca.column('zones', cache=True)
def TOTHH(usim_households, zones):
    s = usim_households.TAZ.groupby(usim_households.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHPOP(usim_persons, zones):
    s = usim_persons.TAZ.groupby(usim_persons.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def EMPRES(usim_households, zones):
    s = usim_households.to_frame().groupby('TAZ')['workers'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ1(usim_households, zones):
    df = usim_households.to_frame()
    df = df[df.income < 30000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ2(usim_households, zones):
    df = usim_households.to_frame()
    df = df[df.income.between(30000, 59999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ3(usim_households, zones):
    df = usim_households.to_frame()
    df = df[df.income .between(60000, 99999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HHINCQ4(usim_households, zones):
    df = usim_households.to_frame()
    df = df[df.income >= 100000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE0004(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age.between(0, 4)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE0519(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age.between(5, 19)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE2044(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age.between(20, 44)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE4564(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age.between(45, 64)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE65P(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age >= 65]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGE62P(usim_persons, zones):
    df = usim_persons.to_frame()
    df = df[df.age >= 62]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def SHPOP62P(zones):
    return (zones.AGE62P / zones.HHPOP).reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def TOTEMP(jobs, zones):
    s = jobs.TAZ.groupby(jobs.TAZ).count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def RETEMPN(jobs, zones):
    df = jobs.to_frame()

    # difference is here (44, 45 vs 4445)
    # sector ids don't match
    df = df[df.sector_id.isin([4445])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def FPSEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([52, 54])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HEREMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([61, 62, 71])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def AGREMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([11])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def MWTEMPN(jobs, zones):
    df = jobs.to_frame()

    # sector ids don't match
    df = df[df.sector_id.isin([42, 3133, 32, 4849])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def OTHEMPN(jobs, zones):
    df = jobs.to_frame()

    # sector ids don't match
    df = df[~df.sector_id.isin([
        4445, 52, 54, 61, 62, 71, 11, 42, 3133, 32, 4849])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def TOTACRE(zones):
    g = zones.geometry.to_crs({'init': 'epsg:3857'})

    # area in square meters
    area_polygons = g.area / 4046.86
    return area_polygons


@orca.column('zones', cache=True)
def RESACRE(blocks, zones):
    df = blocks.to_frame()
    s = df.groupby('TAZ')['RESACRE'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def CIACRE(blocks, zones):
    df = blocks.to_frame()
    s = df.groupby('TAZ')['CIACRE'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', cache=True)
def HSENROLL(schools, zones):
    s = schools.to_frame().groupby('TAZ')['enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')
def TOPOLOGY():
    # assumes everything is flat
    return 1


# Zones variables

@orca.column('zones')
def employment_density(zones):
    return zones.TOTEMP / zones.TOTACRE


@orca.column('zones')
def pop_density(zones):
    return zones.HHPOP / zones.TOTACRE


@orca.column('zones')
def hh_density(zones):
    return zones.TOTHH / zones.TOTACRE


@orca.column('zones')
def hq1_density(zones):
    return zones.HHINCQ1 / zones.TOTACRE


@orca.column('zones')
def PRKCST(zones):
    params = pd.Series(
        [-1.92168743,  4.89511403,  4.2772001 ,  0.65784643],
        index=['pop_density', 'hh_density', 'hq1_density', 'employment_density'])

    cols = zones.to_frame(columns=[
        'employment_density', 'pop_density', 'hh_density', 'hq1_density'])

    s = cols @ params
    return s.where(s > 0, 0)


@orca.column('zones')
def OPRKCST(zones):
    params = pd.Series(
        [-6.17833544, 17.55155703,  2.0786466 ], 
        index=['pop_density', 'hh_density', 'employment_density'])

    cols = zones.to_frame(
        columns=['employment_density', 'pop_density', 'hh_density'])

    s = cols @ params
    return s.where(s > 0, 0)


@orca.column('zones')  # College enrollment
def COLLFTE(colleges, zones):
    s = colleges.to_frame().groupby('TAZ')['full_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')  # College enrollment
def COLLPTE(colleges, zones):
    s = colleges.to_frame().groupby('TAZ')['part_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


@orca.column('zones')
def area_type():
    # Integer, 0=regional core, 1=central business district, 2=urban business,
    # 3=urban, 4=suburban, 5=rural
    return 0  # Assuming all regional core


@orca.column('zones')
def TERMINAL():
    # TO DO:
    # Improve the imputation of this variable
    # Average time to travel from automobile storage location to
    # origin/destination. We assume zero for now
    return 0  # Assuming O


@orca.column('zones')
def COUNTY():
    return 1  # Assuming 1 all San Francisco County


# ** 3. Define Orca Steps **

@inject.step()
def load_usim_data(data_dir, settings):
    """
    Loads UrbanSim outputs into memory as Orca tables. These are then
    manipulated and updated into the format required by ActivitySim.
    """
    hdf = pd.HDFStore(
        os.path.join(data_dir, settings['usim_data_store']))
    households = hdf['/households']
    persons = hdf['/persons']
    blocks = hdf['/blocks']
    jobs = hdf['/jobs']
    hdf.close()

    # add home x,y coords to persons table
    persons_w_res_blk = pd.merge(
        persons, households[['block_id']],
        left_on='household_id', right_index=True)
    persons_w_xy = pd.merge(
        persons_w_res_blk, blocks[['x', 'y']],
        left_on='block_id', right_index=True)
    persons['home_x'] = persons_w_xy['x']
    persons['home_y'] = persons_w_xy['y']

    del persons_w_res_blk
    del persons_w_xy

    orca.add_table('usim_households', households)
    orca.add_table('usim_persons', persons)
    orca.add_table('blocks', blocks)
    orca.add_table('jobs', jobs)


# Export households tables
@inject.step()
def create_inputs_from_usim_data(data_dir):

    persons_table = os.path.exists(os.path.join(data_dir, "persons.csv"))
    households_table = os.path.exists(os.path.join(data_dir, "households.csv"))
    land_use_table = os.path.exists(os.path.join(data_dir, "land_use.csv"))

    # if the input tables don't exist yet, create them from urbansim data
    if not persons_table & households_table & land_use_table:
        logger.info("Creating inputs from UrbanSim data")

        # create households input table
        hh_names_dict = {
            'household_id': 'HHID',
            'persons': 'PERSONS',
            'cars': 'VEHICL',
            'member_id': 'PNUM'}

        usim_households = orca.get_table('usim_households')
        hh_df = usim_households.to_frame().rename(columns=hh_names_dict)
        hh_df = hh_df[~hh_df.TAZ.isnull()]
        hh_df.to_csv(os.path.join(data_dir, 'households.csv'))
        del hh_df

        # create persons input table
        usim_persons = orca.get_table('usim_persons').to_frame()
        p_names_dict = {'member_id': 'PNUM'}
        p_df = usim_persons.rename(columns=p_names_dict)
        p_df = p_df[~p_df.TAZ.isnull()]
        p_df.sort_values('household_id', inplace=True)
        p_df.reset_index(drop=True, inplace=True)
        p_df.index.name = 'person_id'
        p_df.to_csv(os.path.join(data_dir, 'persons.csv'))
        del p_df

        # create land use input table
        zones = orca.get_table('zones')
        lu_df = zones.to_frame()
        lu_df.to_csv(os.path.join(data_dir, 'land_use.csv'))
        del lu_df

    else:
        logger.info("Found existing input tables, no need to re-create.")
