import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import orca
from shapely.geometry import Polygon
from h3 import h3
from urbansim.utils import misc
import requests
import openmatrix as omx
from shapely import wkt
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


def assign_taz(df, gdf):
    '''
    Assigns the gdf index (TAZ ID) for each index in df
    Input: 
    - df columns names x, and y. The index is the ID of the object(blocks, school, college)
    - gdf: Geopandas DataFrame with TAZ as index, geometry and area value. 
    Output:
    A series with df index and corresponding gdf id
    '''

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs = "EPSG:4326")
    gdf.geometry.crs = "EPSG:4326"
    
    assert df.geometry.crs == gdf.geometry.crs

    # Spatial join 
    df = gpd.sjoin(df, gdf, how = 'left', op = 'intersects')

    #Drop duplicates and keep the one with the smallest H3 area
    df = df.sort_values('area')
    index_name = df.index.name
    df.reset_index(inplace = True)
    df.drop_duplicates(subset = [index_name], keep = 'first', inplace = True) 
    df.set_index(index_name, inplace = True)
    
    #Check if there is any assigined object
    if df.index_right.isnull().sum()>0:
    
        #Buffer unassigned ids until they reach a hexbin. 
        null_values = df[df.index_right.isnull()].drop(columns = ['index_right','area'])

        result_list = []
        for index, value in null_values.iterrows():
            buff_size = 0.0001
            matched = False
            geo_value = gpd.GeoDataFrame(value).T
            geo_value.crs = "EPSG:4326"
            while matched == False:
                geo_value.geometry = geo_value.geometry.buffer(buff_size)
                result = gpd.sjoin(geo_value, gdf, how = 'left', op = 'intersects')
                matched = ~result.index_right.isnull()[0]
                buff_size = buff_size + 0.0001
            result_list.append(result.iloc[0:1])

        null_values = pd.concat(result_list)

        # Concatenate newly assigned values to the main values table 
        df = df.dropna()
        df = pd.concat([df, null_values], axis = 0)

        return df.index_right
    
    else:
        return df.index_right


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
    enrollment.rename(columns = {'longitude':'x', 'latitude':'y'}, inplace = True)
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

@orca.column('blocks', cache=True)
def TAZ(blocks, zones):
    blocks_df = blocks.to_frame(columns = ['x', 'y'])
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])
    return assign_taz(blocks_df, h3_gpd)


@orca.column('blocks')
def TOTEMP(blocks, jobs):

    return jobs.to_frame().groupby('block_id')['TAZ'].count().reindex(blocks.index).fillna(0)


@orca.column('blocks')
def TOTPOP(blocks, usim_households):
    hh = usim_households.to_frame()
    return hh.groupby('block_id')['persons'].sum().reindex(blocks.index).fillna(0)


@orca.column('blocks')
def TOTACRE(blocks):
    return blocks['square_meters_land'] / 4046.86


@orca.column('blocks')
def area_type_metric(blocks):

    # we calculate the metric at the block level because h3 zones are so variable
    # in size, so this size-dependent metric is very susceptible to the
    # modifiable areal unit problem. Since blocks are 
    return ((1 * blocks['TOTPOP']) + (2.5 * blocks['TOTEMP'])) / blocks['TOTACRE']


@orca.column('blocks')
def CI_employment(jobs, blocks):
    job = jobs.to_frame()
    job = job[job.sector_id.isin([11, 3133, 42, 4445, 4849, 52, 54, 7172])]
    s = job.groupby('block_id')['sector_id'].count()

    # to avoid division by zero best to have a relative greater number,
    # so that dividing by this number results in a small value
    return s.reindex(blocks.index).fillna(0.01) 


# School Variables

@orca.column('schools', cache = True)
def TAZ(schools, zones):
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])
    school_gpd = orca.get_table('schools').to_frame(columns = ['x', 'y'])
    return assign_taz(school_gpd, h3_gpd)

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


@orca.column('colleges', cache = True)
def TAZ(colleges, zones):
    colleges_df = colleges.to_frame(columns = ['x', 'y'])
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])
    return assign_taz(colleges_df, h3_gpd)


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

    # project to meter-based crs
    g = zones.geometry.to_crs({'init': 'epsg:2768'})
    
    # square meters to acres
    area_polygons = g.area / 4046.86
    return area_polygons


# @orca.column('zones', cache=True)
# def RESACRE(blocks, zones):
#     df = blocks.to_frame()
#     s = df.groupby('TAZ')['RESACRE'].sum()
#     return s.reindex(zones.index).fillna(0)


# @orca.column('zones', cache=True)
# def CIACRE(blocks, zones):
#     df = blocks.to_frame()
#     s = df.groupby('TAZ')['CIACRE'].sum()
#     return s.reindex(zones.index).fillna(0)


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
def area_type_metric(blocks, zones):

    # because of the MAUP, we have to aggregate the area_type_metric values
    # up from the block level to the h3 zone. instead of a simple average,
    # we us a weighted average of the blocks based on the square root of the
    # sum of the number of jobs and residents of each block. this method
    # was found to produce the best results in the Austin metro region compared
    # to the simple average (underclassified urban areas) and the fully
    # weighted average (overclassified too many CBDs).

    # it is probably a good idea to visually assess the accuracy of the
    # metric when implementing in a new region.

    blocks_df = blocks.to_frame(columns=['TAZ', 'TOTPOP', 'TOTEMP', 'area_type_metric'])
    blocks_df['weight'] = np.round(np.sqrt(blocks_df['TOTPOP'] + blocks_df['TOTEMP']))
    blocks_weighted = blocks_df.loc[blocks_df.index.repeat(blocks_df['weight'])]
    area_type_avg = blocks_weighted.groupby('TAZ')['area_type_metric'].mean()
    return area_type_avg.reindex(zones.index).fillna(0)


@orca.column('zones')
def area_type(zones):
    # Integer, 0=regional core, 1=central business district,
    # 2=urban business, 3=urban, 4=suburban, 5=rural
    area_types = pd.cut(
        zones['area_type_metric'],
        [0, 6, 30, 55, 100, 300, float("inf")],
        labels=['5', '4', '3', '2', '1', '0'],
        include_lowest=True
    ).astype(str) 
    return area_types


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
    # mpo_taz = hdf['/mpo_taz']
    
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
