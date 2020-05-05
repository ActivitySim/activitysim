#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


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

from platform import python_version
print('python version: ',python_version())

# austin data
hdf = pd.HDFStore('data/model_data.h5')
households = hdf['/households']
persons = hdf['/persons']
blocks = hdf['/blocks']
jobs = hdf['/jobs']
skims = pd.read_csv(
    'https://beam-outputs.s3.amazonaws.com/output/austin/'
    'austin-prod-200k-skims-with-h3-index-final__2020-04-18_09-44-24_wga/'
    'ITERS/it.0/0.skimsOD.UrbanSim.Full.csv.gz')

orca.add_table('households', households)
orca.add_table('persons', persons)
orca.add_table('blocks', blocks)
orca.add_table('jobs', jobs)
orca.add_table('skims', skims)


# In[6]:


@orca.table('zones', cache = True)
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


# In[7]:


@orca.table(cache= True)
def schools():
    base_url = 'https://educationdata.urban.org/api/v1/{topic}/{source}/{endpoint}/{year}/?{filters}'

#     county_codes = ids.block_id.str.slice(0,5).unique()
    county_codes = blocks.index.str.slice(0,5).unique()

    school_tables = []
    for county in county_codes:
        enroll_filters = 'county_code={0}'.format(county)
        enroll_url = base_url.format(topic='schools', source='ccd', endpoint='directory', 
                                     year='2015', filters=enroll_filters)

        enroll_result = requests.get(enroll_url)
        enroll = pd.DataFrame(enroll_result.json()['results'])
        school_tables.append(enroll)
        
    enrollment = pd.concat(school_tables, axis = 0)
    enrollment = enrollment[['ncessch','county_code','latitude','longitude', 'enrollment']].set_index('ncessch')
    return enrollment.dropna()


# In[8]:


@orca.table(cache= True)
def colleges():

    base_url = 'https://educationdata.urban.org/api/v1/{topic}/{source}/{endpoint}/{year}/?{filters}'
    county_codes = blocks.index.str.slice(0,5).unique()

    colleges_list = []
    for county in county_codes:
        college_filters = 'county_fips={0}'.format(county)
        college_url = base_url.format(topic='college-university', source='ipeds', endpoint='directory', 
                                             year='2015', filters=college_filters)

        college_result = requests.get(college_url)
        college = pd.DataFrame(college_result.json()['results'])
        colleges_list.append(college)

    colleges = pd.concat(colleges_list)
    colleges = colleges[['unitid', 'inst_name','longitude','latitude']].set_index('unitid')
    colleges.rename(columns = {'longitude':'x', 'latitude':'y'}, inplace = True)
    return colleges


# # Variables

# ### In Blocks table

# In[9]:


@orca.column('blocks', cache = True)
def TAZ(blocks, zones):
    #Tranform blocks to a Geopandas dataframe
    blocks_df = blocks.to_frame(columns = ['x', 'y'])
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])
    
    blocks_df = gpd.GeoDataFrame(blocks_df, geometry=gpd.points_from_xy(blocks_df.x, blocks_df.y))

    # Spatial join 
    blocks_df = gpd.sjoin(blocks_df, h3_gpd, how = 'left', op = 'intersects')

    #Drop duplicates and keep the one with the smallest H3 area
    blocks_df = blocks_df.sort_values('area')
    blocks_df.drop_duplicates(subset = ['x', 'y'], keep = 'first', inplace = True) 
    
    #Buffer unassigned blocks until they reach a hexbin. 
    null_blocks = blocks_df[blocks_df.index_right.isnull()].drop(columns = ['index_right','area'])

    result_list = []
    for index, block in null_blocks.iterrows():
        buff_size = 0.0001
        matched = False
        geo_block = gpd.GeoDataFrame(block).T
        while matched == False:
            geo_block.geometry = geo_block.geometry.buffer(buff_size)
            result = gpd.sjoin(geo_block, h3_gpd, how = 'left', op = 'intersects')
            matched = ~result.index_right.isnull()[0]
            buff_size = buff_size + 0.0001
        result_list.append(result.iloc[0:1])

    null_blocks = pd.concat(result_list)
    
    #Concatenate newly assigned blocks to the main blocks table 
    blocks_df = blocks_df.dropna()
    blocks_df = pd.concat([blocks_df, null_blocks], axis = 0)
    
    return blocks_df.index_right


# In[10]:


# %AREAS OF BLOCK BASED ON RESIDENTS AND EMPLOYEES PER BLOCK 
@orca.column('blocks')
def CI_employment(jobs):
    job = jobs.to_frame()
    job = job[job.sector_id.isin([11, 3133, 42, 4445, 4849, 52, 54, 7172])]
    s = job.groupby('block_id')['sector_id'].count()
    
    return s.reindex(blocks.index).fillna(0.01) #to avoid division by zero 
#best to have a relative greater number, so that dividing by this number results in a small value

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


# ### In Schools table

# In[11]:


@orca.column('schools', cache = True)
def TAZ(schools, zones):
    #Tranform blocks to a Geopandas dataframe
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])

    school_gpd = schools.to_frame(columns = ['ncessch','longitude', 'latitude'])
    school_gpd = gpd.GeoDataFrame(school_gpd, 
                                  geometry=gpd.points_from_xy(school_gpd.longitude, 
                                                              school_gpd.latitude))
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
        geo_school = gpd.GeoDataFrame(school).T
        while matched == False:
            geo_school.geometry = geo_school.geometry.buffer(buff_size)
            result = gpd.sjoin(geo_school, h3_gpd, how = 'left', op = 'intersects')
            matched = ~result.index_right.isnull().iloc[0]
            buff_size = buff_size + 0.0001
        result_list.append(result.iloc[0:1])

    null_school = pd.concat(result_list)
    
    #Concatenate newly assigned blocks to the main blocks table 
    school_gdf = school_gdf.dropna()
    school_all = pd.concat([school_gdf, null_school], axis = 0)
    school_all.set_index('ncessch', inplace = True)
    return school_all.index_right


# ### In colleges table

# In[12]:


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
    #Tranform blocks to a Geopandas dataframe
    colleges_df = colleges.to_frame(columns = ['x', 'y'])
    h3_gpd =  zones.to_frame(columns = ['geometry', 'area'])
    
    colleges_df = gpd.GeoDataFrame(colleges_df, geometry=gpd.points_from_xy(colleges_df.x, colleges_df.y))

    # Spatial join 
    colleges_df = gpd.sjoin(colleges_df, h3_gpd, how = 'left', op = 'intersects')

    #Drop duplicates and keep the one with the smallest H3 area
    colleges_df = colleges_df.sort_values('area')
    colleges_df.drop_duplicates(subset = ['x', 'y'], keep = 'first', inplace = True) 
    
    return colleges_df.index_right


# In[13]:


# # Plot of H3 bins with no block within it. 
# blocks = orca.get_table('blocks').to_frame()
# h3s = orca.get_table('h3_bins').to_frame()

# ids = blocks.h3_id.unique()
# h3_with_blocks = h3s[~h3s.h3_id.isin(ids)]

# ax  = h3s.plot(figsize = (10,10), alpha = 0.8, color = 'orange')
# h3_with_blocks.plot(figsize = (10,10), ax= ax, alpha = 0.5);


# ### In household table

# In[14]:


# Adding H3 bin to all tables
@orca.column('households')
def TAZ(blocks, households):
    return misc.reindex(blocks.TAZ, households.block_id)

@orca.column('households')
def HHT(households):
    s = households.persons
    return s.where(s==1, 4)


# ### In Persons table

# In[15]:


@orca.column('persons')
def TAZ(households, persons):
    return misc.reindex(households.TAZ, persons.household_id)

@orca.column('persons')
def ptype(persons):
    #Filters for person type segmentation 
    # https://activitysim.github.io/activitysim/abmexample.html#setup
    age_mask_1 = persons.age >= 18 
    age_mask_2 = persons.age.between(18, 64, inclusive = True)
    age_mask_3 = persons.age >= 65
    work_mask = persons.worker == 1
    student_mask = persons.student == 1

    #Series for each person segmentation 
    type_1 = ((age_mask_1) & (work_mask) & (~student_mask)) * 1 #Full time
    type_4 = ((age_mask_2) & (~work_mask) & (~student_mask)) * 4
    type_5 = ((age_mask_3) & (~work_mask) & (~student_mask)) * 5
    type_3 = ((age_mask_1) & (student_mask)) * 3
    type_6 = (persons.age.between(16, 17, inclusive = True))* 6
    type_7 = (persons.age.between(6, 16, inclusive = True))* 7
    type_8 = (persons.age.between(0, 5, inclusive = True))* 8 
    type_list = [type_1, type_3, type_4, type_5, type_6, type_7, type_8,]

    #Colapsing all series into one series
    for x in type_list:
        type_1.where(type_1 != 0, x, inplace = True)

    return type_1

@orca.column('persons')
def pemploy(persons):
    pemploy_1 = ((persons.worker == 1) & (persons.age >= 16)) * 1
    pemploy_3 = ((persons.worker == 0) & (persons.age >= 16)) * 3
    pemploy_4 = (persons.age < 16) * 4

    #Colapsing all series into one series
    type_list = [pemploy_1, pemploy_3, pemploy_4]
    for x in type_list:
        pemploy_1.where(pemploy_1 != 0, x, inplace = True)

    return pemploy_1

@orca.column('persons')
def pstudent(persons):
    pstudent_1 = (persons.age <= 18) * 1
    pstudent_2 = ((persons.student == 1) & (persons.age > 18)) * 2
    pstudent_3 = (persons.student == 0) * 3

    #Colapsing all series into one series
    type_list = [pstudent_1, pstudent_2, pstudent_3]
    for x in type_list:
        pstudent_1.where(pstudent_1 != 0, x, inplace = True)

    return pstudent_1


# ### In Jobs table

# In[16]:


@orca.column('jobs')
def TAZ(blocks, jobs):
    return misc.reindex(blocks.TAZ, jobs.block_id)


# ### In Zones table

# In[17]:


@orca.column('zones', cache=True)
def TOTHH(households, zones):
    s = households.TAZ.groupby(households.TAZ).count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def HHPOP(persons, zones):
    s = persons.TAZ.groupby(persons.TAZ).count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def EMPRES(households, zones):
    s = households.to_frame().groupby('TAZ')['workers'].sum()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def HHINCQ1(households, zones):
    df = households.to_frame()
    df = df[df.income < 30000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def HHINCQ2(households, zones):
    df = households.to_frame()
    df = df[df.income.between(30000, 59999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def HHINCQ3(households, zones):
    df = households.to_frame()
    df = df[df.income .between(60000, 99999)]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def HHINCQ4(households, zones):
    df = households.to_frame()
    df = df[df.income >= 100000]
    s = df.groupby('TAZ')['income'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE0004(persons, zones):
    df = persons.to_frame()
    df = df[df.age.between(0,4)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE0519(persons, zones):
    df = persons.to_frame()
    df = df[df.age.between(5,19)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE2044(persons, zones):
    df = persons.to_frame()
    df = df[df.age.between(20,44)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE4564(persons, zones):
    df = persons.to_frame()
    df = df[df.age.between(45,64)]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE65P(persons, zones):
    df = persons.to_frame()
    df = df[df.age >= 65]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def AGE62P(persons, zones):
    df = persons.to_frame()
    df = df[df.age >= 62]
    s = df.groupby('TAZ')['age'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def SHPOP62P(zones):
    return (zones.AGE62P/zones.HHPOP).reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def TOTEMP(jobs, zones):
    s = jobs.TAZ.groupby(jobs.TAZ).count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def RETEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([4445])] #difference is here (44, 45 vs 4445)## sector ids don't match
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def FPSEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([52,54])]
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
    df = df[df.sector_id.isin([42, 3133, 32, 4849])]## sector ids don't match
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def OTHEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[~df.sector_id.isin([4445, 52, 54, 61, 62, 
                              71, 11, 42, 3133, 32, 4849])] ## sector ids don't match
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def TOTACRE(zones):
    g = zones.geometry.to_crs({'init': 'epsg:3857'}) #area in square meters
    area_polygons = g.area/4046.86
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
    return 1 #assumes everything is flat


# missing columns

# In[18]:


#Parking variables
@orca.column('zones')
def employment_density(zones):
    return zones.TOTEMP/zones.TOTACRE

@orca.column('zones')
def pop_density(zones):
    return zones.HHPOP/zones.TOTACRE

@orca.column('zones')
def hh_density(zones):
    return zones.TOTHH/zones.TOTACRE

@orca.column('zones')
def hq1_density(zones):
    return zones.HHINCQ1/zones.TOTACRE

@orca.column('zones')
def PRKCST(zones):
    params = pd.Series([-1.92168743,  4.89511403,  4.2772001 ,  0.65784643], 
                       index = ['pop_density', 'hh_density', 'hq1_density', 'employment_density'])
    
    cols = zones.to_frame(columns = ['employment_density', 'pop_density', 'hh_density', 'hq1_density'])
    
    s = cols @ params
    return s.where(s>0, 0)

@orca.column('zones')
def OPRKCST(zones):
    params = pd.Series([-6.17833544, 17.55155703,  2.0786466 ], 
                       index = ['pop_density', 'hh_density', 'employment_density'])
    
    cols = zones.to_frame(columns = ['employment_density', 'pop_density', 'hh_density'])
    
    s = cols @ params
    return s.where(s>0, 0)


# In[19]:


@orca.column('zones') # College enrollment 
def COLLFTE(colleges, zones):
    s = colleges.to_frame().groupby('TAZ')['full_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones') # College enrollment 
def COLLPTE(colleges, zones):
    s = colleges.to_frame().groupby('TAZ')['part_time_enrollment'].sum()
    return s.reindex(zones.index).fillna(0)


# In[20]:


@orca.column('zones')
def area_type():
#     Integer, 0=regional core, 1=central business district, 2=urban business, 3=urban, 4=suburban, 5=rural
    return 0 #Assuming all regional core

@orca.column('zones')
def TERMINAL():
    #TO DO: 
    #Improve the imputation of this variable
    # Average time to travel from automobile storage location to origin/destination
    # We assume zero for now
    return 0 #Assuming O

@orca.column('zones')
def COUNTY():
    #TO DO: 
    #County variable (approximate to Bay area characteristics )
    return 1 #Assuming 1 all San Francisco County

# ## Skims info

# In[21]:


num_hours = len(skims['hour'].unique())
num_modes = len(skims['mode'].unique())
num_od_pairs = len(skims) / num_hours / num_modes
num_taz = np.sqrt(num_od_pairs)
assert num_taz.is_integer()
num_taz = int(num_taz)

hwy_paths = ['SOV', 'HOV2', 'HOV3', 'SOVTOLL', 'HOV2TOLL', 'HOV3TOLL']
transit_modes = ['COM', 'EXP', 'HVY', 'LOC', 'LRF', 'TRN']
access_modes = ['WLK', 'DRV']
egress_modes = ['WLK', 'DRV']
active_modes = ['WALK', 'BIKE']
periods = ['EA', 'AM', 'MD', 'PM', 'EV']

# TO DO: fix bridge toll vs vehicle toll
beam_asim_hwy_measure_map = {
    'TIME': 'gen_cost_min',
    'DIST': 'dist_miles',
    'BTOLL': None,
    'VTOLL': 'generalizedCost'}

# TO DO: get actual values here
beam_asim_transit_measure_map = {
    'WAIT': None,  # other wait time?
    'TOTIVT': 'generalizedTimeInS',  # total in-vehicle time
    'KEYIVT': None,  # light rail IVT
    'FERRYIVT': None,  # ferry IVT
    'FAR': 'generalizedCost',  # fare
    'DTIM': None,  # drive time
    'DDIST': None,  # drive dist
    'WAUX': None,  # walk other time
    'WEGR': None,  # walk egress time
    'WACC': None,  # walk access time
    'IWAIT': None,  # iwait?
    'XWAIT': None,  # transfer wait time
    'BOARDS': None,  # transfers
    'IVT':'generalizedTimeInS' #In vehicle travel time
    }




# ## Steps

# In[28]:


@orca.step()
def households_table(households):
    names_dict = {'household_id': 'HHID',
                  'persons': 'PERSONS', 
                  'cars': 'VEHICL', 
                  'member_id': 'PNUM'}
    
    df = households.to_frame().rename(columns = names_dict)
    df = df[~df.TAZ.isnull()]
    df.to_csv('data/households.csv')


# In[29]:


@orca.step()
def persons_table(persons):
    names_dict = {'member_id': 'PNUM'}
    df = persons.to_frame().rename(columns = names_dict)
    df = df[~df.TAZ.isnull()]
    df.to_csv('data/persons.csv')


# In[30]:


@orca.step()
def land_use_table(zones):
    df = orca.get_table('zones').to_frame()
    df.to_csv('data/land_use.csv')


# In[31]:


@orca.step()
def skims_omx(skims):
    
    skims_df = skims.to_frame()
    skims_df['dist_miles'] = skims_df['distanceInM']*(0.621371/1000)
    skims_df['gen_cost_min'] = skims_df['generalizedTimeInS']/(60)

    skims = omx.open_file('data/skims.omx', 'w')
    # TO DO: get separate walk skims from beam so we don't just have to use
    # bike distances for walk distances
    
    #Adding distance
    tmp_df = skims_df[(skims_df['mode'] == 'CAR')]
    vals = tmp_df[beam_asim_hwy_measure_map['DIST']].values
    mx = vals.reshape((num_taz, num_taz))
    skims['DIST'] = mx

    for mode in active_modes:
        name = 'DIST{0}'.format(mode)
        tmp_df = skims_df[(skims_df['mode'] == 'BIKE')]
        vals = tmp_df[beam_asim_hwy_measure_map['DIST']].values
        mx = vals.reshape((num_taz, num_taz))
        skims[name] = mx
    
    for period in periods:
        df = skims_df

        # highway skims
        for path in hwy_paths:
            tmp_df = df[(df['mode'] == 'CAR')]
            for measure in beam_asim_hwy_measure_map.keys():
                name = '{0}_{1}__{2}'.format(path, measure, period)
                if beam_asim_hwy_measure_map[measure]:
                    vals = tmp_df[beam_asim_hwy_measure_map[measure]].values
                    mx = vals.reshape((num_taz, num_taz))
                else:
                    mx = np.zeros((num_taz, num_taz))
                skims[name] = mx
#                 vals = tmp_df[beam_asim_hwy_measure_map[measure]].values
#                 mx = vals.reshape((num_taz, num_taz))

        # transit skims
        for transit_mode in transit_modes:
            for access_mode in access_modes:
                for egress_mode in egress_modes:
                    path = '{0}_{1}_{2}'.format(
                        access_mode, transit_mode, egress_mode)
                    for measure in beam_asim_transit_measure_map.keys():
                        name = '{0}_{1}__{2}'.format(path, measure, period)

                        # TO DO: something better than setting zero-ing out
                        # all skim values we don't have
                        if beam_asim_transit_measure_map[measure]:
                            vals = tmp_df[beam_asim_transit_measure_map[measure]].values
                            mx = vals.reshape((num_taz, num_taz))
                        else:
                            mx = np.zeros((num_taz, num_taz))
                        skims[name] = mx


# In[32]:


orca.run(['households_table','persons_table','land_use_table','skims_omx'])