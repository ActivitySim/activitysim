import pandas as pd 
import numpy as np 
import orca 
import time 
from urbansim.utils import misc
import geopandas as gpd
import os;
import warnings
warnings.filterwarnings('ignore')



# Importing data  
zones = pd.read_csv('s3://baus-data/spring_2019/zones_shp.csv', index_col='TAZ')


parcels = pd.read_csv('s3://baus-data/spring_2019/parcels.csv', 
                      index_col='primary_id', dtype={
                    'primary_id': int, 'block_id': str, 'apn': str})

jobs = pd.read_csv('s3://baus-data/spring_2019/jobs.csv',index_col = 'job_id')
buildings = pd.read_csv('s3://baus-data/spring_2019/buildings.csv',index_col = 'building_id')
units = pd.read_csv('s3://baus-data/spring_2019/units.csv',index_col = 'unit_id')
households = pd.read_csv('s3://baus-data/spring_2019/households.csv',index_col = 'household_id')
persons = pd.read_csv('s3://baus-data/spring_2019/persons.csv',index_col = 'person_id')
lu_mtc = pd.read_csv('s3://baus-data/spring_2019/land_use_mtc.csv', index_col = 'TAZ')

# zones = zones[['taz1454','district', 'county', 'gacres']]
# zones.columns = ['TAZ','district', 'county', 'TOTACRE']
# zones.set_index('TAZ', inplace = True)
parcels.index.rename('parcel_id', inplace = True)

# #Sampling to make it faster 
# buildings = buildings.sample(1000)
# parcels = parcels[parcels.index.isin(buildings.parcel_id)]
# units = units[units.building_id.isin(buildings.index)]
# households = households[households.unit_id.isin(units.index)]
# persons = persons[persons.household_id.isin(households.index)]
# jobs = jobs[jobs.building_id.isin(buildings.index)]


orca.add_table('zones', zones)
orca.add_table('parcels', parcels)
orca.add_table('buildings', buildings)
orca.add_table('units', units)
orca.add_table('households', households)
orca.add_table('persons', persons)
orca.add_table('jobs', jobs)
orca.add_table('lu_mtc', lu_mtc);


# Adding TAZ to all tables
@orca.column('buildings')
def TAZ(parcels, buildings):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('units')
def TAZ(buildings, units):
    return misc.reindex(buildings.TAZ, units.building_id)

@orca.column('households')
def TAZ(units, households):
    return misc.reindex(units.TAZ, households.unit_id)

@orca.column('persons')
def TAZ(households, persons):
    return misc.reindex(households.TAZ, persons.household_id)

# Adding home coordinates to persons
@orca.column('buildings')
def x(parcels, buildings):
    return misc.reindex(parcels.x, buildings.parcel_id)

@orca.column('buildings')
def y(parcels, buildings):
    return misc.reindex(parcels.y, buildings.parcel_id)

@orca.column('units')
def x(buildings, units):
    return misc.reindex(buildings.x, units.building_id)

@orca.column('units')
def y(buildings, units):
    return misc.reindex(buildings.y, units.building_id)

@orca.column('households')
def home_x(units, households):
    return misc.reindex(units.x, households.unit_id)

@orca.column('households')
def home_y(units, households):
    return misc.reindex(units.y, households.unit_id)


@orca.column('persons')
def home_x(households, persons):
    return misc.reindex(households.home_x, persons.household_id)

@orca.column('persons')
def home_y(households, persons):
    return misc.reindex(households.home_y, persons.household_id)

#Adding columsn to the zones table
@orca.column('zones')
def COUNTY(zones):
    county_dict = {'San Francisco': 1, 'San Mateo': 2, 'Santa Clara': 3, 
                   'Alameda': 4, 'Contra Costa': 5, 'Solano': 6, 
                   'Napa': 7, 'Sonoma': 8, 'Marin': 9}
    return zones.county.replace(county_dict)

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

@orca.column('jobs')
def TAZ(buildings, jobs):
    return misc.reindex(buildings.TAZ, jobs.building_id)

@orca.column('zones', cache=True)
def TOTEMP(jobs, zones):
    s = jobs.TAZ.groupby(jobs.TAZ).count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def RETEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[df.sector_id.isin([44,45])]
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
    df = df[df.sector_id.isin([42, 31, 32, 33, 48, 49])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def OTHEMPN(jobs, zones):
    df = jobs.to_frame()
    df = df[~df.sector_id.isin([44,45, 52, 54, 61, 62, 
                              71, 11, 42, 31, 32, 33, 48, 49])]
    s = df.groupby('TAZ')['sector_id'].count()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def RESACRE(parcels, zones):
    df = parcels.to_frame()
    df = df[df.development_type_id.isin([1, 2, 5])]
    s = df.groupby('zone_id')['acres'].sum()
    return s.reindex(zones.index).fillna(0)

@orca.column('zones', cache=True)
def CIACRE(parcels, zones):
    df = parcels.to_frame()
    df = df[df.development_type_id.isin([7, 9, 10, 13, 14,15])]
    s = df.groupby('zone_id')['acres'].sum()
    return s.reindex(zones.index).fillna(0)

# This columns are copied from the MTC land use file. 
# - TO DO: Figure out a way to calculate this values with our data

# MTC columns
@orca.column('zones')
def PRKCST(lu_mtc):
    return lu_mtc.PRKCST

@orca.column('zones')
def OPRKCST(lu_mtc):
    return lu_mtc.OPRKCST

@orca.column('zones')
def area_type(lu_mtc):
    return lu_mtc.area_type

@orca.column('zones')
def HSENROLL(lu_mtc):
    return lu_mtc.HSENROLL

@orca.column('zones')
def COLLFTE(lu_mtc):
    return lu_mtc.COLLFTE

@orca.column('zones')
def COLLPTE(lu_mtc):
    return lu_mtc.COLLPTE

@orca.column('zones')
def TERMINAL(lu_mtc):
    return lu_mtc.TERMINAL

@orca.column('zones')
def TOPOLOGY(lu_mtc):
    return lu_mtc.TOPOLOGY

@orca.step()
def households_table(households):

    @orca.column('households')
    def HHT(households):
        return households.single_family.replace({True: 4, False: 1})
    
    names_dict = {'household_id': 'HHID','persons': 'PERSONS', 'cars': 'VEHICL', 'member_id': 'PNUM'}
    
    df = households.to_frame().rename(columns = names_dict)
    df = df[~df.TAZ.isnull()]
    
    orca.add_table('households', df)

@orca.step()
def persons_table(persons):

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
    
    names_dict = {'member_id': 'PNUM'}
    
    df = persons.to_frame().rename(columns = names_dict)
    df = df[~df.TAZ.isnull()]
    
    orca.add_table('persons', df)

#Running steps
orca.run(['households_table'])
orca.run(['persons_table'])

#Exporting tables
zones = orca.get_table('zones').to_frame()

#Check the buildings table for this discrepanties
# Zones with 0 CI >> This is just an easy fix for now
missing_TAZ_CIACRE = zones[zones.CIACRE == 0]['CIACRE'].index
zones.loc[missing_TAZ_CIACRE,'CIACRE'] = lu_mtc.loc[missing_TAZ_CIACRE,'CIACRE']

zones.to_csv(os.getcwd() + '/data/land_use.csv')

households = orca.get_table('households').to_frame()
households.to_csv(os.getcwd() +'/data/households.csv')

persons = orca.get_table('persons').to_frame()
persons.to_csv(os.getcwd() +'/data/persons.csv')