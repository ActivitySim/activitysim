import pandas as pd
import numpy as np
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
from shapely import wkt
import orca

# ## Preprocessing the original MPO .shp files with TAZ
# Objective: Get area type as:
# - 0: Regional core
# - 1: CBD
# - 2: Urban Business
# - 3: Urban
# - 4: Suburban
# - 5: Rural

#Load MPO TAZs shapefiles
mpo_taz = gpd.read_file('data/tazs_austin/2015_2045 CAMPO TAZ SHAPE.shp')
mpo_taz = mpo_taz.to_crs('EPSG:4326')

#Transformation values:
mpo_taz = mpo_taz[mpo_taz.SMTDNAME != 'OutofArea']
area_type_dict = {'CBD': 1, 'UrbIntTravis': 2, 'UrbTravis': 3,
                  'SubTravis': 4, 'RurTravis':5,'UrbIntWilliamson': 2,
                  'UrbWilliamson': 3, 'SubWilliamson': 4,'RurWilliamson': 5,
                  'UrbIntHays': 2, 'UrbHays': 3, 'SubHays': 4, 'RurHays': 5,
                  'UrbIntBastrop': 2, 'UrbBastrop': 3, 'SubBastrop': 4,
                  'RurBastrop': 5, 'UrbCaldwell': 3, 'SubCaldwell': 4,
                  'RurCaldwell': 5, 'UrbBurnet': 3,'SubBurnet':4, 'RurBurnet':5}

mpo_taz['area_type'] = mpo_taz.SMTDNAME.replace(area_type_dict)


#Transform geopandas to dataframe
mpo_taz = pd.DataFrame(mpo_taz)
mpo_taz['geometry'] = mpo_taz.geometry.astype('str')

#Save it to .H5 file
hdf = pd.HDFStore('model_data.h5')
hdf.append(key = 'mpo_taz', value = mpo_taz)
hdf.close()
