
# create mtc tm1 asim example data
# Ben Stabler, ben.stabler@rsginc.com, 01/24/19
# run from the mtc tm1 skims folder

import pandas as pd

store = pd.HDFStore("mtc_asim.h5", "w")

col_map = {"HHID":"household_id","AGE":"age", "SEX":"sex", "hworkers":"workers", "HINC":"income", "AREATYPE":"area_type"}

df = pd.read_csv("../landuse/tazData.csv", index_col="ZONE")
df.columns = [col_map.get(s, s) for s in df.columns]
store["land_use/taz_data"] = df

df = pd.read_csv("accessibility.csv", index_col="taz")
df.columns = [col_map.get(s, s) for s in df.columns]
store["skims/accessibility"] = df

df = pd.read_csv("../popsyn/hhFile.csv", index_col="HHID")
df.columns = [col_map.get(s, s) for s in df.columns]
store["households"] = df

df = pd.read_csv("../popsyn/personFile.csv", index_col="PERID")
df.columns = [col_map.get(s, s) for s in df.columns]
store["persons"] = df
