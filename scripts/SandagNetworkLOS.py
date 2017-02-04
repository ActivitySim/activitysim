
#Convert SANDAG network los files to ActivitySim NetworkLOS format
#Ben Stabler, ben.stabler@rsginc.com, 02/03/17

import sys, os.path, openmatrix
import pandas as pd, numpy as np

############################################################
# paramaters
############################################################

#settings
folder = "C:/projects/sandag-asim/toRSG/"
outputDataStoreFileName = "NetworkData.h5"
outputBikeLogsumMatrixFileName = "bikelogsum.omx"

if __name__== "__main__":
  
    #read CSVs and convert to NetworkLOS format
    #https://github.com/UDST/activitysim/wiki/Multiple-Zone-Systems-Design
    bikeMgraLogsum = pd.read_csv(folder + "bikeMgraLogsum.csv")
    walkMgraTapEquivMinutes = pd.read_csv(folder + "walkMgraTapEquivMinutes.csv")
    walkMgraEquivMinutes = pd.read_csv(folder + "walkMgraEquivMinutes.csv")
    mgra13_based_input2012 = pd.read_csv(folder + "mgra13_based_input2012.csv")
    Accessam = pd.read_csv(folder + "Accessam.csv")
    Tap_ptype = pd.read_csv(folder + "Tap_ptype.csv")
    Zone_term = pd.read_csv(folder + "Zone_term.csv")
    Zone_park = pd.read_csv(folder + "Zone_park.csv")

    #read taz and tap skim to get zone ids
    taz_skim = openmatrix.open_file("impdan_AM.omx")
    taz_numbers = taz_skim.mapping("Origin").keys() #keys() shouldn't be needed?
    tap_skim = openmatrix.open_file("implocl_AM.omx")
    tap_numbers = tap_skim.mapping("RCIndex").keys() #keys() shouldn't be needed?

    #convert tables
    TAP = pd.DataFrame({'TAP':tap_numbers})
    TAP = TAP.merge(Tap_ptype, how="outer")
    TAZ = pd.DataFrame({'TAZ':taz_numbers})
    TAZ = TAZ.merge(Zone_term, how="outer")
    TAZ = TAZ.merge(Zone_park, how="outer")
    MAZ = mgra13_based_input2012
    MAZ['MAZ'] = MAZ.mgra
    MAZ['TAZ'] = MAZ.taz
    del MAZ['mgra']
    del MAZ['taz']

    bikeMgraLogsum['OMAZ'] = bikeMgraLogsum.i
    bikeMgraLogsum['DMAZ'] = bikeMgraLogsum.j
    bikeMgraLogsum['MODE'] = 'BIKE'
    walkMgraEquivMinutes['OMAZ'] = walkMgraEquivMinutes.i
    walkMgraEquivMinutes['DMAZ'] = walkMgraEquivMinutes.j
    walkMgraEquivMinutes['MODE'] = 'WALK'
    MAZtoMAZ = pd.concat([bikeMgraLogsum, walkMgraEquivMinutes])
    del MAZtoMAZ['i']
    del MAZtoMAZ['j']

    walkMgraTapEquivMinutes['MAZ'] = walkMgraTapEquivMinutes.mgra
    walkMgraTapEquivMinutes['TAP'] = walkMgraTapEquivMinutes.tap
    walkMgraTapEquivMinutes['MODE'] = 'WALK'
    
    #expand from TAZtoTAP to MAZtoTAP
    tapsPerTaz = Accessam.groupby('TAZ').count()['TAP']
    Accessam.set_index('TAZ', drop=False, inplace=True)
    Accessam = Accessam.loc[MAZ.TAZ] #explode
    MAZ['TAPS'] = tapsPerTaz.loc[MAZ.TAZ].tolist()
    Accessam['MAZ'] = np.repeat(MAZ.MAZ.tolist(),MAZ.TAPS.tolist())
    Accessam['MODE'] = 'DRIVE'
    MAZtoTAP = pd.concat([walkMgraTapEquivMinutes, Accessam])
    del MAZtoTAP['mgra']
    del MAZtoTAP['tap']

    #write tables
    TAP.to_hdf(folder + outputDataStoreFileName, "TAP", complib='zlib',complevel=7)
    TAZ.to_hdf(folder + outputDataStoreFileName, "TAZ")
    MAZ.to_hdf(folder + outputDataStoreFileName, "MAZ")
    MAZtoMAZ.to_hdf(folder + outputDataStoreFileName, "MAZtoMAZ")
    MAZtoTAP.to_hdf(folder + outputDataStoreFileName, "MAZtoTAP")
    print("created " + folder + outputDataStoreFileName)

    #read bikeTazLogsum as convert to OMX
    bikeTazLogsum = pd.read_csv(folder + "bikeTazLogsum.csv")
    
    tazLookup = pd.DataFrame({"offset": range(len(taz_numbers)), "taz":taz_numbers})
    tazLookup.set_index("taz",drop=False, inplace=True)
    bikeTazLogsum['index_i'] = tazLookup.loc[bikeTazLogsum.i].offset.tolist()
    bikeTazLogsum['index_j'] = tazLookup.loc[bikeTazLogsum.j].offset.tolist()
    logsum = np.zeros([len(taz_numbers),len(taz_numbers)]) 
    time = np.zeros([len(taz_numbers),len(taz_numbers)])
    logsum[bikeTazLogsum['index_i'], bikeTazLogsum['index_j']] = bikeTazLogsum.logsum
    time[bikeTazLogsum['index_i'], bikeTazLogsum['index_j']] = bikeTazLogsum.time
    
    omxFile = openmatrix.open_file(folder + outputBikeLogsumMatrixFileName, "w")
    omxFile['logsum'] = logsum
    omxFile['time'] = time
    omxFile.createMapping('taz', taz_numbers)
    omxFile.close()
    print("created " + folder + outputBikeLogsumMatrixFileName)
    