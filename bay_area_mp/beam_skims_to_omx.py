import numpy as np
import pandas as pd
import openmatrix as omx


skims_df = pd.read_csv(
    's3://baus-data/spring_2019/30.skims-smart-23April2019-baseline.csv.gz')
num_hours = len(skims_df['hour'].unique())
num_modes = len(skims_df['mode'].unique())
num_od_pairs = len(skims_df) / num_hours / num_modes
num_taz = np.sqrt(num_od_pairs)
assert num_taz.is_integer()
num_taz = int(num_taz)

skims = omx.open_file('beam_skims.omx', 'w')


hwy_paths = ['SOV', 'HOV2', 'HOV3', 'SOVTOLL', 'HOV2TOLL', 'HOV3TOLL']
transit_modes = ['COM', 'EXP', 'HVY', 'LOC', 'LRF', 'TRN']
access_modes = ['WLK', 'DRV']
egress_modes = ['WLK']
active_modes = ['WALK', 'BIKE']
periods = ['EA', 'AM', 'MD', 'PM', 'EV']

# TO DO: fix bridge toll vs vehicle toll
beam_asim_hwy_measure_map = {
    'TIME': 'generalizedTimeInS',
    'DIST': 'distanceInM',
    'BTOLL': 'generalizedCost',
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
    'BOARDS': None  # transfers
}

# TO DO: get separate walk skims from beam so we don't just have to use
# bike distances for walk distances
for mode in active_modes:
    name = 'DIST{0}__'.format(mode)
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
            vals = tmp_df[beam_asim_hwy_measure_map[measure]].values
            mx = vals.reshape((num_taz, num_taz))
            skims[name] = mx

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



