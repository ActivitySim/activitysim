import os
import openmatrix as omx
import pandas as pd
import numpy as np
import logging

from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger(__name__)

# ActivitySim Skims Variables
hwy_paths = ['SOV', 'HOV2', 'HOV3', 'SOVTOLL', 'HOV2TOLL', 'HOV3TOLL']
transit_modes = ['COM', 'EXP', 'HVY', 'LOC', 'LRF', 'TRN']
access_modes = ['WLK', 'DRV']
egress_modes = ['WLK', 'DRV']
active_modes = ['WALK', 'BIKE']
periods = ['EA', 'AM', 'MD', 'PM', 'EV']

# Map ActivitySim skim measures to input skims
beam_asim_hwy_measure_map = {
    'TIME': 'generalizedTimeInM',  # must be minutes
    'DIST': 'distanceInMi',  # must be miles
    'BTOLL': None,
    'VTOLL': 'generalizedCost'}

beam_asim_transit_measure_map = {
    'WAIT': None,  # other wait time?
    'TOTIVT': 'generalizedTimeInM',  # total in-vehicle time (minutes)
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
    'IVT': 'generalizedTimeInM'  # In vehicle travel time (minutes)
}


@inject.table()
def raw_beam_skims():

    # load skims from url
    beam_skims_url = config.setting('beam_skims_url')
    skims = pd.read_csv(beam_skims_url)

    return skims


# for use in initialize_inputs_from_usim
@inject.injectable()
def h3_zone_ids(raw_beam_skims):
    return raw_beam_skims.origTaz.unique()


@inject.step()
def create_skims_from_beam(raw_beam_skims, data_dir):
    skims_path = os.path.join(data_dir, 'skims.omx')
    skims_exist = os.path.exists(skims_path)

    if skims_exist:
        logger.info("Found existing skims, no need to re-create.")

    else:
        logger.info("Creating skims.omx from BEAM skims")
        skims_df = raw_beam_skims.to_frame()

        # figure out the size of the skim matrices
        num_hours = len(skims_df['hour'].unique())
        num_modes = len(skims_df['mode'].unique())
        num_od_pairs = len(skims_df) / num_hours / num_modes

        # make sure the matrix is square
        num_taz = np.sqrt(num_od_pairs)
        assert num_taz.is_integer()

        num_taz = int(num_taz)

        # convert beam skims to activitysim units (miles and minutes)
        skims_df['distanceInMi'] = skims_df['distanceInM'] * (0.621371 / 1000)
        skims_df['generalizedTimeInM'] = skims_df['generalizedTimeInS'] / (60)

        skims = omx.open_file(skims_path, 'w')

        
        # break out skims by mode
        auto_df = skims_df[(skims_df['mode'] == 'CAR')]
        active_df = skims_df[(skims_df['mode'] == 'BIKE')]
        transit_df = skims_df[(skims_df['mode'] == 'WALK_TRANSIT')]
        # activitysim estimated its models using transit skims from Cube
        # which store time values as scaled integers (e.g. x100), so their
        # models also divide transit skim values by 100. Since our skims
        # aren't coming out of Cube, we multiply by 100 to negate the division. 
        transit_df['generalizedTimeInM'] = transit_df['generalizedTimeInM'] * 100

        # Adding car distance skims
        vals = auto_df[beam_asim_hwy_measure_map['DIST']].values
        mx = vals.reshape((num_taz, num_taz))
        skims['DIST'] = mx

        # active skims
        for mode in active_modes:

            # TO DO: get separate walk skims from beam so we don't
            # just have to use bike distances for walk distances
            name = 'DIST{0}'.format(mode)
            
            vals = active_df[beam_asim_hwy_measure_map['DIST']].values
            mx = vals.reshape((num_taz, num_taz))
            skims[name] = mx

        for period in periods:

            # highway skims
            for path in hwy_paths:
                for measure in beam_asim_hwy_measure_map.keys():
                    name = '{0}_{1}__{2}'.format(path, measure, period)
                    if beam_asim_hwy_measure_map[measure]:
                        vals = auto_df[beam_asim_hwy_measure_map[measure]].values
                        mx = vals.reshape((num_taz, num_taz))
                    else:
                        mx = np.zeros((num_taz, num_taz))
                    skims[name] = mx

            # transit skims
            for transit_mode in transit_modes:
                for access_mode in access_modes:
                    for egress_mode in egress_modes:
                        path = '{0}_{1}_{2}'.format(
                            access_mode, transit_mode, egress_mode)
                        for measure in beam_asim_transit_measure_map.keys():
                            name = '{0}_{1}__{2}'.format(path, measure, period)

                            # TO DO: something better than zero-ing out
                            # all skim values we don't have
                            if beam_asim_transit_measure_map[measure]:

                                vals = transit_df[
                                    beam_asim_transit_measure_map[measure]].values

                                mx = vals.reshape((num_taz, num_taz))
                            else:
                                mx = np.zeros((num_taz, num_taz))
                            skims[name] = mx
        skims.close()
