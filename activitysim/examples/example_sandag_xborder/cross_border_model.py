import pandas as pd
import numpy as np
import os
import sys
import yaml
from collections import OrderedDict
import openmatrix as omx
import argparse
import subprocess
import itertools


def compute_poe_accessibility(
        poe_id, colonias, colonia_pop_field, distance_param):

    pop = colonias[colonia_pop_field]
    dist_col = 'Distance_poe' + str(int(poe_id))
    dists = colonias[dist_col]
    dist_factors = np.exp(dists * distance_param)
    weighted_pop = dist_factors * pop
    total_poe_pop_access = np.log(weighted_pop.sum())

    return total_poe_pop_access


def get_poe_wait_times(settings):

    data_dir = settings['data_dir']
    wait_times = pd.read_csv(
        os.path.join(data_dir, settings['poe_wait_times_input_fname']))
    num_poes = wait_times.poe.nunique()
    wait_times.rename(columns={
        'StandardWait': 'std_wait', 'SENTRIWait': 'sentri_wait',
        'PedestrianWait':'ped_wait'}, inplace=True)
    start_hour_mask = wait_times['StartPeriod'] > 16
    wait_times.loc[start_hour_mask, 'StartHour'] = wait_times.loc[
        start_hour_mask, 'StartHour'] + 12
    end_hour_mask = (
        wait_times['StartPeriod'] > 14) & (wait_times['StartPeriod'] < 40)
    wait_times.loc[end_hour_mask, 'EndHour'] = wait_times.loc[
        end_hour_mask, 'EndHour'] + 12
    wait_times['StartHour'].replace(24, 0, inplace=True)
    wait_times = wait_times.sort_values(['poe', 'StartHour'])

    # full enumeration by the hour
    wait_times['num_hours'] = (
        wait_times['EndHour'] - wait_times['StartHour']) * 2
    wait_times = wait_times.loc[
    wait_times.index.repeat(wait_times['num_hours'])]
    wait_times['asim_start_period'] = np.tile(
        np.array(range(1, 49)), num_poes)

    # pivot wide
    wait_times_wide = wait_times.pivot(
        index='poe', columns='asim_start_period',
        values=['std_wait','sentri_wait','ped_wait'])
    wait_times_wide.columns = [
        '_'.join([top, str(bottom)])
        for top, bottom in wait_times_wide.columns]

    return wait_times_wide


def create_tours(settings):

    tour_settings = settings['tours']
    crossing_type_dict = {0: 'non_sentri', 1: 'sentri'}
    num_tours = tour_settings['num_tours']
    sentri_share = tour_settings['sentri_share']
    lane_shares_by_purpose = tour_settings['lane_shares_by_purpose']
    purpose_probs_by_sentri = tour_settings['purpose_shares']
    
    tours = pd.DataFrame(
        index=range(num_tours), columns=[
            'sentri_crossing', 'lane_type', 'lane_id', 'tour_type',
            'purpose_id'])
    tours.index.name = 'tour_id'

    sentri_scaled_probs = np.subtract(
        np.array(list([sentri_share])), np.random.rand(num_tours, 1))
    sentri = (sentri_scaled_probs + 1).astype('i4')
    
    tours['sentri_crossing'] = sentri
    for sentri, group in tours.groupby('sentri_crossing'):
        
        num_xing_type_tours = len(group)
        crossing_type = crossing_type_dict[sentri]
        purpose_probs = OrderedDict(purpose_probs_by_sentri[crossing_type])
        
        # scale probs to so they sum to 1
        prob_sum = sum(purpose_probs.values())
        purpose_probs = {k: v / prob_sum for k,v in purpose_probs.items()}

        id_to_purpose = {
            i: purpose for i, purpose in enumerate(purpose_probs.keys())}
        purpose_cum_probs = np.array(list(purpose_probs.values())).cumsum()
        purpose_scaled_probs = np.subtract(
            purpose_cum_probs, np.random.rand(num_xing_type_tours, 1))
        purpose = np.argmax((purpose_scaled_probs + 1.0).astype('i4'), axis=1)
        group['purpose_id'] = purpose
        tours.loc[group.index, 'purpose_id'] = purpose

    tours['tour_type'] = tours['purpose_id'].map(id_to_purpose)
    tours['tour_category'] = 'non_mandatory'
    tours.loc[tours['tour_type'].isin(
        ['work', 'school']), 'tour_category'] = 'mandatory'
    
    # for xborder model, only 1 person per tour and 1 tour per person
    tours['number_of_participants'] = 1
    tours['tour_num'] = 1
    tours['tour_count'] = 1

    for purpose, df in tours.groupby('tour_type'):
        lane_probs = OrderedDict(lane_shares_by_purpose[purpose])
        id_to_lane = {i: lane for i, lane in enumerate(lane_probs.keys())}
        lane_cum_probs = np.array(list(lane_probs.values())).cumsum()
        lane_scaled_probs = np.subtract(
            lane_cum_probs, np.random.rand(len(df), 1))
        lane_id = np.argmax((lane_scaled_probs + 1.0).astype('i4'), axis=1)
        df['lane_id'] = lane_id
        df['lane_type'] = df['lane_id'].map(id_to_lane)
        tours.loc[df.index, 'lane_id'] = df['lane_id']
        tours.loc[df.index, 'lane_type'] = df['lane_type']

    return tours


def create_households(settings):

    tour_settings = settings['tours']
    num_tours = tour_settings['num_tours']

    # one household per tour
    households = pd.DataFrame({'household_id': range(num_tours)})

    return households


def create_persons(settings, num_households):

    # one person per household
    persons = pd.DataFrame({'person_id': range(num_households)})
    persons['household_id'] = np.random.choice(
        num_households, num_households, replace=False)

    return persons


def _update_sparse_skims(
    settings, data_dir, filter_col=None, filter_list=None, new_mazs=None):

    input_fname = settings['input_fname']
    output_fname = settings['output_fname']
    filter_col = settings.get('filter_col', filter_col)
    
    df = pd.read_csv(os.path.join(data_dir, input_fname))

    # rename columns from spec
    df.rename(columns=settings['rename_columns'], inplace=True)

    # filter columns from spec
    if filter_col:
        if filter_list is None:
            raise ValueError("filter_list param needed to filter table.")
        df = df.loc[df[filter_col].isin(filter_list)]

    # create new rows for new MAZs based on original MAZ counterparts 
    cols_to_match = ['OMAZ', 'DMAZ', 'MAZ']
    if new_mazs is not None:
        for i, row in new_mazs.iterrows():
            original_maz = row['original_MAZ']
            new_maz_id = row['MAZ']
            for col in cols_to_match:
                if col in df.columns:
                    new_rows = df[df[col] == original_maz].copy()
                    new_rows[col] = new_maz_id
                    df = df.append(new_rows, ignore_index=True)

    df.to_csv(os.path.join(data_dir, output_fname), index=False)

    return


def _rename_skims(settings, skim_type):
    
    data_dir = settings['data_dir']
    skims_settings = settings['skims'][skim_type]
    periods = skims_settings.get('periods', [None])
    walk_speed = settings['walk_speed']

    for period in periods:
        
        if period:
            print('Processing {0} {1} skims.'.format(period, skim_type))
            input_base_fname = skims_settings['input_base_fname']
            output_base_fname = skims_settings['output_base_fname']
            input_fname = input_base_fname + '_' + period + '.omx'
            output_fname = output_base_fname + '_' + period + '.omx'
        else:
            print('Processing {0} skims.'.format(skim_type))
            input_fname = skims_settings['input_fname']
            output_fname = skims_settings['output_fname']

        skims = omx.open_file(
            os.path.join(data_dir, input_fname), 'a')
        skims.copy_file(
            os.path.join(data_dir, output_fname), overwrite=True)
        output_skims = omx.open_file(
            os.path.join(data_dir, output_fname), 'a')

        for skims_name in output_skims.list_matrices():
            name_elems = skims_name.split('_')
            new_name = '_'.join(name_elems[1:]) + '__' + name_elems[0]
            output_skims[skims_name].rename(new_name)

        if period == 'MD':
            output_skims.create_matrix(
                'walkTime', obj=output_skims['SOV_NT_M_DIST__MD'][:, :])
            output_skims['walkTime'][:, :] = output_skims[
                'walkTime'][:, :] / walk_speed * 60

        skims.close()
        output_skims.close()

    return


def create_taps_tap_lines(settings):
    
    skims_settings = settings['skims']
    data_dir = settings['data_dir']

    transit_skims = omx.open_file(os.path.join(
        data_dir, skims_settings['tap_to_tap']['input_fname']), 'a')
    all_taps = pd.DataFrame(
        pd.Series(list(transit_skims.root.lookup.zone_number)))
    all_taps.columns = ['TAP']
    all_taps.to_csv(
        os.path.join(data_dir, settings['taps_output_fname']), index=False)
    tap_lines = pd.read_csv(
        os.path.join(data_dir, settings['tap_lines_input_fname']))
    tap_lines.to_csv(
        os.path.join(data_dir, settings['tap_lines_output_fname'])) 

    transit_skims.close()

    return


def create_skims_and_tap_files(settings, new_mazs=None):

    skims_settings = settings['skims']
    data_dir = settings['data_dir']

    # create taps files and transit skims
    print('Creating tap files.')
    transit_skims = omx.open_file(os.path.join(
        data_dir, skims_settings['tap_to_tap']['input_fname']), 'a')
    all_taps = pd.DataFrame(
        pd.Series(list(transit_skims.root.lookup.zone_number)))
    all_taps.columns = ['TAP']
    all_taps.to_csv(
        os.path.join(data_dir, settings['taps_output_fname']), index=False)
    tap_lines = pd.read_csv(
        os.path.join(data_dir, settings['tap_lines_input_fname']))
    tap_lines.to_csv(
        os.path.join(data_dir, settings['tap_lines_output_fname']))
    transit_skims.close()
    
    
    # update skims/network data
    print('Updating maz-to-maz skims.')
    _update_sparse_skims(
        skims_settings['maz_to_maz']['walk'],
        data_dir=settings['data_dir'],
        new_mazs=new_mazs)
    
    print('Updating maz-to-tap skims.')
    _update_sparse_skims(
        skims_settings['maz_to_tap']['walk'],
        data_dir=settings['data_dir'],
        filter_col='TAP',
        filter_list=all_taps.TAP.values,
        new_mazs=new_mazs)
    
    # rename transit and auto skims
    print('Renaming skim keys')
    for skim_type in ['tap_to_tap', 'taz_to_taz']:
        _rename_skims(settings, skim_type)


def create_stop_freq_specs(settings):

    probs_df = pd.read_csv(os.path.join(
        settings['data_dir'], settings['stop_frequency_input_fname']))

    # drop cargo
    if probs_df['Purpose'].max() == 5:
        probs_df = probs_df.loc[probs_df['Purpose'] != 2, :]
        probs_df.loc[probs_df['Purpose'] > 2, 'Purpose'] = probs_df.loc[
            probs_df['Purpose'] > 2, 'Purpose'] - 1

    probs_df.rename(columns={'Outbound': 'out', 'Inbound': 'in'}, inplace=True)
    probs_df['alt'] = probs_df['out'].astype(str) + 'out_' + \
        probs_df['in'].astype(str) + 'in'

    # convert probs to utils
    probs_df['value'] = np.log(probs_df['Percent']).clip(lower=-999)
    
    # write out alts table
    alts_df = probs_df.drop_duplicates(['out','in','alt'])[['alt','out','in']]
    alts_df.to_csv(os.path.join(
        settings['config_dir'], settings['stop_frequency_alts_output_fname']),
        index=False)

    purpose_id_map = settings['tours']['purpose_ids']

    # iterate through purposes and pivot probability lookup tables to
    # create MNL spec files with only ASC's (no covariates).
    required_cols = ['Label', 'Description', 'Expression']
    for purpose, purpose_id in purpose_id_map.items():
        purpose_probs = probs_df.loc[probs_df['Purpose'] == purpose_id, :].copy()
        purpose_probs['coefficient_name'] = purpose_probs.apply(
            lambda x: 'coef_asc_dur_{0}_{1}_stops_{2}'.format(
                x['DurationLo'],x['DurationHi'],x['alt']), axis=1)

        coeffs_file = purpose_probs[['value','coefficient_name']].copy()
        coeffs_file['Description'] = None
        coeffs_file = coeffs_file[['Description','value','coefficient_name']]
        coeffs_file_fname = settings[
            'stop_frequency_coeffs_output_formattable_fname'].format(
            purpose=purpose)
        coeffs_file.to_csv(
            os.path.join(settings['config_dir'], coeffs_file_fname),
            index=False)

        alt_cols = alts_df['alt'].tolist()
        expr_file =  purpose_probs.pivot(
            index=['DurationLo','DurationHi'], columns='alt',
            values='coefficient_name').reset_index()
        expr_file['Label'] = 'util_ASC_tour_dur_' + \
            expr_file['DurationLo'].astype(str) + \
            '_' + expr_file['DurationHi'].astype(str)
        expr_file['Description'] = 'ASC for tour durations between ' + \
            expr_file['DurationLo'].astype(str) + ' and ' + \
            expr_file['DurationHi'].astype(str)
        expr_file['Expression'] = expr_file['DurationLo'].astype(str) + \
            ' < duration_hours <= ' + expr_file['DurationHi'].astype(str)
        expr_file = expr_file.drop(columns=['DurationLo','DurationHi'])
        expr_file = expr_file[required_cols + [
            col for col in expr_file.columns if col not in required_cols]]
        expr_file_fname = settings[
            'stop_frequency_expressions_output_formattable_fname'].format(
            purpose=purpose)
        expr_file.to_csv(os.path.join(
            settings['config_dir'], expr_file_fname.format(purpose)),
            index=False)

    return


def update_trip_purpose_probs(settings):

    probs_df = pd.read_csv(os.path.join(
        settings['data_dir'], settings['trip_purpose_probs_input_fname']))
    purpose_id_map = settings['tours']['purpose_ids']
    purp_id_to_name = {v: k for k, v in purpose_id_map.items()}

    # drop cargo
    if probs_df['TourPurp'].max() == 5:
        probs_df = probs_df.loc[probs_df['TourPurp'] != 2, :]
        del probs_df['StopPurp2']
        probs_df.rename(columns={
            'StopPurp3': 'StopPurp2',
            'StopPurp4': 'StopPurp3',
            'StopPurp5': 'StopPurp4'}, inplace=True)
        probs_df.loc[probs_df['TourPurp'] > 2, 'TourPurp'] = probs_df.loc[
            probs_df['TourPurp'] > 2, 'TourPurp'] - 1

    probs_df.rename(columns={
        'StopNum': 'trip_num', 'Multiple': 'multistop',
        'StopPurp0': purp_id_to_name[0], 'StopPurp1': purp_id_to_name[1],
        'StopPurp2': purp_id_to_name[2], 'StopPurp3': purp_id_to_name[3],
        'StopPurp4': purp_id_to_name[4]}, inplace=True)
    probs_df['outbound'] = ~probs_df['Inbound'].astype(bool)
    probs_df['primary_purpose'] = probs_df['TourPurp'].map(purp_id_to_name)
    probs_df = probs_df[
        ['primary_purpose','outbound', 'trip_num', 'multistop'] + [
        purp for purp in purpose_id_map.keys()]]
    probs_df.to_csv(os.path.join(
        settings['config_dir'], settings['trip_purpose_probs_output_fname']),
        index=False)

    return


def create_trip_scheduling_duration_probs(settings, los_settings):

    data_dir = settings['data_dir']
    config_dir = settings['config_dir']
    period_settings = los_settings['skim_time_periods']
    num_asim_periods = int(
        period_settings['time_window'] / period_settings['period_minutes'])

    outbound = pd.read_csv(os.path.join(
        data_dir,
        settings['trip_scheduling_probs_input_fnames']['outbound']))
    outbound['outbound'] = True
    inbound = pd.read_csv(os.path.join(
        data_dir,
        settings['trip_scheduling_probs_input_fnames']['inbound']))
    inbound['outbound'] = False
    assert len(outbound) == len(inbound)

    outbound = outbound.melt(
        id_vars=['RemainingLow','RemainingHigh','Stop','outbound'],
        var_name='duration_offset', value_name='prob')
    inbound = inbound.melt(
        id_vars=['RemainingLow','RemainingHigh','Stop','outbound'],
        var_name='duration_offset', value_name='prob')
    
    duration_probs = pd.concat((outbound, inbound), axis=0, ignore_index=True)
    duration_probs.rename(columns={
        'Stop': 'stop_num',
        'RemainingHigh': 'periods_left_max',
        'RemainingLow': 'periods_left_min'}, inplace=True)

    # convert to asim format
    max_periods_left = duration_probs['periods_left_max'].max()
    duration_probs['periods_left_max'] = duration_probs[
        'periods_left_max'].replace(
            max_periods_left, num_asim_periods - 1)

    duration_probs.to_csv(os.path.join(
        config_dir, settings['trip_scheduling_probs_output_fname']),
        index=False)

    return


def update_tour_purpose_reassignment_probs(settings):

    data_dir = settings['data_dir']
    config_dir = settings['config_dir']
    tour_settings = settings['tours']
    poes = [str(poe) for poe in list(settings['poes'].keys())]

    probs_df = pd.read_csv(os.path.join(
        data_dir, settings['tour_purpose_control_probs_input_fname']), 
        index_col=['Description', 'Purpose'])
    probs_df = probs_df.loc[:, poes]
    probs_df.reset_index(inplace=True)
    probs_df['Description'] = probs_df['Description'].str.lower()
    
    # drop cargo
    probs_df = probs_df.loc[probs_df['Description'] != 'cargo', :]
    probs_df['Purpose'] = probs_df['Description'].map(
        tour_settings['purpose_ids'])

    # adjust probs to ensure they sum to 1
    for poe in poes:
        probs_df[poe] = np.round(probs_df[poe] / probs_df[poe].sum(), 3)

    probs_df.to_csv(
        os.path.join(
            config_dir,
            settings['tour_purpose_control_probs_output_fname']),
        index=False)

    return


def create_land_use_file(settings):

    print('Creating land use (maz) table.')

    data_dir = settings['data_dir']
    maz_input_fname = settings['maz_input_fname']
    maz_id_field = settings['maz_id_field']
    poe_id_field = settings['poe_id_field']
    poe_access_field = settings['poe_access_field']
    colonia_input_fname = settings['colonia_input_fname']
    colonia_pop_field = settings['colonia_pop_field']
    distance_param = settings['distance_param']

    # load maz/mgra + colonia data
    colonias = pd.read_csv(os.path.join(data_dir, colonia_input_fname))
    mazs = pd.read_csv(os.path.join(data_dir, maz_input_fname))

    
    mazs[poe_id_field] = -1
    mazs['original_MAZ'] = -1
    mazs['external_TAZ'] = -1
    mazs['external_MAZ'] = -1
    for poe_id, poe_attrs in settings['poes'].items():

        # get poe id for maz's that have one
        maz_mask = mazs[maz_id_field] == poe_attrs['maz_id']
        mazs.loc[maz_mask, poe_id_field] = poe_id
        mazs.loc[maz_mask, 'external_TAZ'] = poe_attrs['ext_taz_id']

        # add poes as new mazs
        row = mazs.loc[maz_mask].copy()
        for col in row.columns:
            if pd.api.types.is_float_dtype(row[col]):
                row[col] = 0.0
            elif pd.api.types.is_integer_dtype(row[col]):
                row[col] = 0
            else:
                row[col] = None
        row[maz_id_field] = mazs[maz_id_field].max() + 1
        row['external_TAZ'] = -1
        row['external_MAZ'] = -1
        mazs.loc[maz_mask, 'external_MAZ'] = row[maz_id_field]
        row['taz'] = poe_attrs['ext_taz_id']
        row[poe_id_field] = poe_id
        row['original_MAZ'] = mazs.loc[maz_mask, 'mgra']
        mazs = mazs.append(row, ignore_index=True)

    # compute colonia accessibility for poe mazs
    mazs[poe_access_field] = None
    poe_mask = mazs[poe_id_field] >= 0
    mazs.loc[poe_mask, poe_access_field] = \
        mazs.loc[poe_mask, poe_id_field].apply(
            compute_poe_accessibility, colonias=colonias,
            colonia_pop_field=colonia_pop_field,
            distance_param=distance_param)
    mazs = mazs.rename(columns={'mgra': 'MAZ', 'taz': 'TAZ'})

    # merge in wide wait times
    wide_wait_times = get_poe_wait_times(settings)
    mazs = pd.merge(
        mazs, wide_wait_times, left_on='poe_id', right_on='poe', how='left')

    return mazs


def create_scheduling_probs_and_alts(settings, los_settings):

    config_dir = settings['config_dir']
    data_dir = settings['data_dir']
    input_fname = settings['tour_scheduling_probs_input_fname']
    output_probs_fname = settings['tour_scheduling_probs_output_fname']
    output_alts_fname = settings['tour_scheduling_alts_output_fname']

    # load ctramp probs
    scheduling_probs = pd.read_csv(os.path.join(data_dir, input_fname))
    scheduling_probs.rename(columns={
        'Purpose': 'purpose_id', 'EntryPeriod': 'entry_period',
        'ReturnPeriod': 'return_period', 'Percent': 'prob'}, inplace=True)

    # drop cargo tours
    scheduling_probs = scheduling_probs.loc[
        scheduling_probs['purpose_id'] != 2]
    update_mask = scheduling_probs['purpose_id'] > 2
    scheduling_probs.loc[update_mask, 'purpose_id'] = scheduling_probs.loc[
        update_mask, 'purpose_id'] - 1

    # vars we'll need
    num_ctramp_periods = int(scheduling_probs['entry_period'].nunique())
    max_period = scheduling_probs['entry_period'].max()
    assert num_ctramp_periods == max_period
    num_purposes = int(scheduling_probs['purpose_id'].nunique())

    # create tour scheduling alts 
    period_settings = los_settings['skim_time_periods']
    num_periods = int(
        period_settings['time_window'] / period_settings['period_minutes'])
    tour_scheduling_alts = pd.DataFrame()
    tour_scheduling_alts['start'] = np.tile(
        range(1, num_periods + 1), num_periods)
    tour_scheduling_alts['end'] = np.repeat(
        range(1, num_periods + 1), num_periods)

    # ctramp allows tours to return after midnight b/c period 40 = 12-3am
    # asim periods cannot accommodate these tours, but we can't set these probs
    # to zero or else our tour purpose probs won't sum to one. Instead we
    # force tours that start before 12am and end after 12am to return in the
    # period just prior to midnight (39), and redistribute these probabilities
    # accordingly. For tours that start *and* end after midnight, we simply
    # set the entry and return periods to 0, which will later get expanded
    # to create asim periods 1-6. At this point, the table is missing rows
    # where the return period is 0 and the entry period is > 0. All of these
    # missing rows will have zero probability, but we still need them for the
    # table to be complete. The last step of the process is to create a
    # dataframe of these missing rows and merge them back into the probability
    # table.
    trunc_tours_mask = (
        scheduling_probs['return_period'] == max_period) & (
        scheduling_probs['entry_period'] < max_period)
    scheduling_probs.loc[trunc_tours_mask, 'return_period'] = max_period - 1
    scheduling_probs['entry_period'].replace(
        num_ctramp_periods, 0, inplace=True)
    scheduling_probs['return_period'].replace(
        num_ctramp_periods, 0, inplace=True) 
    scheduling_probs = scheduling_probs.groupby(
        ['purpose_id', 'entry_period', 'return_period'])[
        'prob'].sum().reset_index()
    missing_df = pd.DataFrame({
        'purpose_id': np.repeat(
            range(0, num_purposes), num_ctramp_periods - 1),
        'entry_period': np.tile(range(1, num_ctramp_periods), num_purposes),
        'return_period': [0] * (num_ctramp_periods - 1) * num_purposes})
    missing_df['prob'] = 0
    assert len(missing_df) + len(scheduling_probs) == (
        num_ctramp_periods**2) * num_purposes
    scheduling_probs = pd.concat(
        (scheduling_probs, missing_df), ignore_index=True)
    scheduling_probs = scheduling_probs.sort_values(
        ['purpose_id','return_period','entry_period']).reset_index(drop=True)
    assert scheduling_probs['prob'].sum() == num_purposes

    # compute expansion factor for reach row based on the ctramp period. this
    # will be used to convert 40-period probabilities to 48-period probs.
    scheduling_probs['half_hour_entry_periods'] = 1
    scheduling_probs.loc[
        scheduling_probs['entry_period'] == 1, 'half_hour_entry_periods'] = 4 
    scheduling_probs.loc[
        scheduling_probs['entry_period'] == 0, 'half_hour_entry_periods'] = 6
    scheduling_probs['half_hour_return_periods'] = 1
    scheduling_probs.loc[
        scheduling_probs['return_period'] == 1, 'half_hour_return_periods'] = 4 
    scheduling_probs.loc[
        scheduling_probs['return_period'] == 0, 'half_hour_return_periods'] = 6

    # When expanding the probabilities from 40 to 48 periods, we need to divide
    # the probabilities by the number of times a given row was repeated during
    # expansion. Otherwise, the probs will sum to a number greater than 1 for
    # each tour purpose. However, we will also need to force probabilities to
    # zero wherever the entry period is later than the return period. These
    # zero prob. rows should not count towards the number of repeats used for
    # dividing the probability, otherwise the probabilities will sum to a 
    # number *less* than 1 for each tour purpose. To arrive at the correct 
    # probability divisor, we must compute the number of return periods that
    # are earlier than their entry periods, and subtract this number from the
    # total number of repeats. Surely there must be a better way to do this :(
    asim_scheduling_probs = scheduling_probs.copy()
    asim_scheduling_probs['max_entry_period'] = asim_scheduling_probs[
        'entry_period'] + 9
    asim_scheduling_probs['min_entry_period'] = asim_scheduling_probs[
        'entry_period'] + 9
    asim_scheduling_probs.loc[
        asim_scheduling_probs['entry_period'] == 0, 'max_entry_period'] = 6
    asim_scheduling_probs.loc[
        asim_scheduling_probs['entry_period'] == 0, 'min_entry_period'] = 1
    asim_scheduling_probs.loc[
        asim_scheduling_probs['entry_period'] == 1, 'max_entry_period'] = 10
    asim_scheduling_probs.loc[
        asim_scheduling_probs['entry_period'] == 1, 'min_entry_period'] = 7
    asim_scheduling_probs['max_return_period'] = asim_scheduling_probs[
        'return_period'] + 9
    asim_scheduling_probs['min_return_period'] = asim_scheduling_probs[
        'return_period'] + 9
    asim_scheduling_probs.loc[
        asim_scheduling_probs['return_period'] == 0, 'max_return_period'] = 6
    asim_scheduling_probs.loc[
        asim_scheduling_probs['return_period'] == 0, 'min_return_period'] = 1
    asim_scheduling_probs.loc[
        asim_scheduling_probs['return_period'] == 1, 'max_return_period'] = 10
    asim_scheduling_probs.loc[
        asim_scheduling_probs['return_period'] == 1, 'min_return_period'] = 7
    asim_scheduling_probs['num_returns_lt_entry'] = \
        asim_scheduling_probs.apply(lambda x: sum([
            i > j for i, j in itertools.product(
            range(
                int(x['min_entry_period']), int(x['max_entry_period']) + 1),
            range(
                int(x['min_return_period']), int(x['max_return_period']) + 1))
        ]), axis=1)
    asim_scheduling_probs['repeats'] = asim_scheduling_probs[
        'half_hour_return_periods'] * asim_scheduling_probs[
        'half_hour_entry_periods']
    asim_scheduling_probs['prob_divisor'] = asim_scheduling_probs[
        'repeats'] - asim_scheduling_probs['num_returns_lt_entry']

    # expand entry periods
    asim_scheduling_probs = asim_scheduling_probs.loc[
        scheduling_probs.index.repeat(
            scheduling_probs['half_hour_entry_periods'])].reset_index(drop=True)
    asim_scheduling_probs['asim_entry_period'] = np.tile(
        range(1, num_periods + 1), num_ctramp_periods * num_purposes)

    # expand return periods
    asim_scheduling_probs = asim_scheduling_probs.loc[
        asim_scheduling_probs.index.repeat(
            asim_scheduling_probs['half_hour_return_periods'])].reset_index(drop=True)
    asim_scheduling_probs = asim_scheduling_probs.sort_values(
        ['purpose_id','asim_entry_period', 'return_period'])
    asim_scheduling_probs['asim_return_period'] = np.tile(
        range(1, num_periods + 1), num_periods * num_purposes)
    asim_scheduling_probs = asim_scheduling_probs.sort_values(
        ['purpose_id', 'asim_return_period', 'asim_entry_period'])
    assert len(asim_scheduling_probs) == (num_periods**2) * (num_purposes)

    # adjust probs
    asim_scheduling_probs['asim_prob'] = asim_scheduling_probs['prob']
    return_lt_entry_mask = asim_scheduling_probs[
        'asim_entry_period'] > asim_scheduling_probs['asim_return_period']
    asim_scheduling_probs.loc[return_lt_entry_mask, 'asim_prob'] = 0.0
    nonzero_divisor_mask = asim_scheduling_probs['prob_divisor'] > 0
    asim_scheduling_probs.loc[nonzero_divisor_mask, 'asim_prob'] = \
        asim_scheduling_probs.loc[
            nonzero_divisor_mask, 'asim_prob'] / asim_scheduling_probs.loc[
            nonzero_divisor_mask, 'prob_divisor']
    assert asim_scheduling_probs['asim_prob'].sum() == num_purposes


    # sanity check entry periods
    for asim_entry_period in range(1, 7):
        # asim periods 1-6 = ctramp period 0 (40)
        entry_mask = asim_scheduling_probs[
            'asim_entry_period'] == asim_entry_period
        assert all(asim_scheduling_probs.loc[entry_mask, 'entry_period'] == 0)
    for asim_entry_period in range(7, 11):
        entry_mask = asim_scheduling_probs[
        'asim_entry_period'] == asim_entry_period
        # asim periods 7-10 = ctramp period 1
        assert all(asim_scheduling_probs.loc[entry_mask, 'entry_period'] == 1)
    dif_9_mask = (asim_scheduling_probs['entry_period'] >= 2)
    pd.testing.assert_series_equal(
        asim_scheduling_probs.loc[dif_9_mask, 'entry_period'],
        asim_scheduling_probs.loc[dif_9_mask, 'asim_entry_period'] - 9,
        check_names=False, check_dtype=False
    )
    
    # sanity check return periods
    for asim_return_period in range(1, 7):
        return_mask = asim_scheduling_probs[
            'asim_return_period'] == asim_return_period
        # asim periods 1-6 = ctramp period 0 (40)
        assert all(
            asim_scheduling_probs.loc[return_mask, 'return_period'] == 0)
    for asim_return_period in range(7, 11):
        return_mask = asim_scheduling_probs[
            'asim_return_period'] == asim_return_period
        # asim periods 7-10 = ctramp period 1
        assert all(
            asim_scheduling_probs.loc[return_mask, 'return_period'] == 1)
    dif_9_mask = (asim_scheduling_probs['return_period'] >= 2)
    pd.testing.assert_series_equal(
        asim_scheduling_probs.loc[dif_9_mask, 'return_period'],
        asim_scheduling_probs.loc[dif_9_mask, 'asim_return_period'] - 9,
        check_names=False, check_dtype=False)
    
    # pivot wide
    asim_scheduling_probs = asim_scheduling_probs.pivot(
        index='purpose_id',
        columns=['asim_entry_period', 'asim_return_period'],
        values='asim_prob')
    asim_scheduling_probs.columns = [
        str(col[0]) + '_' + str(col[1])
        for col in asim_scheduling_probs.columns]

    asim_scheduling_probs.to_csv(os.path.join(config_dir, output_probs_fname))
    tour_scheduling_alts.to_csv(
        os.path.join(config_dir, output_alts_fname), index=False)

    return


def assign_hh_p_to_tours(tours, persons):

    num_tours = len(tours)

    # assign persons and households to tours
    tours['household_id'] = np.random.choice(
        num_tours, num_tours, replace=False)
    tours['person_id'] = persons.set_index('household_id').reindex(
        tours['household_id'])['person_id'].values

    return tours


if __name__ == '__main__':

    # runtime args
    parser = argparse.ArgumentParser(prog='preprocessor')
    parser.add_argument(
        '-w', '--wait_times',
        action='store_true', help='Update POE wait times.')
    parser.add_argument(
        '-p', '--preprocess',
        action='store_true', help='Run preprocessor.')
    parser.add_argument(
        '-a', '--asim',
        action='store_true', help='Run activitysim.')
    
    args = parser.parse_args()
    update_wait_times = args.wait_times
    run_preprocessor = args.preprocess
    run_asim = args.asim

    # load settings
    with open('configs/preprocessing.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        data_dir = settings['data_dir']
        config_dir = settings['config_dir']
    with open(os.path.join(config_dir, 'network_los.yaml')) as f:
        los_settings = yaml.load(f, Loader=yaml.FullLoader)

    if run_preprocessor:
        print('RUNNING PREPROCESSOR!')

        # create input data
        mazs = create_land_use_file(settings)
        new_mazs = mazs[mazs['original_MAZ'] > 0]
        tours = create_tours(settings)
        households = create_households(settings)  # 1 per tour
        persons = create_persons(settings, num_households=len(households))
        tours = assign_hh_p_to_tours(tours, persons)

        # # store input files to disk
        mazs.to_csv(os.path.join(
            data_dir, settings['mazs_output_fname']), index=False)
        tours.to_csv(os.path.join(
            data_dir, settings['tours_output_fname']))
        households.to_csv(os.path.join(
            data_dir, settings['households_output_fname']), index=False)
        persons.to_csv(os.path.join(
            data_dir, settings['persons_output_fname']), index=False)

        # create/update configs in place
        create_scheduling_probs_and_alts(settings, los_settings)
        create_skims_and_tap_files(settings, new_mazs)
        create_stop_freq_specs(settings)
        update_trip_purpose_probs(settings)
        create_trip_scheduling_duration_probs(settings, los_settings)
        update_tour_purpose_reassignment_probs(settings)

    if update_wait_times:

        # load settings
        wait_time_settings = settings['wait_time_updating']
        period_settings = los_settings['skim_time_periods']

        # instantiate data matrix
        num_periods = int(
            period_settings['time_window'] / period_settings['period_minutes'])
        periods = list(range(1, num_periods + 1))
        poes = list(settings['poes'].keys())
        num_poes = len(poes)
        lane_types = list(
            settings['tours']['lane_shares_by_purpose']['work'].keys())
        num_lanes = len(lane_types)
        num_obs = num_poes * num_lanes * num_periods
        x_df = pd.DataFrame({
            'poe_id': np.repeat(poes, num_lanes * num_periods),
            'lane_type': np.tile(np.repeat(lane_types, num_periods), num_poes),
            'start': np.tile(periods, num_poes * num_lanes)
        })

        # coefficients
        coef_df = pd.DataFrame(wait_time_settings['coeffs']).T
        assert len(coef_df) == num_lanes
        
        # load existing wait times from last iteration
        mazs = pd.read_csv(
            os.path.join(data_dir, settings['mazs_output_fname']))
        wait_times_wide = mazs.sort_values('poe_id').loc[mazs['MAZ'].isin(
            [poe['maz_id'] for poe_id, poe in settings['poes'].items()]),
            [col for col in mazs.columns if 'wait' in col] + 
            ['poe_id']].set_index('poe_id')
        
        # ctramp inputs didn't have ready lanes so we must
        # create them and mark them as unavailable.
        if not any(['ready' in col for col in wait_times_wide.columns]):
            ready_cols = wait_times_wide[[
                col for col in wait_times_wide.columns if 'std' in col]].copy()
            ready_cols.columns = [
                col.replace('std', 'ready') for col in ready_cols.columns]
            ready_cols[:] = 999
            wait_times_wide = pd.concat((wait_times_wide, ready_cols), axis=1)

        all_wait_times = [wait_times_wide]
        all_vol_dfs = []
        
        num_iters = wait_time_settings['iters'] + 1
        for i in range(1, num_iters):

            print('UPDATING POE WAIT TIMES: ITER {0}'.format(i))
            process = subprocess.Popen([
                    'python', '-u', 'simulation.py', '--settings_file',
                    'wait_time_mode.yaml'],
                stdout=sys.stdout, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.SubprocessError(stderr.decode())
            
            # compute crossing volume from tour POEs
            tours = pd.read_csv('./output/wait_time_tours.csv')

            vol_df = tours.groupby(['poe_id','lane_type','start']).agg(
                vol=('tour_id', 'count')).reset_index()

            # get missing rows and set vol to 0 for them
            vol_df = pd.merge(x_df, vol_df, how='left').fillna(0)
            vol_df['iter'] = i
            
            # compute vol per lane  
            lane_df = pd.DataFrame(settings['poes']).T[[
                'name','veh_lanes','ped_lanes']]
            vol_df = vol_df.merge(
                lane_df, left_on='poe_id', right_index=True, how='left')
            vol_df['vol_per_lane'] = vol_df['vol'] / vol_df['veh_lanes']
            ped_mask = vol_df['lane_type'] == 'ped'
            vol_df.loc[ped_mask, 'vol_per_lane'] = vol_df.loc[
                ped_mask, 'vol'] / vol_df.loc[ped_mask, 'ped_lanes']

            # compute dummies
            vol_df['otay'] = (vol_df['name'] == 'Otay Mesa').astype(int)
            vol_df['tecate'] = (vol_df['name'] == 'Tecate').astype(int)
            vol_df['EA'] = (vol_df['start'] <= 11).astype(int)
            vol_df['EV'] = (vol_df['start'] > 37).astype(int)

            # data matrix
            x = np.zeros((num_obs, len(coef_df.columns)))
            x[:, 0] = 1
            x[:, 1] = vol_df['otay']
            x[:, 2] = vol_df['tecate']
            x[:, 3] = vol_df['vol_per_lane']
            x[:, 4] = vol_df['vol_per_lane'] * vol_df['otay']
            x[:, 5] = vol_df['vol_per_lane'] * vol_df['tecate']
            x[:, 6] = vol_df['vol_per_lane'] * vol_df['EA']
            x[:, 7] = vol_df['vol_per_lane'] * vol_df['EV']
            x[:, 8] = vol_df['vol_per_lane'] * vol_df['otay'] * vol_df['EA']
            x[:, 9] = vol_df['vol_per_lane'] * vol_df['otay'] * vol_df['EV']
            x[:, 10] = vol_df['vol_per_lane'] * vol_df['tecate'] * vol_df['EV']

            # coefficient matrix
            W = coef_df.reindex(vol_df['lane_type'])

            # pair-wise multiply and sum across rows to get regression results
            vol_df['wait_time'] = np.sum(x * W, axis=1).values

            # reshape the results and apply min bound to 0 minutes
            wait_times_wide = vol_df.pivot(
                index=['poe_id'], columns=['lane_type', 'start'],
                values=['wait_time'])
            wait_times_wide.columns = [
                '_wait_'.join([lane_type, str(start)])
                for wt, lane_type, start in wait_times_wide.columns]
            wait_times_wide = wait_times_wide[all_wait_times[0].columns]
            wait_times_wide = wait_times_wide.where(wait_times_wide >= 0, 0)

            # method of successive averaging
            msa_frac = 1.0 / i
            last_iter_wait_times = all_wait_times[-1]
            assert last_iter_wait_times.shape == wait_times_wide.shape
            last_iter_wait_times_weighted = last_iter_wait_times * (
                1.0 - msa_frac)
            wait_times_wide_weighted = wait_times_wide * msa_frac
            new_wait_times_wide = last_iter_wait_times_weighted.add(
                wait_times_wide_weighted)

            # replace averaged value with latest value where last iter had
            # nulls bc we don't want to average null values
            new_wait_times_wide = new_wait_times_wide.where(
                last_iter_wait_times != 999, wait_times_wide)

            # some wait times must be hard-coded as nulls (999) to indicate
            # unavailable lane types. 
            # tecate has no sentri lane and no ready lane, and is only open
            # from 5am (period 11) to 11pm (period 47)
            unavail_tecate_cols = [
                col for col in new_wait_times_wide.columns if
                ('sentri' in col) or
                ('ready' in col) or
                (int(col.split('_')[-1]) < 11) or
                ((int(col.split('_')[-1]) > 46))]

            # otay mesa east has to ped lane
            unavail_om_east_cols = [
                col for col in new_wait_times_wide.columns if 'ped' in col]

            # jacumba has no sentri lane and no ped lane
            unavail_jacumba_cols = [
                col for col in new_wait_times_wide.columns if
                ('ped' in col) or ('sentri' in col)]

            if 2 in new_wait_times_wide.index.values:
                new_wait_times_wide.loc[2, unavail_tecate_cols] = 999
            
            if 3 in new_wait_times_wide.index.values:
                new_wait_times_wide.loc[3, unavail_om_east_cols] = 999
            
            if 4 in new_wait_times_wide.index.values:
                new_wait_times_wide.loc[4, unavail_jacumba_cols] = 999

            all_vol_dfs.append(vol_df)
            all_wait_times.append(new_wait_times_wide)
            mazs = mazs[[
                col for col in mazs.columns
                if col not in wait_times_wide.columns]]
            mazs = pd.merge(
                mazs, wait_times_wide, left_on='poe_id', right_index=True,
                how='left')
            mazs.to_csv(os.path.join(
                data_dir, settings['mazs_output_fname']), index=False)

        all_wait_times = pd.concat(all_wait_times)
        all_wait_times['iter'] = np.array(range(0, num_iters)).repeat(
            all_wait_times.index.nunique())
        all_wait_times.to_csv('./data/all_wait_times.csv')

        pd.concat(all_vol_dfs).to_csv('./data/all_vol_dfs.csv')

    if run_asim:

        print('RUNNING ACTIVITYSIM!')
        process = subprocess.Popen(
            ['python', '-u', 'simulation.py'],
            stdout=sys.stdout, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.SubprocessError(stderr.decode())
