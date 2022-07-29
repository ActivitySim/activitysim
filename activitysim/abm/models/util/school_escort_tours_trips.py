from activitysim.core import pipeline

import pandas as pd
import numpy as np
import warnings


def determine_chauf_outbound_flag(row, i):
    if (row['direction'] == 'outbound'):
        outbound = True
    elif (row['direction'] == 'inbound') & (i == 0) & (row['escort_type'] == 'pure_escort'):
        # chauf is going to pick up the first child
        outbound = True
    else:
        # chauf is inbound and has already picked up a child or taken their mandatory tour
        outbound = False
    return outbound


def create_chauf_trip_table(row):
    dropoff = True if row['direction'] == 'outbound' else False

    row['person_id'] = row['chauf_id']
    row['destination'] = row['school_destinations'].split('_')

    participants = []        
    school_escort_trip_num = []
    outbound = []
    purposes = []

    for i, child_id in enumerate(row['escortees'].split('_')):
        if dropoff:
            # have the remaining children in car
            participants.append('_'.join(row['escortees'].split('_')[i:]))
        else:
            # remaining children not yet in car
            participants.append('_'.join(row['escortees'].split('_')[:i+1]))
        school_escort_trip_num.append(i + 1)
        outbound.append(determine_chauf_outbound_flag(row, i))
        purposes.append('escort')
      
    if not dropoff:
        # adding trip home
        outbound.append(False)
        school_escort_trip_num.append(i + 2)
        purposes.append('home')
        row['destination'].append(row['home_zone_id'])
        # kids aren't in car until after they are picked up, inserting empty car for first trip
        participants = [''] + participants

    row['escort_participants'] = participants
    row['school_escort_trip_num'] = school_escort_trip_num
    row['outbound'] = outbound
    row['purpose'] = purposes
    return row


def create_chauf_escort_trips(bundles):

    chauf_trip_bundles = bundles.apply(lambda row: create_chauf_trip_table(row), axis=1)
    chauf_trip_bundles['tour_id'] = bundles['chauf_tour_id'].astype(int)

    # departure time is the first school start in the outbound direction and the last school end in the inbound direction
    starts = chauf_trip_bundles['school_starts'].str.split('_', expand=True).astype(float)
    ends = chauf_trip_bundles['school_ends'].str.split('_', expand=True).astype(float)
    chauf_trip_bundles['depart'] = np.where(chauf_trip_bundles['direction'] == 'outbound', starts.min(axis=1), ends.max(axis=1))

    # create a new trip for each escortee destination
    chauf_trips = chauf_trip_bundles.explode(['destination', 'escort_participants', 'school_escort_trip_num', 'outbound', 'purpose']).reset_index()

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    outbound_trip_num = -1 * (chauf_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=False) + 1)
    inbound_trip_num = 100 + chauf_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=True)
    chauf_trips['trip_num'] = np.where(chauf_trips.outbound == True, outbound_trip_num, inbound_trip_num)

    # --- determining trip origin    
    # origin is previous destination
    chauf_trips['origin'] = chauf_trips.groupby('tour_id')['destination'].shift()
    # outbound trips start at home
    first_outbound_trips = ((chauf_trips['outbound'] == True) & (chauf_trips['school_escort_trip_num'] == 1))
    chauf_trips.loc[first_outbound_trips, 'origin'] = chauf_trips.loc[first_outbound_trips, 'home_zone_id']
    

    chauf_trips['primary_purpose'] = np.where(chauf_trips['escort_type'] == 'pure_escort', 'escort', 'work')

    chauf_trips['trip_id'] = chauf_trips['tour_id'].astype(int) * 10 + chauf_trips.groupby('tour_id').cumcount()

    trip_cols = ['trip_id', 'household_id', 'person_id', 'tour_id', 'destination', 'depart', 'escort_participants',
                 'school_escort_trip_num', 'outbound', 'trip_num', 'primary_purpose', 'purpose', 'direction', 'home_zone_id']
    chauf_trips = chauf_trips[trip_cols]

    chauf_trips.loc[chauf_trips['purpose'] == 'home', 'trip_num'] = 999  # trips home are always last
    chauf_trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)

    return chauf_trips


def create_child_escorting_stops(row, escortee_num):
    escortees = row['escortees'].split('_')
    if escortee_num > (len(escortees) - 1):
        # this bundle does not have this many escortees
        return row
    dropoff = True if row['direction'] == 'outbound' else False
    
    row['person_id'] = int(escortees[escortee_num])
    row['tour_id'] = row['school_tour_ids'].split('_')[escortee_num]
    school_dests = row['school_destinations'].split('_')

    destinations = []
    purposes = []
    participants = []
    school_escort_trip_num = []

    escortee_order = escortees[:escortee_num + 1] if dropoff else escortees[escortee_num:]

    # for i, child_id in enumerate(escortees[:escortee_num+1]):
    for i, child_id in enumerate(escortee_order):
        is_last_stop = (i == len(escortee_order) - 1)

        if dropoff:
            # dropping childen off
            # children in car are the child and the children after
            participants.append('_'.join(escortees[i:]))
            dest = school_dests[i]
            purpose = 'school' if row['person_id'] == int(child_id) else 'escort'

        else:
            # picking children up
            # children in car are the child and those already picked up
            participants.append('_'.join(escortees[:escortee_num + i +1]))
            # going home if last stop, otherwise to next school destination
            dest = row['home_zone_id'] if is_last_stop else school_dests[i+1]
            purpose = 'home' if is_last_stop else 'escort'
            
        
        # filling arrays
        destinations.append(dest)
        school_escort_trip_num.append(i + 1)
        purposes.append(purpose)

    row['escort_participants'] = participants
    row['school_escort_trip_num'] = school_escort_trip_num
    row['purpose'] = purposes
    row['destination'] = destinations
    return row


def create_escortee_trips(bundles):

    escortee_trips = []
    for escortee_num in range(0, bundles.num_escortees.max() + 1):
        escortee_bundles = bundles.apply(lambda row: create_child_escorting_stops(row, escortee_num), axis=1)
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)
    escortee_trips = escortee_trips[~escortee_trips.person_id.isna()]

    # departure time is the first school start in the outbound direction and the last school end in the inbound direction
    starts = escortee_trips['school_starts'].str.split('_', expand=True).astype(float)
    ends = escortee_trips['school_ends'].str.split('_', expand=True).astype(float)
    escortee_trips['outbound'] = np.where(escortee_trips['direction'] == 'outbound', True, False)
    escortee_trips['depart'] = np.where(escortee_trips['direction'] == 'outbound', starts.min(axis=1), ends.max(axis=1))
    escortee_trips['primary_purpose'] = 'school'

    # create a new trip for each escortee destination
    escortee_trips = escortee_trips.explode(['destination', 'escort_participants', 'school_escort_trip_num', 'purpose']).reset_index()

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    outbound_trip_num = -1 * (escortee_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=False) + 1)
    inbound_trip_num = 100 + escortee_trips.groupby(['tour_id', 'outbound']).cumcount(ascending=True)
    escortee_trips['trip_num'] = np.where(escortee_trips.outbound == True, outbound_trip_num, inbound_trip_num)

    # FIXME placeholders
    escortee_trips['trip_id'] = escortee_trips['tour_id'].astype(int) + 100 * escortee_trips.groupby('tour_id')['trip_num'].transform('count')

    trip_cols = ['trip_id', 'household_id', 'person_id', 'tour_id', 'destination', 'depart', 'escort_participants',
                'school_escort_trip_num', 'outbound', 'primary_purpose', 'purpose', 'direction', 'trip_num', 'home_zone_id']
    escortee_trips = escortee_trips[trip_cols]

    for col in escortee_trips.columns:
        if '_id' in col:
            escortee_trips[col] = escortee_trips[col].astype(int)

    escortee_trips.loc[escortee_trips['purpose'] == 'home', 'trip_num'] = 999  # trips home are always last
    escortee_trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)

    return escortee_trips


# def merge_school_escorting_trips(chauf_trips, escortee_trips, trips, tours):
#     # create filters to remove trips that were created or are not unallowed
#     # primary tour destination trip for chauf is replaced
#     no_chauf_primary_dest = (~(trips.tour_id.isin(chauf_trips.tour_id)
#         & (trips['outbound'] == True)
#         & (trips['trip_num'] == trips['trip_count'])
#         & (trips['primary_purpose'] == 'escort')
#     ))

#     # outbound escortee trips to primary school destination
#     outbound_school_escorting_tours = tours[~tours.school_esc_outbound.isna()]
#     no_escortee_out_primary_dest = (~(trips.tour_id.isin(outbound_school_escorting_tours.index)
#         & (trips['outbound'] == True)
#         & (trips['trip_num'] == trips['trip_count'])
#         & (trips['purpose'] == 'school')
#     ))
    
#     # inbound escortee trips to home
#     inbound_school_escorting_tours = tours[~tours.school_esc_inbound.isna()]
#     no_escortee_inb_home = (~(trips.tour_id.isin(inbound_school_escorting_tours.index)
#         & (trips['outbound'] == False)
#         & (trips['trip_num'] == trips['trip_count'])
#         & (trips['purpose'] == 'home')
#     ))

#     cut_trips = trips[
#         no_chauf_primary_dest & no_escortee_out_primary_dest & no_escortee_inb_home
#     ]
#     cut_trips.reset_index(inplace=True)

#     all_trips = pd.concat([chauf_trips, escortee_trips, cut_trips])

#     all_trips.loc[all_trips['purpose'] == 'home', 'trip_num'] = 999  # trips home are always last
#     all_trips.sort_values(by=['household_id', 'tour_id', 'outbound', 'trip_num'], ascending=[True, True, False, True], inplace=True)

#     # recomputing trip statistics
#     all_trips['trip_num'] = all_trips.groupby(['tour_id', 'outbound']).cumcount() + 1
#     all_trips['trip_count'] = all_trips.groupby(['tour_id', 'outbound'])['trip_num'].transform('count')
#     # all tours start at home
#     first_trips = ((all_trips['outbound'] == True) & (all_trips['trip_num'] == 1))
#     all_trips.loc[first_trips, 'origin'] = all_trips.loc[first_trips, 'home_zone_id']
#     all_trips['origin'] = np.where(all_trips['origin'].isna(), all_trips.groupby('tour_id')['destination'].shift(), all_trips['origin'])

#     # participants aren't in the car until the subsequent trip in the inbound direction
#     # all_trips['escort_participants'] = np.where(all_trips['outbound'] == False, all_trips.groupby('tour_id')['escort_participants'].shift(), all_trips['escort_participants'])
#     all_trips.set_index('trip_id', inplace=True)

#     return all_trips

def create_school_escort_trips(escort_bundles):
    chauf_trips = create_chauf_escort_trips(escort_bundles)
    escortee_trips = create_escortee_trips(escort_bundles)
    school_escort_trips = pd.concat([chauf_trips, escortee_trips], axis=0)
    return school_escort_trips
    


# def create_school_escort_trips():

#     # start with creating outbound chauffer trips
#     bundles = pipeline.get_table('escort_bundles')
#     tours = pipeline.get_table('tours')
#     trips = pipeline.get_table('trips')

#     bundles.to_csv('escort_bundles.csv')
#     tours.to_csv('tours.csv')
#     trips.to_csv('trips.csv')

#     chauf_trips = create_chauf_escort_trips(bundles, trips, tours)
#     escortee_trips = create_escortee_trips(bundles, trips, tours)
#     trips = merge_school_escorting_trips(chauf_trips, escortee_trips, trips, tours)

#     pipeline.replace_table("trips", trips)
#     pipeline.replace_table("escort_bundles", bundles)
#     # since new trips were created, we need to reset the random number generator
#     pipeline.get_rn_generator().drop_channel('trips')
#     pipeline.get_rn_generator().add_channel('trips', trips)


def create_pure_school_escort_tours(bundles):
    # creating home to school tour for chauffers making pure escort tours
    # ride share tours are already created since they go off the mandatory tour

    # FIXME: can I just move all of this logic to a csv and annotate??
    # bundles = pipeline.get_table('escort_bundles')
    persons = pipeline.get_table('persons')
    pe_tours = bundles[bundles['escort_type'] == 'pure_escort']

    pe_tours['origin'] = pe_tours['home_zone_id']
    # desination is the last dropoff / pickup location
    pe_tours['destination'] = pe_tours['school_destinations'].str.split('_').str[-1].astype(int)
    # start is the first start time
    pe_tours['start'] = pe_tours['school_starts'].str.split('_').str[0].astype(int)

    school_time_cols = ['time_home_to_school' + str(i) for i in range(1,4)]
    # FIXME hard coded mins per time bin, is rounding down appropriate?
    pe_tours['end'] = pe_tours['start'] + (pe_tours[school_time_cols].sum(axis=1) / 30).astype(int)

    pe_tours['person_id'] = pe_tours['chauf_id']
    # FIXME should probably put this when creating the bundles table
    assert all(pe_tours['person_id'].isin(persons.index)), \
        f"Chauffer ID(s) not present in persons table {pe_tours.loc[~pe_tours['person_id'].isin(persons.index), 'person_id']}"
    # pe_tours = pe_tours[pe_tours['person_id'].isin(persons.index)]

    pe_tours['tour_category'] = 'non_mandatory'
    pe_tours['number_of_participants'] = 1
    pe_tours['tour_type'] = 'escort'
    # FIXME join tdd from tdd_alts
    pe_tours['tdd'] = pd.NA
    pe_tours['duration'] = pe_tours['end'] - pe_tours['start']
    pe_tours['school_esc_outbound'] = np.where(pe_tours['direction'] == 'outbound', 'pure_escort', pd.NA)
    pe_tours['school_esc_inbound'] = np.where(pe_tours['direction'] == 'inbound', 'pure_escort', pd.NA)

    # FIXME need consistent tour ids
    pe_tours['tour_id'] = pe_tours['chauf_tour_id'].astype(int)
    pe_tours.set_index('tour_id', inplace=True)

    # for col in tours.columns:
    #     if col not in pe_tours.columns:
    #         pe_tours[col] = pd.NA
    #         print(col)

    # pe_tours[tours.columns].to_csv('pure_escort_tours.csv')
    pe_tours.to_csv('pure_escort_tours.csv')

    # tours = pd.concat([tours, pe_tours[tours.columns]])

    # # tours = pe_tours[tours.columns]

    # grouped = tours.groupby(['person_id', 'tour_type'])
    # tours['tour_type_num'] = grouped.cumcount() + 1
    # tours['tour_type_count'] = tours['tour_type_num'] + grouped.cumcount(ascending=False)

    # grouped = tours.groupby('person_id')
    # tours['tour_num'] = grouped.cumcount() + 1
    # tours['tour_count'] = tours['tour_num'] + grouped.cumcount(ascending=False)

    # tours.sort_values(by=['household_id', 'person_id', 'tour_num'], inplace=True)

    # assert tours.index.is_unique, "Non-unique tour_id's!!"

    # pipeline.replace_table("tours", tours)
    # # since new tours were created, we need to reset the random number generator
    # pipeline.get_rn_generator().drop_channel('tours')
    # pipeline.get_rn_generator().add_channel('tours', tours)

    return pe_tours
