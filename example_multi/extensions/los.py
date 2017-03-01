# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd
import orca

logger = logging.getLogger(__name__)


class NetworkLOS(object):


    def __init__(self, taz, maz, tap, maz2maz, maz2tap,
                 taz_skim_dict, taz_skim_stack, tap_skim_stack, tap_skim_dict):

        self.taz_df = taz
        self.maz_df = maz
        self.tap_df = tap

        # print "maz_df unique maz", len(self.maz_df.index)

        # maz2maz_df
        m = maz2maz.DMAZ.max() + 1
        maz2maz['i'] = maz2maz.OMAZ * m + maz2maz.DMAZ
        maz2maz.set_index('i', drop=True, inplace=True, verify_integrity=True)
        self.maz2maz_df = maz2maz
        self.maz2maz_max_omaz = m

        # print "maz_df unique maz pairs", len(maz2maz.index)

        # maz2tap_df
        m = maz2tap.TAP.max() + 1
        maz2tap['i'] = maz2tap.MAZ * m + maz2tap.TAP
        maz2tap.set_index('i', drop=True, inplace=True, verify_integrity=True)
        self.maz2tap_df = maz2tap
        self.maz2tap_max_tap = m

        # print "maz2tap_df unique pairs", len(maz2tap.index)
        # print "maz2tap_df unique maz", len(maz2tap.MAZ.unique())

        self.taz_skim_dict = taz_skim_dict
        self.taz_skim_stack = taz_skim_stack

        self.tap_skim_dict = tap_skim_dict
        self.tap_skim_stack = tap_skim_stack


    def get_taz_offsets(self, taz_list):
        return self.taz_df.offset.loc[taz_list]

    def get_taz(self, taz_list, attribute):
        return self.taz_df.loc[taz_list][attribute]

    def get_tap_offsets(self, tap_list):
        return self.tap_df.offset.loc[tap_list]

    def get_tap(self, tap_list, attribute):
        return self.tap_df.loc[tap_list][attribute]

    def get_maz(self, maz_list, attribute):
        return self.maz_df.loc[maz_list][attribute]

    def get_tazpairs(self, otaz, dtaz, key):
        otaz = self.get_taz_offsets(otaz)
        dtaz = self.get_taz_offsets(dtaz)
        skim = self.taz_skim_dict.get(key)
        s = skim.get(otaz, dtaz)
        return s

    def get_tazpairs3d(self, otaz, dtaz, dim3, key):
        otaz = self.get_taz_offsets(otaz).astype('int')
        dtaz = self.get_taz_offsets(dtaz).astype('int')
        stacked_skim_data, skim_keys_to_indexes = self.taz_skim_stack.get(key)
        skim_indexes = np.vectorize(skim_keys_to_indexes.get)(dim3)
        s = stacked_skim_data[otaz, dtaz, skim_indexes]
        return s

    def get_tappairs(self, otap, dtap, key):
        otap = self.get_tap_offsets(otap)
        dtap = self.get_tap_offsets(dtap)
        skim = self.tap_skim_dict.get(key)
        s = skim.get(otap, dtap)
        return s

    def get_tappairs3d(self, otap, dtap, dim3, key):
        otap = self.get_tap_offsets(otap).astype('int')
        dtap = self.get_tap_offsets(dtap).astype('int')
        stacked_skim_data, skim_keys_to_indexes = self.tap_skim_stack.get(key)
        skim_indexes = np.vectorize(skim_keys_to_indexes.get)(dim3)
        s = stacked_skim_data[otap, dtap, skim_indexes]
        return s

    def get_mazpairs(self, omaz, dmaz, attribute):

        # # this is slower
        # s = pd.merge(pd.DataFrame({'OMAZ': omaz, 'DMAZ': dmaz}),
        #              self.maz2maz_df,
        #              how="left")[attribute]

        # synthetic index method i : omaz_dmaz
        i = np.asanyarray(omaz) * self.maz2maz_max_omaz + np.asanyarray(dmaz)
        s = self.maz2maz_df[attribute].loc[i]

        # FIXME - no point in returning series? unless maz and tap have sme index?
        return np.asanyarray(s)

    def get_maztappairs(self, maz, tap, attribute):

        # synthetic i method : maz_tap
        i = np.asanyarray(maz) * self.maz2tap_max_tap + np.asanyarray(tap)
        s = self.maz2tap_df[attribute].loc[i]

        # FIXME - no point in returning series? unless maz and tap have sme index?
        return np.asanyarray(s)

    def get_taps_mazs(self, omaz, attribute=None):

        # idx is just the 0-based index of the omaz series so we know which rows belong together

        if attribute:
            maz2tap_df = self.maz2tap_df[ ['MAZ', 'TAP', attribute]]

            # filter out null attribute rows
            maz2tap_df = maz2tap_df[ pd.notnull(self.maz2tap_df[attribute]) ]
        else:
            maz2tap_df = self.maz2tap_df[['MAZ', 'TAP']]

        df = pd.merge(pd.DataFrame({'MAZ': omaz, 'idx': range(len(omaz))}),
                      maz2tap_df,
                      how="inner")

        return df


    def get_tappairs_mazpairs(self, omaz, dmaz, tod, criteria):


        # Step 1 - get nearby boarding TAPs to origin
        omaz_btap_table = self.get_taps_mazs(omaz, 'drive_time')

        #print "\nomaz_btap_table\n", omaz_btap_table

        # Step 2 - get nearby alighting TAPs to destination
        dmaz_atap_table = self.get_taps_mazs(dmaz, 'drive_time')

        #print "\ndmaz_atap_table\n", dmaz_atap_table

        atap_btap = pd.merge(omaz_btap_table, dmaz_atap_table, on='idx', how="inner")

        atap_btap.rename(columns={'MAZ_x': 'omaz', 'TAP_x': 'btap', 'drive_time_x': 'omaz_btap_cost',
                                  'MAZ_y': 'dmaz', 'TAP_y': 'atap', 'drive_time_y': 'dmaz_atap_cost'},
                         inplace=True)

        print "\ntap_df\n", self.tap_df.head(1)

        atap_btap['btap_cost'] = np.asanyarray(self.get_tap(atap_btap.btap, "distance"))
        atap_btap['atap_cost'] = np.asanyarray(self.get_tap(atap_btap.atap, "distance"))

        atap_btap['btap_atap_cost'] = self.get_tappairs3d(atap_btap.atap, atap_btap.btap, tod, 'LOCAL_BUS_INITIAL_WAIT')

        # drop rows if no travel between taps
        # atap_btap = atap_btap[ atap_btap.btap_atap_cost > 0 ]

        print "\natap_btap\n", atap_btap

        return omaz_btap_table


    def __str__(self):

        return "\n".join((
            "taz (%s)" % len(self.taz_df.index),
            "maz (%s)" % len(self.maz_df.index),
            "tap (%s)" % len(self.tap_df.index),
            "maz2maz (%s)" % len(self.maz2maz_df.index),
            "maz2tap (%s)" % len(self.maz2tap_df.index),
            "taz_skim_dict (%s keys)" % self.taz_skim_dict.key_count(),
            "tap_skim_dict (%s keys)" % self.tap_skim_dict.key_count(),
            "taz_skim_stack (%s keys)" % self.taz_skim_stack.key_count(),
            "tap_skim_stack (%s keys)" % self.tap_skim_stack.key_count(),
        ))

@orca.injectable(cache=True)
def network_los(store, taz_skim_dict, tap_skim_dict, taz_skim_stack, tap_skim_stack):

    taz = store["TAZ"]
    maz = store["MAZ"]
    tap = store["TAP"]
    maz2maz = store["MAZtoMAZ"]
    maz2tap = store["MAZtoTAP"]


    print "taz index %s columns %s" % (taz.index.name, taz.columns.values)
    print "tap index %s columns %s" % (tap.index.name, tap.columns.values)

    # print "tap index %s columns %s" % (tap.index.name, tap.columns.values)
    # print "tap_skim_offsets index %s columns %s" % (tap_skim_offsets.index.name, tap_skim_offsets.columns.values)

    nlos = NetworkLOS(taz, maz, tap, maz2maz, maz2tap,
                      taz_skim_dict, taz_skim_stack, tap_skim_stack, tap_skim_dict)

    return nlos

