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

        # synthetic i field for OD lookup
        m = max(maz2maz.OMAZ.max(), maz2maz.DMAZ.max()) + 1
        maz2maz['i'] = maz2maz.OMAZ * m + maz2maz.DMAZ
        maz2maz.set_index('i', drop=True, inplace=True, verify_integrity=True)
        self.maz2maz_df = maz2maz
        self.maz2maz_max = m

        self.maz2tap_df = maz2tap

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

        # synthetic i method
        i = np.asanyarray(omaz) * self.maz2maz_max + np.asanyarray(dmaz)
        s = self.maz2maz_df[attribute].loc[i]

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

