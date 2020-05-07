# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import skim as askim
from activitysim.core.util import quick_loc_df

from activitysim.core.tracing import print_elapsed_time

logger = logging.getLogger('activitysim')


class NetworkLOS(object):

    def __init__(self, taz, maz, tap, maz2maz, maz2tap,
                 taz_skim_dict, tap_skim_dict):

        self.taz_df = taz
        self.maz_df = maz
        self.tap_df = tap

        # print "maz_df unique maz", len(self.maz_df.index)

        # maz2maz_df
        self.maz2maz_df = maz2maz
        # create single index for fast lookup
        m = maz2maz.DMAZ.max() + 1
        maz2maz['i'] = maz2maz.OMAZ * m + maz2maz.DMAZ
        maz2maz.set_index('i', drop=True, inplace=True, verify_integrity=True)
        self.maz2maz_cardinality = m

        # maz2tap_df
        self.maz2tap_df = maz2tap
        # create single index for fast lookup
        m = maz2tap.TAP.max() + 1
        maz2tap['i'] = maz2tap.MAZ * m + maz2tap.TAP
        maz2tap.set_index('i', drop=True, inplace=True, verify_integrity=True)
        self.maz2tap_cardinality = m

        self.taz_skim_dict = taz_skim_dict
        self.taz_skim_stack = askim.SkimStack(taz_skim_dict)

        self.tap_skim_dict = tap_skim_dict
        self.tap_skim_stack = askim.SkimStack(tap_skim_dict)

    def get_taz(self, taz_list, attribute):
        return quick_loc_df(taz_list, self.taz_df, attribute)

    def get_tap(self, tap_list, attribute):
        return quick_loc_df(tap_list, self.tap_df, attribute)

    def get_maz(self, maz_list, attribute):
        return quick_loc_df(maz_list, self.maz_df, attribute)

    def get_tazpairs(self, otaz, dtaz, key):
        skim = self.taz_skim_dict.get(key)
        s = skim.get(otaz, dtaz)
        return s

    def get_tazpairs3d(self, otaz, dtaz, dim3, key):
        s = self.taz_skim_stack.lookup(otaz, dtaz, dim3, key)
        return s

    def get_tappairs(self, otap, dtap, key):
        skim = self.tap_skim_dict.get(key)
        s = skim.get(otap, dtap)

        n = (skim.data < 0).sum()
        p = (skim.data >= 0).sum()
        nan = np.isnan(skim.data).sum()
        print "get_tappairs %s %s neg %s po %s nan" % (key, n, p, nan)

        return s

    def get_tappairs3d(self, otap, dtap, dim3, key):
        s = self.tap_skim_stack.lookup(otap, dtap, dim3, key)
        return s

    def get_mazpairs(self, omaz, dmaz, attribute):

        # # this is slower
        # s = pd.merge(pd.DataFrame({'OMAZ': omaz, 'DMAZ': dmaz}),
        #              self.maz2maz_df,
        #              how="left")[attribute]

        # synthetic index method i : omaz_dmaz
        i = np.asanyarray(omaz) * self.maz2maz_cardinality + np.asanyarray(dmaz)
        s = quick_loc_df(i, self.maz2maz_df, attribute)

        # FIXME - no point in returning series? unless maz and tap have same index?
        return np.asanyarray(s)

    def get_maztappairs(self, maz, tap, attribute):

        # synthetic i method : maz_tap
        i = np.asanyarray(maz) * self.maz2tap_cardinality + np.asanyarray(tap)
        s = quick_loc_df(i, self.maz2tap_df, attribute)

        # FIXME - no point in returning series? unless maz and tap have sme index?
        return np.asanyarray(s)

    def get_taps_mazs(self, maz, attribute=None, filter=None):

        # we return multiple tap rows for each maz, so we add an 'idx' row to tell caller
        # which maz-taz rows belong to which row in the original maz list
        # i.e. idx contains the index of the original maz series so we know which
        # rows belong together
        # if maz is a series, then idx has the original maz series index values
        # otherwise it has the 0-based integer offset of the original maz

        if filter:
            maz2tap_df = self.maz2tap_df[pd.notnull(self.maz2tap_df[filter])]
        else:
            maz2tap_df = self.maz2tap_df

        if attribute:
            # FIXME - not sure anyone needs this feature
            maz2tap_df = maz2tap_df[['MAZ', 'TAP', attribute]]
            # filter out null attribute rows
            maz2tap_df = maz2tap_df[pd.notnull(self.maz2tap_df[attribute])]
        else:
            maz2tap_df = maz2tap_df[['MAZ', 'TAP']]

        if isinstance(maz, pd.Series):
            # idx based on index of original maz series
            maz_df = pd.DataFrame({'MAZ': maz, 'idx': maz.index})
        else:
            # 0-based index of original maz
            maz_df = pd.DataFrame({'MAZ': maz, 'idx': range(len(maz))})

        df = pd.merge(maz_df, maz2tap_df, how="inner", sort=False)

        return df

    def get_tappairs_mazpairs(network_los, omaz, dmaz, ofilter=None, dfilter=None):

        # get nearby boarding TAPs to origin
        omaz_btap_df = network_los.get_taps_mazs(omaz, ofilter)

        # get nearby alighting TAPs to destination
        dmaz_atap_df = network_los.get_taps_mazs(dmaz, dfilter)

        # expand to one row for every btab-atap pair
        atap_btap_df = pd.merge(omaz_btap_df, dmaz_atap_df, on='idx', how="inner")
        atap_btap_df.rename(
            columns={'MAZ_x': 'omaz', 'TAP_x': 'btap', 'MAZ_y': 'dmaz', 'TAP_y': 'atap'},
            inplace=True)

        return atap_btap_df

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


@inject.injectable(cache=True)
def network_los(store, taz_skim_dict, tap_skim_dict):

    taz = store["TAZ"]
    maz = store["MAZ"]
    tap = store["TAP"]
    maz2maz = store["MAZtoMAZ"]
    maz2tap = store["MAZtoTAP"]

    print "taz index %s columns %s" % (taz.index.name, taz.columns.values)
    print "tap index %s columns %s" % (tap.index.name, tap.columns.values)
    print "maz index %s columns %s" % (maz.index.name, maz.columns.values)

    print "maz2maz index %s columns %s" % (maz2maz.index.name, maz2maz.columns.values)
    print "maz2tap index %s columns %s" % (maz2tap.index.name, maz2tap.columns.values)

    # print "tap index %s columns %s" % (tap.index.name, tap.columns.values)
    # print "tap_skim_offsets index %s columns %s" % (tap_skim_offsets.index.name,
    #                                                 tap_skim_offsets.columns.values)

    nlos = NetworkLOS(taz, maz, tap, maz2maz, maz2tap, taz_skim_dict, tap_skim_dict)

    return nlos
