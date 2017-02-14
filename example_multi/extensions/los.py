# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd
import orca

logger = logging.getLogger(__name__)


class NetworkLOS(object):


    def __init__(self, taz, maz, tap, maz2maz, maz2tap,
                 taz_skim_dict, taz_skim_stack, tap_skim_stack,
                 tap_skim_dict, taz_skim_offsets, tap_skim_offsets):

        self.taz = taz
        self.maz = maz
        self.tap = tap
        self.maz2maz = maz2maz
        self.maz2tap = maz2tap

        self.taz_skim_dict = taz_skim_dict
        self.taz_skim_stack = taz_skim_stack
        self.taz_skim_offsets = taz_skim_offsets

        self.tap_skim_dict = tap_skim_dict
        self.tap_skim_stack = tap_skim_stack
        self.tap_skim_offsets = tap_skim_offsets


    def get_taz_for_taps(self, taps):

        return taps

    def __str__(self):

        return "\n".join((
            "taz (%s)" % len(self.taz.index),
            "maz (%s)" % len(self.maz.index),
            "tap (%s)" % len(self.tap.index),
            "maz2maz (%s)" % len(self.maz2maz.index),
            "maz2tap (%s)" % len(self.maz2tap.index),
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
                      taz_skim_dict, taz_skim_stack, tap_skim_stack)

    return nlos

