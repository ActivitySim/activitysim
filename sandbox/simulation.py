import os

import openmatrix as omx
import orca
import numpy as np
import pandas as pd

from activitysim import defaults
from activitysim import skim as askim
from activitysim import activitysim as asim

import os
import psutil
import resource
import gc


def high_water_mark():
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = (peak_bytes / (1024 * 1024 * 1024.0))
    return "%s GB" % (round(peak_gb, 2),)


def set_random_seed():
    np.random.seed(0)


def print_table_schema(orca_table_name):
    df = orca.get_table(orca_table_name).to_frame()
    print "\n", orca_table_name
    for col in df.columns:
       print "  %s: %s" % (col, df[col].dtype)

@orca.injectable()
def configs_dir():
    #return '/Users/jeff.doyle/work/activitysim/example'
    return '/Users/jeff.doyle/work/activitysim/sandbox'

@orca.injectable()
def data_dir():
    #return '/Users/jeff.doyle/work/activitysim-data/mtc_tm1_sf_test'
    #return '/Users/jeff.doyle/work/activitysim-data/mtc_tm1_sf'
    return '/Users/jeff.doyle/work/activitysim-data/mtc_tm1'

orca.add_injectable("set_random_seed", set_random_seed)

print "gc enabled:", gc.isenabled()
print "gc get_threshold:", gc.get_threshold()

print "households_sample_size =", orca.get_injectable('settings')['households_sample_size']
print "preload_3d_skims", orca.get_injectable('preload_3d_skims')
print "data dir", orca.get_injectable('data_dir')

#gc.set_debug(gc.DEBUG_STATS)

print asim.usage('pre-skim')

skims = orca.get_injectable('skims')
print asim.usage('after skim load\n')

skims = orca.get_injectable('stacked_skims')
print asim.usage('after stacked_skims load\n')


orca.run(["school_location_simulate"])
print asim.usage('after school_location_simulate\n')

orca.run(["workplace_location_simulate"])
print asim.usage('after workplace_location_simulate\n')

orca.run(["auto_ownership_simulate"])
print asim.usage('after auto_ownership_simulate\n')

orca.run(["cdap_simulate"])
print asim.usage('after cdap_simulate\n')

orca.run(['mandatory_tour_frequency'])
print asim.usage('after mandatory_tour_frequency\n')

orca.run(["mandatory_scheduling"])
print asim.usage('after mandatory_scheduling\n')

orca.run(['non_mandatory_tour_frequency'])
print asim.usage('after non_mandatory_tour_frequency\n')

orca.run(["destination_choice"])
print asim.usage('after destination_choice\n')

orca.run(["non_mandatory_scheduling"])
print asim.usage('after non_mandatory_scheduling\n')

# FIXME - jwd - choose more felicitous name or do this elsewhere?
orca.run(["patch_mandatory_tour_destination"])
print asim.usage('after patch_mandatory_tour_destination\n')

orca.run(['tour_mode_choice_simulate'])
print asim.usage('after tour_mode_choice_simulate\n')

orca.run(['trip_mode_choice_simulate'])
print asim.usage('after trip_mode_choice_simulate\n')


# print "\n\nFinal Tables"
# for table_name in ["households", "persons", "accessibility", "land_use", "tours", "tours_merged", "persons_merged"]:
#     print_table_schema(table_name)

orca.get_injectable('store').close()
orca.get_injectable('omx_file').close()

print "data dir", orca.get_injectable('data_dir')
print "households_sample_size =", orca.get_injectable('settings')['households_sample_size']
print "preload_3d_skims = ", orca.get_injectable('preload_3d_skims')
print "max memory footprint = %s" % high_water_mark()
