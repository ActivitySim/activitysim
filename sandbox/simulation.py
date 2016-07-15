import logging
import logging.config

import gc
import os
import os.path
import resource
import yaml

import numpy as np
import openmatrix as omx
import orca
import pandas as pd

from activitysim import defaults
from activitysim import activitysim as asim
from activitysim import tracing


# you will want to configure this with the locations of the canonical datasets
DATA_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'activitysim-data')
DATA_DIRS = {
        'test': os.path.join(DATA_REPO, 'mtc_tm1_sf_test'),
        'example': os.path.join(DATA_REPO, 'mtc_tm1_sf'),
        'full': os.path.join(DATA_REPO, 'mtc_tm1'),
    }


# these shouldn't have to change
CONFIGS_DIRS = {
        'example': os.path.join(os.path.dirname(__file__), '..', 'example'),
        'sandbox': os.path.join(os.path.dirname(__file__)),
    }


def inject_settings(config='sandbox',
                    data='test',
                    households_sample_size=0,
                    preload_3d_skims=True,
                    chunk_size = 0,
                    hh_chunk_size = 0):

    assert config in  CONFIGS_DIRS.keys(), 'Unknown config dir %s' % config
    config_dir = CONFIGS_DIRS.get(config)
    orca.add_injectable("configs_dir", config_dir)

    assert data in DATA_DIRS.keys(), 'Unknown dataset %s' % data
    data_dir = DATA_DIRS.get(data)
    orca.add_injectable("data_dir", data_dir)

    with open(os.path.join(config_dir, "configs", "settings.yaml")) as f:
        settings = yaml.load(f)
        if households_sample_size is not None:
            settings['households_sample_size'] = households_sample_size
        settings['preload_3d_skims'] = preload_3d_skims
        settings['chunk_size'] = chunk_size
        settings['hh_chunk_size'] = hh_chunk_size
    orca.add_injectable("settings", settings)


def print_settings():

    print "data_dir: %s" % orca.get_injectable('data_dir')
    print "configs_dir: %s" % orca.get_injectable('configs_dir')
    print "households_sample_size = %s" % orca.get_injectable('settings')['households_sample_size']
    print "preload_3d_skims = %s" % orca.get_injectable('preload_3d_skims')
    print "chunk_size = %s" % orca.get_injectable('chunk_size')
    print "hh_chunk_size = %s" % orca.get_injectable('hh_chunk_size')

    print "garbage collection enabled: %s" % gc.isenabled()
    print "garbage collection threshold: %s" % str(gc.get_threshold())
    print "numpy floating-point error-handling settings: %s" % np.geterr()
    print "pandas display options max_rows=%s max_columns=%s" % \
          (pd.options.display.max_rows, pd.options.display.max_columns)


def print_table_schema(table_names):
    for table_name in table_names:
        df = orca.get_table(table_name).to_frame()
        print "\n", table_name
        for col in df.columns:
           print "  %s: %s" % (col, df[col].dtype)


def log_memory_info(logger, message):
    logger.debug("%s %s" % (message, asim.memory_info()))


def set_random_seed():
    np.random.seed(0)


def run_model(model_name):
    orca.run([model_name])
    log_memory_info(logger, 'after %s' % model_name)


orca.add_injectable("output_dir", 'output')
tracing.config_logger('logging.yaml')

logger = logging.getLogger(__name__)

# pandas display options
pd.options.display.max_columns = 500
pd.options.display.max_rows = 20

#gc.set_debug(gc.DEBUG_STATS)

orca.add_injectable("set_random_seed", set_random_seed)

# config = 'sandbox' or 'example'
# data = 'test', 'example', or 'full'
# inject_settings(config='example',
#                 data='full',
#                 households_sample_size=0,
#                 preload_3d_skims=True,
#                 chunk_size = 50000,
#                 hh_chunk_size = 50000)

inject_settings(config='sandbox',
                data='full',
                households_sample_size=50,
                preload_3d_skims=True,
                chunk_size = 0,
                hh_chunk_size = 0)

print_settings()


log_memory_info(logger, 'startup')
skims = orca.get_injectable('skims')
log_memory_info(logger, 'after skim load')
skims = orca.get_injectable('stacked_skims')
log_memory_info(logger, 'after stacked_skims load')

#t = orca.get_table('persons')
# df = orca.get_table('persons_merged').to_frame()
# df = df[ ( df.hhsize > 2 )]
# df = df[['household_id', 'is_student', 'is_worker']]
# print df.head(20)

run_model('compute_accessibility')
run_model('school_location_simulate')
run_model('workplace_location_simulate')
run_model('auto_ownership_simulate')
run_model('cdap_simulate')
run_model('mandatory_tour_frequency')
run_model('mandatory_scheduling')
run_model('non_mandatory_tour_frequency')
run_model('destination_choice')
run_model('non_mandatory_scheduling')
run_model('patch_mandatory_tour_destination')
run_model('tour_mode_choice_simulate')
run_model('trip_mode_choice_simulate')

orca.get_injectable('store').close()
orca.get_injectable('omx_file').close()

# print "\n\nFinal Tables"
# table_names = ["households", "persons", "accessibility", "land_use", "tours",
#                "tours_merged", "persons_merged"]
# print_table_schema(table_names)

#print_settings()

# this may not work on all systems...
peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_gb = (peak_bytes / (1024 * 1024 * 1024.0))
logger.debug("max memory footprint = %s (%s GB)" % (peak_bytes, round(peak_gb, 2),))

