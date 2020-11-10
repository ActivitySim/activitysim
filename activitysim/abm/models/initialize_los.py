# ActivitySim
# See full license in LICENSE.txt.
import logging
import warnings
import os
import itertools

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import simulate
from activitysim.core import expressions

from activitysim.core.util import reindex
from activitysim.core.input import read_input_table

from activitysim.abm.models import initialize

from activitysim.abm.models.util import tour_frequency as tf


logger = logging.getLogger(__name__)


TAP_TAP_UID = 'TT_ID'


def compute_utilities(network_los, model_settings, choosers, model_constants,
                      trace_label, trace, chooser_tag_col_name=None):
    trace_label = tracing.extend_trace_label(trace_label, 'compute_utilities')

    logger.debug(f"{trace_label} Running compute_utilities with {choosers.shape[0]} choosers")

    locals_dict = {'np': np, 'los': network_los}
    locals_dict.update(model_constants)

    # we don't grok coefficients, but allow them to use constants in spec alt columns
    spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    for c in spec.columns:
        if c != simulate.SPEC_LABEL_NAME:
            spec[c] = spec[c].map(lambda s: model_constants.get(s, s)).astype(float)

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.get('PREPROCESSOR')
    if preprocessor_settings:

        # don't want to alter caller's dataframe
        choosers = choosers.copy()

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    utilities = simulate.eval_utilities(
        spec,
        choosers,
        locals_d=locals_dict,
        trace_all_rows=trace,
        trace_label=trace_label,
        chooser_tag_col_name=chooser_tag_col_name)

    return utilities

class TapTapUtilityCalculator(object):

    def __init__(self, network_los, trace_label):
        self.trace_label = trace_label
        self.network_los = network_los
        self.tap_ceiling = network_los.tap_ceiling

        # initialize integerizer
        self.integerizer = {}
        zone_ids = network_los.get_skim_dict('tap').zone_ids
        self.integerizer['btap'] = pd.Series(range(len(zone_ids)), index=zone_ids)
        self.integerizer['atap'] = self.integerizer['btap']
        segmentation = network_los.setting('TRANSIT_VIRTUAL_PATH_SETTINGS.tour_mode_choice.precompute_tap_tap_segments')
        for k, v in segmentation.items():
            self.integerizer[k] = pd.Series(range(len(v)), index=v)

        # for k,v in self.integerizer.items():
        #     print(f"\nintegerizer {k}\n{v}")

    def unique_tap_tap_attribute_id(self, df, **scalar_attributes):
        """
        assign a unique 
        btap and atap will be in dataframe, but the other attributes may be either df columns or scalar_attributes
        
        Parameters
        ----------
        df: pandas DataFrame
            with btap, atap, and optionally additional attribute columns
        scalar_attributes: dict
            dict of scalar attributes e.g. {'tod': 'AM', 'demographic_segment': 0}

        Returns
        -------

        """
        tap_tap_uid = np.zeros(len(df), dtype=int)

        # need to know cardinality and integer representation of each tap/attribute
        for name, integerizer in self.integerizer.items():

            assert (name in df) != (name in scalar_attributes), \
                f"attribute '{name}' must be in EITHER df OR scalar_attributes, not both." \
                f" df: {list(df.columns)} scalar_attributes: {list(scalar_attributes.keys())}"

            cardinality = integerizer.max() + 1

            if name in df:
                tap_tap_uid = tap_tap_uid * cardinality + np.asanyarray(df[name].map(integerizer))
            else:
                tap_tap_uid = tap_tap_uid * cardinality + integerizer.at[scalar_attributes[name]]

        return tap_tap_uid

    def compute_tap_tap_utilities(self, orig, dest, **scalar_attributes):

        # populate df columns with scalar_attributes
        df = pd.DataFrame({'btap': orig,  'atap': dest})
        for attribute, value in scalar_attributes.items():
            df[attribute] = value
        df.index = self.unique_tap_tap_attribute_id(df)
        assert not df.index.duplicated().any()
        print(df)

        # only btap and atap in df, scalar_attributes in dict
        df = pd.DataFrame({'btap': orig,  'atap': dest})
        df.index = self.unique_tap_tap_attribute_id(df, **scalar_attributes)
        assert not df.index.duplicated().any()
        print(df)

        bug

        return df


@inject.step()
def initialize_los(network_los):

    trace_label = 'initialize_los'

    utility_calculator = TapTapUtilityCalculator(network_los, trace_label)

    taps = network_los.tap_df['TAP']
    #taps = taps[:3]

    # don't assume they are the same: orig may be sliced if we are multiprocessing
    orig_zones = taps.values
    dest_zones = taps.values

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'btap': np.repeat(orig_zones, len(dest_zones)),
            'atap': np.tile(dest_zones, len(orig_zones))
        }
    )

    segmentation = network_los.setting('TRANSIT_VIRTUAL_PATH_SETTINGS.tour_mode_choice.precompute_tap_tap_segments')

    # attribute names as list of strings
    attribute_names = list(segmentation.keys())

    # list of attribute combination tuples (e.g. [(0, 'AM'), (1, 'AM'), (0, 'MD'),...])
    attribute_combinations = list(itertools.product(*list(segmentation.values())))

    df_list = []
    for attribute_value_tuple in attribute_combinations:

        # attribute_value_tuple is an attribute combination tuple e.g. (0, 'AM')

        # build dict of attribute {<name>: <value>} (e.g. {'demographic_segment': 0, 'tod': 'AM'})
        scalar_attributes = {name: value for name, value in zip(attribute_names, attribute_value_tuple)}

        # scalar attribute values as named parameters
        df = utility_calculator.compute_tap_tap_utilities(od_df.btap, od_df.atap, **scalar_attributes)

        df_list.append(df)

    df = pd.concat(df_list)

    assert not df.index.duplicated().any()

    print(df)
    bug
