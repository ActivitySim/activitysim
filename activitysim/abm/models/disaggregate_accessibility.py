import yaml
import collections
import os
import sys
import itertools
import logging
import subprocess
import pkg_resources
import pandas as pd
from collections import Iterable
from orca import orca

from activitysim.core import (inject,
                              tracing,
                              config,
                              pipeline,
                              util)

from activitysim.abm.models import location_choice
from activitysim.abm.models.util import (tour_destination, estimation)
from activitysim.abm.tables import shadow_pricing
from activitysim.core.expressions import assign_columns

logger = logging.getLogger(__name__)


class ProtoPop:
    def __init__(self, land_use_df, model_settings, pipeline=True):
        self.model_settings = model_settings
        self.params = self.read_table_settings(land_use_df)
        self.create_proto_pop()

        # Switch in case someone wants to generate a proto-pop in just python alone
        if pipeline:
            self.inject_tables()
            self.annotate_tables()
            self.merge_persons(land_use_df)

    def read_table_settings(self, land_use_df):
        # Check if setup properly
        assert 'CREATE_TABLES' in self.model_settings.keys()
        create_tables = self.model_settings['CREATE_TABLES']
        assert all([True for k, v in create_tables.items() if 'VARIABLES' in v.keys()])
        assert all([True for x in ['PERSONS', 'HOUSEHOLDS', 'TOURS'] if x in create_tables.keys()])

        params = {}
        for name, table in create_tables.items():
            # Ensure table variables are all lists
            params[name.lower()] = {
                'variables': {k: (v if isinstance(v, list) else [v]) for k, v in table['VARIABLES'].items()},
                'mapped': table.get('mapped_fields', []),
                'filter': table.get('filter_rows', []),
                'join_on': table.get('JOIN_ON', []),
                'index_col': table.get('index_col', []),
                'zone_col': table.get('zone_col', []),
                'rename_columns': table.get('rename_columns', [])
            }

        # Add in the zone variables
        # TODO need to implement the crosswalk join downstream for 3 zone
        zone_list = {z: (land_use_df.index.tolist() if z == land_use_df.index.name
                         else land_use_df[z].tolist())
                     for z in self.model_settings['zones']}

        # Add zones to households dicts as vary_on variable
        params['proto_households']['variables'] = {**params['proto_households']['variables'], **zone_list}

        return params

    def generate_replicates(self, table_name):
        """
        Generates replicates finding the cartesian product of the non-mapped field variables.
        The mapped fields are then annotated after replication
        """
        # Generate replicates
        df = pd.DataFrame(util.named_product(**self.params[table_name]['variables']))

        # Applying mapped variables
        if len(self.params[table_name]['mapped']) > 0:
            for mapped_from, mapped_to_pair in self.params[table_name]['mapped'].items():
                for name, mapped_to in mapped_to_pair.items():
                    df[name] = df[mapped_from].map(mapped_to)

        # Perform filter step
        if (len(self.params[table_name]['filter'])) > 0:
            for filter in self.params[table_name]['filter']:
                df[eval(filter)].reset_index(drop=True)

        return df

    def create_proto_pop(self):
        # Separate out the mapped data from the varying data and create base replicate tables
        klist = ['proto_households', 'proto_persons', 'proto_tours']
        households, persons, tours = [self.generate_replicates(k) for k in klist]

        # Names
        households.name, persons.name, tours.name = klist

        # Create ID columns, defaults to "%tablename%_id"
        hhid, perid, tourid = [self.params[x]['index_col']
                               if len(self.params[x]['index_col']) > 0
                               else x + '_id' for x in klist]

        # Create hhid
        households[hhid] = households.index + 1
        households['household_serial_no'] = households[hhid]

        # Assign persons to households
        rep = pd.DataFrame(
            util.named_product(hhid=households[hhid], index=persons.index)
        ).set_index('index').rename(columns={'hhid': hhid})
        persons = rep.join(persons).sort_values(hhid).reset_index(drop=True)
        persons[perid] = persons.index + 1

        # Assign persons to tours
        tkey, pkey = list(self.params['proto_tours']['join_on'].items())[0]
        tours = tours.merge(persons[[pkey, hhid, perid]], left_on=tkey, right_on=pkey)
        tours.index = tours.index.set_names([tourid])
        tours = tours.reset_index().drop(columns=[pkey])

        # Set index
        households.set_index(hhid, inplace=True, drop=True)
        persons.set_index(perid, inplace=True, drop=True)
        tours.set_index(tourid, inplace=True, drop=True)

        # Store tables
        self.proto_pop = {'proto_households': households, 'proto_persons': persons, 'proto_tours': tours}

        # Rename any columns. Do this first before any annotating
        for tablename, df in self.proto_pop.items():
            colnames = self.params[tablename]['rename_columns']
            if len(colnames) > 0:
                df.rename(columns=colnames, inplace=True)

    def inject_tables(self):
        # Update canonical tables lists
        inject.add_injectable('traceable_tables',
                              inject.get_injectable('traceable_tables') + list(self.proto_pop.keys())
                              )
        for tablename, df in self.proto_pop.items():
            inject.add_table(tablename, df)
            pipeline.get_rn_generator().add_channel(tablename, df)
            tracing.register_traceable_table(tablename, df)
            # pipeline.get_rn_generator().drop_channel(tablename)

    def annotate_tables(self):
        # Extract annotations
        for annotations in self.model_settings['annotate_proto_tables']:
            tablename = annotations['tablename']
            df = pipeline.get_table(tablename)
            assert df is not None
            assert annotations is not None
            # pipeline.get_rn_generator().add_channel(tablename, df)
            assign_columns(
                df=df,
                model_settings={**annotations['annotate'], **self.model_settings['suffixes']},
                trace_label=tracing.extend_trace_label('ProtoPop.annotate', tablename))
            pipeline.replace_table(tablename, df)
            # pipeline.get_rn_generator().drop_channel(tablename)

    def merge_persons(self, land_use_df):
        persons = pipeline.get_table('proto_persons')
        households = pipeline.get_table('proto_households')

        # For dropping any extra columns created during merge
        cols_to_use = households.columns.difference(persons.columns)

        # persons_merged to emulate the persons_merged table in the pipeline
        persons_merged = persons.reset_index().join(households[cols_to_use]).merge(
            land_use_df,
            left_on=self.params['proto_households']['zone_col'],
            right_on=self.model_settings['zones'])

        perid = self.params['proto_persons']['index_col']
        persons_merged.set_index(perid, inplace=True, drop=True)
        self.proto_pop['proto_persons_merged'] = persons_merged

        # Store in pipeline
        inject.add_table('proto_persons_merged', persons_merged)
        # pipeline.get_rn_generator().add_channel('proto_persons_merged', persons_merged)
        # pipeline.get_rn_generator().drop_channel('persons_merged')

def get_disaggregate_logsums(network_los, chunk_size, trace_hh_id):
    logsums = {}
    persons_merged = pipeline.get_table('proto_persons_merged').sort_index(inplace=False)
    disagg_model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    for model_name in ['workplace_location', 'school_location', 'non_mandatory_tour_destination']:
        trace_label = tracing.extend_trace_label(model_name, 'accessibilities')
        model_settings = config.read_model_settings(model_name + '.yaml')
        model_settings['SAMPLE_SIZE'] = disagg_model_settings.get('SAMPLE_SIZE')
        estimator = estimation.manager.begin_estimation(trace_label)
        if estimator:
            location_choice.write_estimation_specs(estimator, model_settings, model_name + '.yaml')

        # Append table references in settings with "proto_"
        # This avoids having to make duplicate copies of config files for disagg accessibilities
        model_settings = util.suffix_tables_in_settings(model_settings)

        # Include the suffix tags to pass onto downstream logsum models (e.g., tour mode choice)
        if model_settings.get('LOGSUM_SETTINGS', None):
            suffixes = util.concat_suffix_dict(disagg_model_settings.get('suffixes'))
            suffixes.insert(0, model_settings.get('LOGSUM_SETTINGS'))
            model_settings['LOGSUM_SETTINGS'] = ' '.join(suffixes)

        if model_name != 'non_mandatory_tour_destination':
            shadow_price_calculator = shadow_pricing.load_shadow_price_calculator(model_settings)
            _logsums, _ = location_choice.run_location_choice(
                persons_merged,
                network_los,
                shadow_price_calculator=shadow_price_calculator,
                want_logsums=True,
                want_sample_table=True,
                estimator=estimator,
                model_settings=model_settings,
                chunk_size=chunk_size,
                chunk_tag=trace_label,
                trace_hh_id=trace_hh_id,
                trace_label=trace_label,
                skip_choice=True)

            # Merge onto persons
            if _logsums is not None and len(_logsums.index) > 0:
                keep_cols = list(set(_logsums.columns).difference(persons_merged.columns))
                logsums[model_name] = persons_merged.merge(_logsums[keep_cols], on='person_id')

        else:
            tours = pipeline.get_table('proto_tours')
            tours = tours[tours.tour_category == 'non_mandatory']

            _logsums, _ = tour_destination.run_tour_destination(
                tours,
                persons_merged,
                want_logsums=True,
                want_sample_table=True,
                model_settings=model_settings,
                network_los=network_los,
                estimator=estimator,
                chunk_size=chunk_size,
                trace_hh_id=trace_hh_id,
                trace_label=trace_label,
                skip_choice=True
            )

            # Merge onto persons & tours
            if _logsums is not None and len(_logsums.index) > 0:
                tour_logsums = tours.merge(_logsums['logsums'].to_frame(), left_index=True, right_index=True)
                keep_cols = list(set(tour_logsums.columns).difference(persons_merged.columns))
                logsums[model_name] = \
                    persons_merged.merge(tour_logsums[keep_cols], on='person_id')

    return logsums

@inject.step()
def initialize_proto_population():
    print('INITIALIZE PROTO-POPULATION FOR DISAGGREGATE ACCESSIBILITY MODEL')

    # Synthesize the proto-population
    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')
    land_use_df = pipeline.get_table('land_use')
    ProtoPop(land_use_df, model_settings)

    return

@inject.step()
def compute_disaggregate_accessibility(network_los, chunk_size, trace_hh_id):
    """
       Compute enhanced disaggregate accessibility for user specified population segments,
       as well as each zone in land use file using expressions from accessibility_spec.

    """
    print('RUNNING DISAGGREGATE ACCESSIBILITY MODEL')

    # Synthesize the proto-population
    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    # this can now be called as a standalone model step instead, add_size_tables
    add_size_tables = model_settings.get('add_size_tables', True)
    if add_size_tables:
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        shadow_pricing.add_size_tables(model_settings.get('suffixes'))

    # Re-Register tables in this step, necessary for multiprocessing
    for tablename in ['proto_households', 'proto_persons', 'proto_tours', 'proto_persons_merged']:
        df = inject.get_table(tablename).to_frame()
        traceables = inject.get_injectable('traceable_tables')
        if tablename not in traceables:
            tracing.register_traceable_table(tablename, df)
            inject.add_injectable('traceable_tables', traceables + [tablename])
        if tablename not in pipeline.get_rn_generator().channels:
            pipeline.get_rn_generator().add_channel(tablename, df)

    # Run location choice
    logsums = get_disaggregate_logsums(network_los, chunk_size, trace_hh_id)

    # # De-register the channel so it can get re-registered with actual pop tables
    # [pipeline.get_rn_generator().drop_channel(x) for x in ['persons', 'households']]
    [pipeline.drop_table(x) for x in ['school_destination_size', 'workplace_destination_size', 'tours']]

    # Inject accessibilities into pipeline
    logsums = {k + '_accessibility': v for k, v in logsums.items()}
    [inject.add_table(k, df) for k, df in logsums.items()]

    # TODO MOVE TO WRITE TABLES??? OR JUST DELETE?
    # Override output tables to include accessibilities in write_table
    new_settings = inject.get_injectable("settings")
    if 'disaggregate_accessibility' in new_settings.get('output_tables').get('tables'):
        new_settings['output_tables']['tables'].remove('disaggregate_accessibility')
        new_settings['output_tables']['tables'] += logsums.keys()
    inject.add_injectable("settings", new_settings)

    return

@inject.step()
def initialize_disaggregate_accessibility():
    """
    This step initializes pre-computed disaggregate accessibilities and merges it onto the full synthetic population.
    Function adds merged all disaggregate accessibility tables to the pipeline but returns nothing.

    """
    # TODO NOT WORKING YET....
    trace_label = "initialize_disaggregate_accessibilities"
    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    return

@inject.step()
def disaggregate_accessibility_subprocess():
    """
    Spins up a sub process, not currently implemented
    """
    run_file = os.path.join('abm', 'models', 'disaggregate_accessibility_run.py') # No longer exists
    run_file = pkg_resources.resource_filename('activitysim', run_file)
    subprocess.run(['coverage', 'run', '-a', run_file] + sys.argv[1:], check=True, shell=True)

if __name__ == "__main__":
    # FOR TESTING PROTO-POP
    base_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples'
    acc_configs = os.path.join(base_dir, 'prototype_mtc_accessibilities/configs/disaggregate_accessibility.yaml')
    data_dir = os.path.join(base_dir, 'prototype_mtc/data')

    # Model Settings
    with open(acc_configs, 'r') as file:
        # model_settings = ordered_load(file)
        model_settings = yaml.load(file, Loader=yaml.SafeLoader)

    land_use_df = pd.read_csv(os.path.join(data_dir, 'land_use.csv'))
    land_use_df = land_use_df.rename(columns={'TAZ': 'zone_id'}).set_index('zone_id')

    PP = ProtoPop(land_use_df, model_settings, pipeline=False)


