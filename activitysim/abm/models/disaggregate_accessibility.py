import yaml
import collections
import os
import sys
import itertools
import logging
import subprocess
import pkg_resources
import pandas as pd

from activitysim.core import (inject,
                              tracing,
                              config,
                              pipeline,
                              chunk)

from activitysim.abm.models import location_choice
from activitysim.abm.models.util import (tour_destination, estimation)
from activitysim.abm.tables import shadow_pricing
from activitysim.core.expressions import assign_columns

logger = logging.getLogger(__name__)

# Generic helper functions
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=collections.OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def named_product(**d):
    names = d.keys()
    vals = d.values()
    for res in itertools.product(*vals):
        yield dict(zip(names, res))


class ProtoPop:
    def __init__(self, land_use_df, model_settings, pipeline=True):
        self.model_settings = model_settings
        self.params = self.read_table_settings(land_use_df)
        self.create_proto_pop()

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
        params['households']['variables'] = {**params['households']['variables'], **zone_list}

        return params

    def generate_replicates(self, table_name):
        """
        Generates replicates finding the cartesian product of the non-mapped field variables.
        The mapped fields are then annotated after replication
        """
        # Generate replicates
        df = pd.DataFrame(named_product(**self.params[table_name]['variables']))

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
        klist = ['households', 'persons', 'tours']
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
            named_product(hhid=households[hhid], index=persons.index)
        ).set_index('index').rename(columns={'hhid': hhid})
        persons = rep.join(persons).sort_values(hhid).reset_index(drop=True)
        persons[perid] = persons.index + 1

        # Assign persons to tours
        tkey, pkey = list(self.params['tours']['join_on'].items())[0]
        tours = tours.merge(persons[[pkey, hhid, perid]], left_on=tkey, right_on=pkey)
        tours.index = tours.index.set_names([tourid])
        tours = tours.reset_index().drop(columns=[pkey])

        # Set index
        households.set_index(hhid, inplace=True, drop=False)
        persons.set_index(perid, inplace=True, drop=False)
        tours.set_index(tourid, inplace=True, drop=False)

        # Store tables
        self.proto_pop = {'households': households, 'persons': persons, 'tours': tours}

        # Rename any columns. Do this first before any annotating
        for tablename, df in self.proto_pop.items():
            colnames = self.params[tablename]['rename_columns']
            if len(colnames) > 0:
                df.rename(columns=colnames, inplace=True)

    def inject_tables(self):
        for tablename, df in self.proto_pop.items():
            inject.add_table(tablename, df)
            pipeline.get_rn_generator().add_channel(tablename, df)
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
                model_settings=annotations['annotate'],
                trace_label=tracing.extend_trace_label('ProtoPop.annotate', tablename))
            pipeline.replace_table(tablename, df)
            # pipeline.get_rn_generator().drop_channel(tablename)

    def merge_persons(self, land_use_df):
        persons = pipeline.get_table('persons')
        households = pipeline.get_table('households')

        # For dropping any extra columns created during merge
        cols_to_use = households.columns.difference(persons.columns)

        # persons_merged to emulate the persons_merged table in the pipeline
        persons_merged = persons.join(households[cols_to_use]).merge(
            land_use_df,
            left_on=self.params['households']['zone_col'],
            right_on=self.model_settings['zones'])

        perid = self.params['persons']['index_col']
        persons_merged.set_index(perid, inplace=True, drop=True)
        self.proto_pop['persons_merged'] = persons_merged

        # Store in pipeline
        inject.add_table('persons_merged', persons_merged)
        pipeline.get_rn_generator().add_channel('persons_merged', persons_merged)
        # pipeline.get_rn_generator().drop_channel('persons_merged')

def get_disaggregate_logsums(network_los, chunk_size, trace_hh_id):
    logsums = {}
    persons_merged = pipeline.get_table('persons_merged').sort_index(inplace=False)

    for model_name in ['workplace_location', 'school_location', 'non_mandatory_tour_destination']:
        trace_label = tracing.extend_trace_label(model_name, 'accessibilities')
        model_settings = config.read_model_settings(model_name + '.yaml')
        model_settings['SAMPLE_SIZE'] = 0
        estimator = estimation.manager.begin_estimation(trace_label)
        if estimator:
            location_choice.write_estimation_specs(estimator, model_settings, model_name + '.yaml')

        if model_name is not 'non_mandatory_tour_destination':
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
                trace_label=trace_label)

            # Merge onto persons
            if _logsums is not None:
                logsums[model_name + "_accessibilities"] = persons_merged.join(_logsums)
        else:
            tours = pipeline.get_table('tours')
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
            if _logsums is not None:
                tour_logsums = tours.merge(_logsums['logsums'].to_frame(), left_index=True, right_index=True)
                keep_cols = set(tour_logsums.columns).difference(persons_merged.columns)
                logsums[trace_label + "_accessibilities"] = \
                    persons_merged.merge(tour_logsums[keep_cols], left_on="person_id", right_on='person_id')

    return logsums

@inject.step()
def compute_disaggregate_accessibility(network_los, chunk_size, trace_hh_id):
    """
       Compute enhanced disaggregate accessibility for user specified population segments,
       as well as each zone in land use file using expressions from accessibility_spec.

    """
    print('RUNNING DISAGGREGATE ACCESSIBILITY MODEL')

    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    # Initialize land_use
    land_use_df = pipeline.get_table('land_use')

    # Synthesize the proto-population
    proto = ProtoPop(land_use_df, model_settings)

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    # this can now be called as a standalone model step instead, add_size_tables
    add_size_tables = model_settings.get('add_size_tables', True)
    if add_size_tables:
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        shadow_pricing.add_size_tables()

    # Run location choice
    # logsums = disaggregate_location_choice(network_los, chunk_size, trace_hh_id)
    logsums = get_disaggregate_logsums(network_los, chunk_size, trace_hh_id)

    if model_settings.get('trim_output', False):
        # Segment vars
        segment_vars = []
        for v in ['persons', 'households']:
            segment_vars.extend(list(proto.params[v]['variables'].keys()))
            # Drop zone and index vars
            _id = proto.model_settings['zones'] + [proto.params[v]['index_col'], proto.params[v]['zone_col']]
            segment_vars = list(set(segment_vars).difference(set(filter(None, _id))))
        segment_vars.extend(['alt_dest', 'pick_count', 'mode_choice_logsum'])

        # Clean up to keep only our target vars
        logsums = {k: v[segment_vars] for k, v in logsums.items()}

    # Inject accessibilities into pipeline
    [inject.add_table(k, df) for k, df in logsums.items()]

    # Override output tables to include accessibilities
    new_settings = inject.get_injectable("settings")
    new_settings['output_tables']['tables'] = list(logsums.keys())
    new_settings['output_tables']['prefix'] = ""

    inject.add_injectable("settings", new_settings)

    return

@inject.step()
def initialize_disaggregate_accessibility():
    """
    This step initializes pre-computed disaggregate accessibilities and merges it onto the full synthetic population.
    Function adds merged all disaggregate accessibility tables to the pipeline but returns nothing.

    """
    trace_label = "initialize_disaggregate_accessibilities"

    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    with chunk.chunk_log(trace_label, base=True):

        chunk.log_rss(f"{trace_label}.inside-yield")

        # TODO 1) load tables, 2) merge with full pop, this can be done on the injection side in tables/disaggregate_accessibility
        # Load disaggregate accessibilities and merge
        # {k: inject.get_table(k).to_frame() for k in model_settings.get('initialize_tables')}

    return

@inject.step()
def disaggregate_accessibility_subprocess():
    run_file = os.path.join('abm', 'models', 'disaggregate_accessibility_run.py')
    run_file = pkg_resources.resource_filename('activitysim', run_file)

    subprocess.run(['coverage', 'run', '-a', run_file] + sys.argv[1:], check=True, shell=True)
    # subprocess.Popen(['python', '-u', __file__] + sys.argv[1:], stdout=sys.stdout, stderr=subprocess.PIPE)


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


