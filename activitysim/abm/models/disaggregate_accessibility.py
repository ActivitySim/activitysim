import sys

import orca
import pandas as pd
import yaml
import collections
import os
import itertools
import logging
import subprocess

from activitysim.core import config
from activitysim.core import assign
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import los

from activitysim.abm.models import location_choice
from activitysim.abm.tables import shadow_pricing
from activitysim.abm.models.util import estimation

# TODO enable preprocessing so the tables are filterable by row
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
        rep = pd.DataFrame(named_product(**self.params[table_name]['variables']))

        # Applying mapped variables
        if len(self.params[table_name]['mapped']) > 0:
            for mapped_from, mapped_to_pair in self.params[table_name]['mapped'].items():
                for name, mapped_to in mapped_to_pair.items():
                    rep[name] = rep[mapped_from].map(mapped_to)

        return rep

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

    def annotate_tables(self):
        # Extract annotations
        for annotations in self.model_settings['annotate_proto_tables']:
            tablename = annotations['tablename']
            df = pipeline.get_table(tablename)
            assert df is not None
            assert annotations is not None
            pipeline.get_rn_generator().add_channel(tablename, df)
            assign_columns(
                df=df,
                model_settings=annotations['annotate'],
                trace_label=tracing.extend_trace_label('ProtoPop.annotate', tablename))
            pipeline.replace_table(tablename, df)
            pipeline.get_rn_generator().drop_channel(tablename)

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
        persons_merged.set_index(perid, inplace=True, drop=False)
        self.proto_pop['persons_merged'] = persons_merged

        # Store in pipeline
        inject.add_table('persons_merged', persons_merged)
        pipeline.get_rn_generator().add_channel('persons_merged', persons_merged)
        # pipeline.get_rn_generator().drop_channel('persons_merged')


def disaggregate_location_choice(segment_vars, network_los, chunk_size, trace_hh_id, locutor):
    """
    Generalized workplace/school_location choice to produce output for accessibilities

    iterate_location_choice adds location choice column and annotations to persons table
    """

    persons_merged = pipeline.get_table('persons_merged')
    persons = pipeline.get_table('persons')
    households = pipeline.get_table('households')


    logsums = {}

    # outer segment loop here
    for segment_name, choosers in persons_merged.groupby(segment_vars):

        for trace_label in ['workplace_location', 'school_location']:
            model_settings = config.read_model_settings(trace_label + '.yaml')

            # Get model params
            estimator = estimation.manager.begin_estimation(trace_label)
            if estimator:
                location_choice.write_estimation_specs(estimator, model_settings, trace_label + '.yaml')

            shadow_price_calculator = shadow_pricing.load_shadow_price_calculator(model_settings)

            # size_term and shadow price adjustment - one row per zone
            dest_size_terms = shadow_price_calculator.dest_size_terms(segment_name)

            assert dest_size_terms.index.is_monotonic_increasing, \
                f"shadow_price_calculator.dest_size_terms({segment_name}) not monotonic_increasing"

            if choosers.shape[0] == 0:
                logger.info(f"{trace_label} skipping segment {segment_name}: no choosers")
                continue

            # no chunking in this case?
            chunk_tag = trace_label

            # TODO Need to get logsums of all alternatives, not the selected values
            # logsums[_label] = \
            #     location_choice.iterate_location_choice(
            #         model_settings,
            #         persons_merged, persons, households,
            #         network_los,
            #         estimator,
            #         chunk_size, trace_hh_id, locutor, _label
            #     )

            # - location_sample
            location_sample_df = \
                location_choice.run_location_sample(
                    segment_name,
                    choosers,
                    network_los,
                    dest_size_terms,
                    estimator,
                    model_settings,
                    chunk_size,
                    chunk_tag,
                    trace_label=tracing.extend_trace_label(trace_label, 'sample.%s' % segment_name))

            # - location_logsums
            location_sample_df = \
                location_choice.run_location_logsums(
                    segment_name,
                    choosers,
                    network_los,
                    location_sample_df,
                    model_settings,
                    chunk_size, chunk_tag=f'{chunk_tag}.logsums',
                    trace_label=tracing.extend_trace_label(trace_label, 'logsums.%s' % segment_name))


        # Nonmandatory destination
        # location_sample_df = \
        #     run_destination_sample(
        #         spec_segment_name,
        #         choosers,
        #         persons_merged,
        #         model_settings,
        #         network_los,
        #         segment_destination_size_terms,
        #         estimator,
        #         chunk_size=chunk_size,
        #         trace_label=tracing.extend_trace_label(segment_trace_label, 'sample'))
        #
        # # - destination_logsums
        # tour_purpose = segment_name  # tour_purpose is segment_name
        # location_sample_df = \
        #     run_destination_logsums(
        #         tour_purpose,
        #         persons_merged,
        #         location_sample_df,
        #         model_settings,
        #         network_los,
        #         chunk_size=chunk_size,
        #         trace_label=tracing.extend_trace_label(segment_trace_label, 'logsums'))

        logsums[trace_label] = location_sample_df
        # TEST
        base_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples/example_mtc_accessibilities'
        logsums[trace_label].to_csv(os.path.join(base_dir, 'output', trace_label + '.csv'))

        if estimator:
            estimator.end_estimation()

@inject.step()
def compute_disaggregate_accessibility(network_los, chunk_size, trace_hh_id, locutor):
    """
       Compute enhanced disaggregate accessibility for user specified population segments,
       as well as each zone in land use file using expressions from accessibility_spec.

    """

    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')

    # Initialize land_use
    land_use_df = pipeline.get_table('land_use')

    # Synthesize the proto-population
    PP = ProtoPop(land_use_df, model_settings)

    # Segment vars
    segment_vars = []
    for v in ['persons', 'households']:
        segment_vars.extend(list(PP.params[v]['variables'].keys()))
        # Drop zone and index vars
        _id = PP.model_settings['zones'] + [PP.params[v]['index_col'], PP.params[v]['zone_col']]
        segment_vars = list(set(segment_vars).difference(set(filter(None, _id))))

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    # this can now be called as a stand alone model step instead, add_size_tables
    add_size_tables = model_settings.get('add_size_tables', True)
    if add_size_tables:
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        shadow_pricing.add_size_tables()

    # Run location choice
    disaggregate_location_choice(segment_vars, network_los, chunk_size, trace_hh_id, locutor)

# def save_output(self):
#     OUTPUT_PREFIX = 'accessibilities'
#     for name, model in self.output.items():
#         outfile = "{}_{}.csv".format(OUTPUT_PREFIX, name)
#         model.to_csv(outfile, index=False)
#         print("Wrote {} lines to {}".format(len(model), outfile))


if __name__ == "__main__":
    # FOR TESTING
    base_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples'
    acc_configs = os.path.join(base_dir, 'example_mtc_accessibilities/configs/disaggregate_accessibility.yaml')
    data_dir = os.path.join(base_dir, 'example_mtc/data')

    # Model Settings
    with open(acc_configs, 'r') as file:
        # model_settings = ordered_load(file)
        model_settings = yaml.load(file, Loader=yaml.SafeLoader)

    land_use_df = pd.read_csv(os.path.join(data_dir, 'land_use.csv'))
    land_use_df = land_use_df.rename(columns={'TAZ': 'zone_id'}).set_index('zone_id')
    # trace_hh_id = 982875
    # locutor = True
    # network_los = los.Network_LOS(los_settings_file_name='network_los.yaml').load_data()

    PP = ProtoPop(land_use_df, model_settings, pipeline=False)

