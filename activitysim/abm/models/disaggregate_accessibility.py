import yaml
import collections
import os
import itertools
import logging
import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import mem
from activitysim.core import chunk

from activitysim.cli.run import handle_standard_args
from activitysim.cli.run import cleanup_output_files

from activitysim.abm.models import location_choice
from activitysim.abm.models.util import tour_destination
from activitysim.abm.tables import shadow_pricing
from activitysim.abm.models.util import estimation
from activitysim.core.expressions import assign_columns


logger = logging.getLogger(__name__)
INJECTABLES = ['data_dir', 'configs_dir', 'output_dir', 'settings_file_name']
MODELS = ['initialize_landuse', 'compute_disaggregate_accessibility']


# TODO enable preprocessing so the tables are filterable by row?


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


def disaggregate_location_choice(network_los, chunk_size, trace_hh_id):
    """
    Generalized location choice to produce output for accessibilities
    """

    # Work/School location choice
    def fixed_location(persons_merged, network_los, chunk_size, trace_hh_id):
        logsums = {}

        for model_name in ['workplace_location', 'school_location']:
            trace_label = tracing.extend_trace_label(model_name, 'accessibilities')
            model_settings = config.read_model_settings(model_name + '.yaml')
            shadow_price_calculator = shadow_pricing.load_shadow_price_calculator(model_settings)

            estimator = estimation.manager.begin_estimation(trace_label)
            if estimator:
                location_choice.write_estimation_specs(estimator, model_settings, model_name + '.yaml')

            choices_df, alt_logsums = location_choice.run_location_choice(
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
            if alt_logsums is not None:
                logsums[model_name + "_accessibilities"] = persons_merged.join(alt_logsums)

        return logsums

    # Non-mandatory tour destination
    def non_mandatory_location(persons_merged, network_los, chunk_size, trace_hh_id):
        logsums = {}
        trace_label = 'non_mandatory_tour_destination'
        model_settings = config.read_model_settings(trace_label + '.yaml')
        size_term_calculator = tour_destination.SizeTermCalculator(model_settings['SIZE_TERM_SELECTOR'])
        tours = pipeline.get_table('tours')
        tours = tours[tours.tour_category == 'non_mandatory']
        chooser_segment_column = model_settings.get('CHOOSER_SEGMENT_COLUMN_NAME', None)

        # maps segment names to compact (integer) ids
        # segments = model_settings['SEGMENTS']
        segments = set(tours[chooser_segment_column])

        if chooser_segment_column is None:
            assert len(segments) == 1, \
                f"CHOOSER_SEGMENT_COLUMN_NAME not specified in model_settings to slice SEGMENTS: {segments}"

        for segment_name in segments:
            segment_trace_label = tracing.extend_trace_label(trace_label, segment_name)

            if chooser_segment_column is not None:
                choosers = tours[tours[chooser_segment_column] == segment_name]
            else:
                choosers = tours.copy()

            # Note: size_term_calculator omits zones with impossible alternatives (where dest size term is zero)
            segment_destination_size_terms = size_term_calculator.dest_size_terms_df(segment_name, segment_trace_label)

            estimator = estimation.manager.begin_estimation(trace_label)
            if estimator:
                estimator.write_coefficients(model_settings=model_settings)
                # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
                estimator.write_spec(model_settings, tag='SPEC')
                estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
                estimator.write_table(inject.get_injectable('size_terms'), 'size_terms', append=False)
                estimator.write_table(inject.get_table('land_use').to_frame(), 'landuse', append=False)
                estimator.write_model_settings(model_settings, trace_label + '.yaml')

            location_sample_df = \
                tour_destination.run_destination_sample(
                    segment_name,
                    choosers,
                    persons_merged,
                    model_settings,
                    network_los,
                    segment_destination_size_terms,
                    estimator,
                    chunk_size=chunk_size,
                    trace_label=tracing.extend_trace_label(segment_trace_label, 'sample'))

            # - destination_logsums
            tour_purpose = segment_name  # tour_purpose is segment_name
            # location_sample_df = \
            alt_logsums = tour_destination.run_destination_logsums(
                    tour_purpose,
                    persons_merged,
                    location_sample_df,
                    model_settings,
                    network_los,
                    chunk_size=chunk_size,
                    trace_label=tracing.extend_trace_label(segment_trace_label, 'logsums'))

            print(persons_merged, alt_logsums)
            # Merge onto persons
            if alt_logsums is not None:
                logsums[trace_label + "_accessibilities"] = \
                    persons_merged.merge(alt_logsums, left_index=True, right_on='person_id')

            return logsums

    # interaction_sample expects chooser index to be monotonic increasing
    persons_merged = pipeline.get_table('persons_merged').sort_index(inplace=False)

    # Create handy kwargs
    kw = ['persons_merged', 'network_los', 'chunk_size', 'trace_hh_id']
    kwargs = {n: v for n, v in locals().items() if n in kw}

    logsums = fixed_location(**kwargs)
    logsums = {**logsums, **non_mandatory_location(**kwargs)}

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

    # Segment vars
    segment_vars = []
    for v in ['persons', 'households']:
        segment_vars.extend(list(proto.params[v]['variables'].keys()))
        # Drop zone and index vars
        _id = proto.model_settings['zones'] + [proto.params[v]['index_col'], proto.params[v]['zone_col']]
        segment_vars = list(set(segment_vars).difference(set(filter(None, _id))))
    segment_vars.extend(['pick_count', 'mode_choice_logsum'])

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    # this can now be called as a standalone model step instead, add_size_tables
    add_size_tables = model_settings.get('add_size_tables', True)
    if add_size_tables:
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        shadow_pricing.add_size_tables()

    # Run location choice
    logsums = disaggregate_location_choice(network_los, chunk_size, trace_hh_id)

    # Clean up to keep only our target vars
    logsums = {k: v[segment_vars] for k, v in logsums.items()}

    # Inject accessibilities into pipeline
    [inject.add_table(k, df) for k, df in logsums.items()]

    return


@inject.step()
def initialize_disaggregate_acessibility():
    pass

# Modified 'run' function from activitysim.cli.run to override the models list in settings.yaml with MODELS list above
# and run only the compute_disaggregate_accessibility step but retain all other main model settings.
# This enables it to be run as either a model step, or a one-off model.
# example model run is in examples/example_mtc_accessibility/disaggregate_accessibility_model.py
def run_disaggregate_accessibility(args):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    returns:
        int: sys.exit exit code
    """

    # register abm steps and other abm-specific injectables
    # by default, assume we are running activitysim.abm
    # other callers (e.g. populationsim) will have to arrange to register their own steps and injectables
    # (presumably) in a custom run_simulation.py instead of using the 'activitysim run' command
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm  # register abm steps and other abm-specific injectables

    tracing.config_logger(basic=True)
    handle_standard_args(args)  # possibly update injectables

    # cleanup
    cleanup_output_files()

    tracing.config_logger(basic=False)  # update using possibly new logging configs
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # directories
    for k in ['configs_dir', 'settings_file_name', 'data_dir', 'output_dir']:
        logger.info('SETTING %s: %s' % (k, inject.get_injectable(k, None)))

    log_settings = inject.get_injectable('log_settings', {})
    for k in log_settings:
        logger.info('SETTING %s: %s' % (k, config.setting(k)))

    # OMP_NUM_THREADS: openmp
    # OPENBLAS_NUM_THREADS: openblas
    # MKL_NUM_THREADS: mkl
    for env in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        logger.info(f"ENV {env}: {os.getenv(env)}")

    np_info_keys = [
        'atlas_blas_info',
        'atlas_blas_threads_info',
        'atlas_info',
        'atlas_threads_info',
        'blas_info',
        'blas_mkl_info',
        'blas_opt_info',
        'lapack_info',
        'lapack_mkl_info',
        'lapack_opt_info',
        'mkl_info']

    for cfg_key in np_info_keys:
        info = np.__config__.get_info(cfg_key)
        if info:
            for info_key in ['libraries']:
                if info_key in info:
                    logger.info(f"NUMPY {cfg_key} {info_key}: {info[info_key]}")

    t0 = tracing.print_elapsed_time()

    try:
        # if config.setting('multiprocess', False):
        #     logger.info('run multiprocess simulation')
        #
        #     from activitysim.core import mp_tasks
        #     injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
        #     mp_tasks.run_multiprocess(injectables)
        #
        #     assert not pipeline.is_open()
        #
        #     if config.setting('cleanup_pipeline_after_run', False):
        #         pipeline.cleanup_pipeline()
        #
        # else:
            logger.info('run single process simulation')

            pipeline.run(models=MODELS, resume_after=None)
            destination_models = ['workplace_location', 'school_location', 'non_mandatory_tour_destination']

            # workplace, school, etc.
            if 'acc_to_csv' in args and args.acc_to_csv:
                for acc_name in destination_models:
                    acc_name += '_accessibilities'
                    if pipeline.is_table(acc_name):
                        acc_model = pipeline.get_table(acc_name)
                        outpath = os.path.join(args.output, "{}.csv".format(acc_name))
                        acc_model.to_csv(outpath, index=True)
                        print("Wrote {} lines to {}".format(len(acc_model), outpath))
                    else:
                        print("No data in {}, skipping".format(acc_name))

            if config.setting('cleanup_pipeline_after_run', False):
                pipeline.cleanup_pipeline()  # has side effect of closing open pipeline
            else:
                pipeline.close_pipeline()

            mem.log_global_hwm()  # main process
    except Exception:
        # log time until error and the error traceback
        tracing.print_elapsed_time('all models until this error', t0)
        logger.exception('activitysim run encountered an unrecoverable error')
        raise

    chunk.consolidate_logs()
    mem.consolidate_logs()

    tracing.print_elapsed_time('all models', t0)

    return 0

if __name__ == "__main__":
    # FOR TESTING PROTO-POP
    base_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples'
    acc_configs = os.path.join(base_dir, 'prototype_mtc_accessibilities/configs/disaggregate_accessibility.yaml')
    data_dir = os.path.join(base_dir, 'example_mtc/data')

    # Model Settings
    with open(acc_configs, 'r') as file:
        # model_settings = ordered_load(file)
        model_settings = yaml.load(file, Loader=yaml.SafeLoader)

    land_use_df = pd.read_csv(os.path.join(data_dir, 'land_use.csv'))
    land_use_df = land_use_df.rename(columns={'TAZ': 'zone_id'}).set_index('zone_id')

    PP = ProtoPop(land_use_df, model_settings, pipeline=False)