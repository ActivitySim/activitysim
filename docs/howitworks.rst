
How the System Works
====================

This page describes how the software works, how multiprocessing works, and the primary example model data schema.  The code snippets below may not exactly match the latest version of the software, but they are close enough to illustrate how the system works.  

.. _how_the_system_works:

Execution Flow
--------------

The example model run starts by running the steps in :ref:`example_run`.

Initialization
~~~~~~~~~~~~~~

The first significant step of the ``run`` command is:

::

  from activitysim import abm

which loads :mod:`activitysim.abm.__init__`, which calls:

::

   import misc
   import tables
   import models

which then loads the misc, tables, and models class definitions.  Loading :mod:`activitysim.abm.misc` calls:

::

   from activitysim.core import config
   from activitysim.core import inject

which loads the config and inject classes.  These define Inject injectables (functions) and
helper functions for running models.  For example, the Python decorator ``@inject.injectable`` overrides the function definition ``settings`` to
execute this function whenever the ``settings`` object is called by the system.  The :mod:`activitysim.core.inject` manages the data
pipeline.

::

   @inject.injectable(cache=True)
   def settings():
       settings_dict = read_settings_file('settings.yaml', mandatory=True)
       return settings_dict

Next, the tables module executes the following import statements in :mod:`activitysim.abm.tables.__init__` to
define the dynamic inject tables (households, persons, skims, etc.), but does not load them. It also defines the
core dynamic injectables (functions) defined in the classes. The Python decorator ``@inject.table`` override the function
definitions so the name of the function becomes the name of the table when dynamically called by the system.

::

  from . import households
  from . import persons
  #etc...

  #then in households.py
  @inject.table()
  def households(households_sample_size, override_hh_ids, trace_hh_id):

The models module then loads all the sub-models, which are registered as model steps with
the ``@inject.step()`` decorator.  These steps will eventually be run by the data pipeliner.

::

  from . import accessibility
  from . import atwork_subtour_destination
  from . import atwork_subtour_frequency
  #etc...

  #then in accessibility.py
  @inject.step()
  def compute_accessibility(accessibility, network_los, land_use, trace_od):

Back in the main ``run`` command, the next steps are to load the tracing, configuration, setting, and pipeline classes
to get the system management components up and running.

::

  from activitysim.core import tracing
  from activitysim.core import config
  from activitysim.core import pipeline


The next step in the example is to define the ``run`` method, call it if the script is being run as the program entry point, and handle the
arguments passed in via the command line.

::

  def run():
    #etc...

  if __name__ == '__main__':
    run()


.. note::
   For more information on run options, type ``activitysim run -h`` on the command line


The first key thing that happens in the ``run`` function is ``resume_after = setting('resume_after', None)``, which causes the system
to go looking for ``setting``.  Earlier we saw that ``setting`` was defined as an injectable and so the system gets this object if it
is already in memory, or if not, calls this function which loads the ``config/settings.yaml`` file.  This is called lazy loading or
on-demand loading. Next, the system loads the models list and starts the pipeline:

::

  pipeline.run(models=setting('models'), resume_after=resume_after)

The :func:`activitysim.core.pipeline.run` method loops through the list of models, calls ``inject.run([step_name])``,
and manages the data pipeline.  The first disaggregate data processing step (or model) run is ``initialize_households``, defined in
:mod:`activitysim.abm.models.initialize`.  The ``initialize_households`` step is responsible for requesting reading of the raw
households and persons into memory.

Initialize Households
~~~~~~~~~~~~~~~~~~~~~

The initialize households step/model is run via:

::

   @inject.step()
   def initialize_households():

      trace_label = 'initialize_households'
      model_settings = config.read_model_settings('initialize_households.yaml', mandatory=True)
      annotate_tables(model_settings, trace_label)

This step reads the ``initialize_households.yaml`` config file, which defines the :ref:`table_annotation` below.  Each table
annotation applies the expressions specified in the annotate spec to the relevant table.  For example, the ``persons`` table
is annotated with the results of the expressions in ``annotate_persons.csv``.  If the table is not already in memory, then
inject goes looking for it as explained below.

::

   #initialize_households.yaml
   annotate_tables:
     - tablename: persons
       annotate:
         SPEC: annotate_persons
         DF: persons
         TABLES:
           - households
     - tablename: households
       column_map:
         PERSONS: hhsize
         workers: num_workers
       annotate:
         SPEC: annotate_households
         DF: households
         TABLES:
           - persons
           - land_use
     - tablename: persons
       annotate:
         SPEC: annotate_persons_after_hh
         DF: persons
         TABLES:
           - households

   #initialize.py
   def annotate_tables(model_settings, trace_label):

    annotate_tables = model_settings.get('annotate_tables', [])

    for table_info in annotate_tables:

        tablename = table_info['tablename']
        df = inject.get_table(tablename).to_frame()

        # - annotate
        annotate = table_info.get('annotate', None)
        if annotate:
            logger.info("annotated %s SPEC %s" % (tablename, annotate['SPEC'],))
            expressions.assign_columns(
                df=df,
                model_settings=annotate,
                trace_label=trace_label)

        # - write table to pipeline
        pipeline.replace_table(tablename, df)


Remember that the ``persons`` table was previously registered as an injectable table when the persons table class was
imported.  Now that the ``persons`` table is needed, inject calls this function, which requires the ``households`` and
``trace_hh_id`` objects as well.  Since ``households`` has yet to be loaded, the system run the households inject table operation
as well.  The various calls also setup logging, tracing, stable random number management, etc.

::

  #persons in persons.py requires households, trace_hh_id
  @inject.table()
  def persons(households, trace_hh_id):

    df = read_raw_persons(households)

    logger.info("loaded persons %s" % (df.shape,))

    df.index.name = 'person_id'

    # replace table function with dataframe
    inject.add_table('persons', df)

    pipeline.get_rn_generator().add_channel('persons', df)

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "raw.persons", warn_if_empty=True)

    return df

  #households requires households_sample_size, override_hh_ids, trace_hh_id
  @inject.table()
  def households(households_sample_size, override_hh_ids, trace_hh_id):

    df_full = read_input_table("households")


The process continues until all the dependencies are resolved.  It is the ``read_input_table`` function that
actually reads the input tables from the input HDF5 or CSV file using the ``input_table_list`` found in ``settings.yaml``

::

  input_table_list:
    - tablename: households
      filename: households.csv
      index_col: household_id
      column_map:
        HHID: household_id

School Location
~~~~~~~~~~~~~~~

Now that the persons, households, and other data are in memory, and also annotated with additional fields
for later calculations, the school location model can be run.  The school location model is defined
in :mod:`activitysim.abm.models.location_choice`.  As shown below, the school location model
actually uses the ``persons_merged`` table, which includes joined household, land use, and accessibility
tables as well.  The school location model also requires the network_los object, which is discussed next.
Before running the generic iterate location choice function, the model reads the model settings file, which
defines various settings, including the expression files, sample size, mode choice logsum
calculation settings, time periods for skim lookups, shadow pricing settings, etc.

::

   #persons.py
   # another common merge for persons
   @inject.table()
   def persons_merged(persons, households, land_use, accessibility):
        return inject.merge_tables(persons.name, tables=[persons, households, land_use, accessibility])

   #location_choice.py
   @inject.step()
   def school_location(
        persons_merged, persons, households,
        network_los, chunk_size, trace_hh_id, locutor
        ):

     trace_label = 'school_location'
     model_settings = config.read_model_settings('school_location.yaml')

     iterate_location_choice(
        model_settings,
        persons_merged, persons, households,
        network_los,
        chunk_size, trace_hh_id, locutor, trace_label


Deep inside the method calls, the skim matrix lookups required for this model are configured via ``network_los``. The following
code sets the keys for looking up the skim values for this model. In this case there is a ``TAZ`` column
in the households table that is renamed to `TAZ_chooser`` and a ``TAZ`` in the alternatives generation code.
The skims are lazy loaded under the name "skims" and are available in the expressions using the ``@skims`` expression.

::

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap('home_zone_id', 'zone_id')

    locals_d = {
        'skims': skims,
    }

The next step is to call the :func:`activitysim.core.interaction_sample.interaction_sample` function which
selects a sample of alternatives by running a MNL choice model simulation in which alternatives must be
merged with choosers because there are interaction terms.  The choosers table, the alternatives table, the
sample size, the model specification expressions file, the skims, the skims lookups, the chunk size, and the
trace labels are passed in.

::

    #interaction_sample
    choices = interaction_sample(
       choosers,
       alternatives,
       sample_size=sample_size,
       alt_col_name=alt_dest_col_name,
       spec=spec_for_segment(model_spec, segment_name),
       skims=skims,
       locals_d=locals_d,
       chunk_size=chunk_size,
       trace_label=trace_label)

This function solves the utilities, calculates probabilities, draws random numbers, selects choices with
replacement, and returns the choices. This is done in a for loop of chunks of chooser records in order to avoid
running out of RAM when building the often large data tables. This method does a lot, and eventually
calls :func:`activitysim.core.interaction_simulate.eval_interaction_utilities`, which loops through each
expression in  the expression file and solves it at once for all records in the chunked chooser
table using Python's ``eval``.

The :func:`activitysim.core.interaction_sample.interaction_sample` method is currently only a multinomial
logit choice model.  The :func:`activitysim.core.simulate.simple_simulate` method supports both MNL and NL as specified by
the ``LOGIT_TYPE`` setting in the model settings YAML file.   The ``auto_ownership.yaml`` file for example specifies
the ``LOGIT_TYPE`` as ``MNL.``

If the expression is a skim matrix, then the entire column of chooser OD pairs is retrieved from the matrix (i.e. numpy array)
in one vectorized step.  The ``orig`` and ``dest`` objects in ``self.data[orig, dest]`` in :mod:`activitysim.core.los` are vectors
and selecting numpy array items with vector indexes returns a vector.  Trace data is also written out if configured (not shown below).

::

    # evaluate expressions from the spec multiply by coefficients and sum
    interaction_utilities, trace_eval_results \
        = eval_interaction_utilities(spec, interaction_df, locals_d, trace_label, trace_rows)

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.values.reshape(len(choosers), alternative_count),
        index=choosers.index)

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, allow_zero_probs=allow_zero_probs,
                                 trace_label=trace_label, trace_choosers=choosers)

    choices_df = make_sample_choices(
        choosers, probs, alternatives, sample_size, alternative_count, alt_col_name,
        allow_zero_probs=allow_zero_probs, trace_label=trace_label)

    # pick_count is number of duplicate picks
    pick_group = choices_df.groupby([choosers.index.name, alt_col_name])

    # number each item in each group from 0 to the length of that group - 1.
    choices_df['pick_count'] = pick_group.cumcount(ascending=True)
    # flag duplicate rows after first
    choices_df['pick_dup'] = choices_df['pick_count'] > 0
    # add reverse cumcount to get total pick_count (conveniently faster than groupby.count + merge)
    choices_df['pick_count'] += pick_group.cumcount(ascending=False) + 1

    # drop the duplicates
    choices_df = choices_df[~choices_df['pick_dup']]
    del choices_df['pick_dup']

    return choices_df

The model creates the ``location_sample_df`` table using the choices above.  This table is
then used for the next model step - solving the logsums for the sample.

::

     # - location_logsums
     location_sample_df = run_location_logsums(
                segment_name,
                choosers,
                network_los,
                location_sample_df,
                model_settings,
                chunk_size,
                trace_hh_id,
                tracing.extend_trace_label(trace_label, 'logsums.%s' % segment_name))

The next steps are similar to what the sampling model does, except this time the sampled locations
table is the choosers and the model is calculating and adding the tour mode choice logsums using the
logsums settings and expression files.  The resulting logsums are added to the chooser table as the
``mode_choice_logsum`` column.

::

    #inside run_location_logsums() defined in location_choice.py
    logsums = logsum.compute_logsums(
       choosers,
       tour_purpose,
       logsum_settings, model_settings,
       network_los,
       chunk_size,
       trace_label)

    location_sample_df['mode_choice_logsum'] = logsums

The :func:`activitysim.abm.models.util.logsums.compute_logsums` method goes through a similar series
of steps as the interaction_sample function but ends up calling
:func:`activitysim.core.simulate.simple_simulate_logsums` since it supports nested logit models, which
are required for the mode choice logsum calculation.  The
:func:`activitysim.core.simulate.simple_simulate_logsums` returns a vector of logsums (instead of a vector
choices).

The final school location choice model operates on the ``location_sample_df`` table created
above and is called as follows:

::

	  # - location_simulate
	  choices = \
	      run_location_simulate(
	          segment_name,
	          choosers,
	          location_sample_df,
	          network_los,
	          dest_size_terms,
	          model_settings,
	          chunk_size,
	          tracing.extend_trace_label(trace_label, 'simulate.%s' % segment_name))

	  choices_list.append(choices)

The operations executed by this model are very similar to the earlier models, except
this time the sampled locations table is the choosers and the model selects one alternative for
each chooser using the school location simulate expression files and the
:func:`activitysim.core.interaction_sample_simulate.interaction_sample_simulate` function.

Back in ``iterate_location_choice()``, the model adds the choices as a column to the ``persons`` table and adds
additional output columns using a postprocessor table annotation if specified in the settings file.  Refer
to :ref:`table_annotation` for more information and the :func:`activitysim.abm.models.util.expressions.assign_columns`
function.  The overall school location model is run within a shadow pricing iterative loop as shown below.  Refer
to :ref:`shadow_pricing` for more information.

::


   # in iterate_location_choice() in location_choice.py
	 for iteration in range(1, max_iterations + 1):

        if spc.use_shadow_pricing and iteration > 1:
            spc.update_shadow_prices()

        choices = run_location_choice(
            persons_merged_df,
            network_los,
            spc,
            model_settings,
            chunk_size, trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, 'i%s' % iteration))

        choices_df = choices.to_frame('dest_choice')
        choices_df['segment_id'] = \
            persons_merged_df[chooser_segment_column].reindex(choices_df.index)

        spc.set_choices(choices_df)

        if locutor:
            spc.write_trace_files(iteration)

        if spc.use_shadow_pricing and spc.check_fit(iteration):
            logging.info("%s converged after iteration %s" % (trace_label, iteration,))
            break

    # - shadow price table
    if locutor:
        if spc.use_shadow_pricing and 'SHADOW_PRICE_TABLE' in model_settings:
            inject.add_table(model_settings['SHADOW_PRICE_TABLE'], spc.shadow_prices)
        if 'MODELED_SIZE_TABLE' in model_settings:
            inject.add_table(model_settings['MODELED_SIZE_TABLE'], spc.modeled_size)

    dest_choice_column_name = model_settings['DEST_CHOICE_COLUMN_NAME']
    tracing.print_summary(dest_choice_column_name, choices, value_counts=True)

    persons_df = persons.to_frame()

    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    NO_DEST_TAZ = -1
    persons_df[dest_choice_column_name] = \
        choices.reindex(persons_df.index).fillna(NO_DEST_TAZ).astype(int)

    # - annotate persons table
    if 'annotate_persons' in model_settings:
        expressions.assign_columns(
            df=persons_df,
            model_settings=model_settings.get('annotate_persons'),
            trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

        pipeline.replace_table("persons", persons_df)


Finishing Up
~~~~~~~~~~~~

The last models to be run by the data pipeline are:

* ``write_data_dictionary``, which writes the table_name, number of rows, number of columns, and number of bytes for each checkpointed table
* ``track_skim_usage``, which tracks skim data memory usage
* ``write_tables``, which writes pipeline tables as CSV files as specified by the output_tables setting

Back in the main ``run`` command, the final steps are to:

* close the data pipeline (and attached HDF5 file)

Additional Notes
----------------

The rest of the microsimulation models operate in a similar fashion with a few notable additions:

* creating new tables
* vectorized 3D skims indexing
* aggregate (OD-level) accessibilities model

Creating New Tables
~~~~~~~~~~~~~~~~~~~

In addition to calculating the mandatory tour frequency for a person, the model must also create mandatory tour records.
Once the number of tours is known, then the next step is to create tours records for subsequent models.  This is done by the
:func:`activitysim.abm.models.util.tour_frequency.process_tours` function, which is called by the
:func:`activitysim.abm.models.mandatory_tour_frequency.mandatory_tour_frequency` function, which adds the tours to
the ``tours`` table managed in the data pipeline.  This is the same basic pattern used for creating new tables - tours, trips, etc.

::

  @inject.step()
  def mandatory_tour_frequency(persons_merged, chunk_size, trace_hh_id):

    choosers['mandatory_tour_frequency'] = choices
      mandatory_tours = process_mandatory_tours(
        persons=choosers,
        mandatory_tour_frequency_alts=alternatives
    )

    tours = pipeline.extend_table("tours", mandatory_tours)


Vectorized 3D Skim Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mode choice model uses a collection of skims with a third dimension, which in this case
is time period.  Setting up the 3D index for skims is done as follows:

::

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    orig_col_name = 'home_zone_id'
    dest_col_name = 'destination'

    out_time_col_name = 'start'
    in_time_col_name = 'end'
    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col_name, dest_key=dest_col_name,
                                               dim3_key='out_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col_name, dest_key=orig_col_name,
                                               dim3_key='in_period')
    odr_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col_name, dest_key=dest_col_name,
                                               dim3_key='in_period')
    dor_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col_name, dest_key=orig_col_name,
                                               dim3_key='out_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name,
        'out_time_col_name': out_time_col_name,
        'in_time_col_name': in_time_col_name
    }

When model expressions such as ``@odt_skims['WLK_LOC_WLK_TOTIVT']`` are solved,
the ``WLK_LOC_WLK_TOTIVT`` skim matrix values for all chooser table origins, destinations, and
out_periods can be retrieved in one vectorized request.

All the skims are preloaded (cached) by the pipeline manager at the beginning of the model
run in order to avoid repeatedly reading the skims from the OMX files on disk.  This saves
significant model runtime.

See :ref:`los_in_detail` for more information on skim handling.

Accessibilities Model
~~~~~~~~~~~~~~~~~~~~~

Unlike the microsimulation models, which operate on a table of choosers, the accessibilities model is
an aggregate model that calculates accessibility measures by origin zone to all destination zones.  This
model could be implemented with a matrix library such as numpy since it involves a series of matrix
and vector operations.  However, all the other ActivitySim AB models - the
microsimulation models - are implemented with pandas.DataFrame tables, and so this would be a
different approach for just this model.  The benefits of keeping with the same table approach to
data setup, expression management, and solving means ActivitySim has one expression syntax, is
easier to understand and document, and is more efficiently implemented.

As illustrated below, in order to convert the
accessibility calculation into a table operation, a table of OD pairs is first built using numpy
``repeat`` and ``tile`` functions.  Once constructed, the additional data columns are added to the
table in order to solve the accessibility calculations.  The skim data is also added in column form.
After solving the expressions for each OD pair row, the accessibility module aggregates the results
to origin zone and write them to the datastore.

::

  # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(np.asanyarray(land_use_df.index), zone_count),
            'dest': np.tile(np.asanyarray(land_use_df.index), zone_count)
        }
    )


.. index:: multiprocessing

.. _multiprocessing:

Multiprocessing
---------------

Most models can be implemented as a series of independent vectorized operations on pandas DataFrames and
numpy arrays. These vectorized operations are much faster than sequential Python because they are
implemented by native code (compiled C) and are to some extent multi-threaded. But the benefits of
numpy multi-processing are limited because they only apply to atomic numpy or pandas calls, and as
soon as control returns to Python it is single-threaded and slow.

Multi-threading is not an attractive strategy to get around the Python performance problem because
of the limitations imposed by Python's global interpreter lock (GIL). Rather than struggling with
Python multi-threading, ActivitySim uses the
Python `multiprocessing <https://docs.python.org/2/library/multiprocessing.html>`__ library to parallelize
most models.

ActivitySim's modular and extensible architecture makes it possible to not hardwire the multiprocessing
architecture. The specification of which models should be run in parallel, how many processers
should be used, and the segmentation of the data between processes are all specified in the
settings config file.

Mutliprocessing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multiprocess_steps setting below indicate that the simulation should be broken into three steps.

::

    models:
      ### mp_initialize step
      - initialize_landuse
      - compute_accessibility
      - initialize_households
      ### mp_households step
      - school_location
      - workplace_location
      - auto_ownership_simulate
      - free_parking
      ### mp_summarize step
      - write_tables

    multiprocess_steps:
      - name: mp_initialize
        begin: initialize_landuse
      - name: mp_households
        begin: school_location
        num_processes: 2
        slice:
          tables:
            - households
            - persons
      - name: mp_summarize
        begin: write_tables


The first multiprocess_step, ``mp_initialize``, begins with the initialize landuse step and is
implicity single-process because there is no 'slice' key indicating how to apportion the tables.
This first step includes all models listed in the 'models' setting up until the first step
in the next multiprocess_steps.

The second multiprocess_step, ``mp_households``, starts with the school location model and continues
through auto ownership. The 'slice' info indicates that the tables should be sliced by
``households``, and that ``persons`` is a dependent table and so ``persons`` with a ref_col (foreign key
column with the same name as the ``Households`` table index) referencing a household record should be
taken to 'belong' to that household. Similarly, any other table that either share an index
(i.e. having the same name) with either the ``households`` or ``persons`` table, or have a ref_col to
either of their indexes, should also be considered a dependent table.

The num_processes setting of 2 indicates that the pipeline should be split in two, and half of the
households should be apportioned into each subprocess pipeline, and all dependent tables should
likewise be apportioned accordingly. All other tables (e.g. ``land_use``) that do share an index (name)
or have a ref_col should be considered mirrored and be included in their entirety.

The primary table is sliced by num_processes-sized strides. (e.g. for num_processes == 2, the
sub-processes get every second record starting at offsets 0 and 1 respectively. All other dependent
tables slices are based (directly or indirectly) on this primary stride segmentation of the primary
table index.

Two separate sub-process are launched (num_processes == 2) and each passed the name of their
apportioned pipeline file. They execute independently and if they terminate successfully, their
contents are then coalesced into a single pipeline file whose tables should then be essentially
the same as it had been generated by a single process.

We assume that any new tables that are created by the sub-processes are directly dependent on the
previously primary tables or are mirrored. Thus we can coalesce the sub-process pipelines by
concatenating the primary and dependent tables and simply retaining any copy of the mirrored tables
(since they should all be identical.)

The third multiprocess_step, ``mp_summarize``, then is handled in single-process mode and runs the
``write_tables`` model, writing the results, but also leaving the tables in the pipeline, with
essentially the same tables and results as if the whole simulation had been run as a single process.

Shared Data
~~~~~~~~~~~

Although multiprocessing subprocesses each have their apportioned pipeline, they also share some
data passed to them by the parent process:

  * read-only shared data such as skim matrices
  * read-write shared memory when needed.  For example when school and work modeled destinations by zone are compared to target zone sizes (as calculated by the size terms).

Outputs
~~~~~~~

When multiprocessing is run, the following additional outputs are created, which are useful for understanding how multiprocessing works:

  * run_list.txt - which contains the expanded model run list with additional annotation for single and multiprocessed steps
  * Log files for each multiprocess step and process, for example ``mp_households_0-activitysim.log`` and ``mp_households_1-activitysim.log``
  * Pipeline file for each multiprocess step and process, for example ``mp_households_0-pipeline.h5``
  * mem.csv - memory used for each step
  * breadcrumbs.yaml - multiprocess global info

See the :ref:`multiprocessing_in_detail` section for more detail.


.. index:: data tables
.. index:: tables
.. index:: data schema

Data Schema
-----------

The ActivitySim data schema depends on the sub-models implemented.  The data schema listed below is for
the primary TM1 example model.  These tables and skims are defined in the :mod:`activitysim.abm.tables` package.

.. index:: constants
.. index:: households
.. index:: input store
.. index:: land use
.. index:: persons
.. index:: size terms
.. index:: time windows table
.. index:: tours
.. index:: trips

Data Tables
~~~~~~~~~~~

The following tables are currently implemented:

  * households - household attributes for each household being simulated.  Index: ``household_id`` (see ``activitysim.abm.tables.households.py``)
  * landuse - zonal land use (such as population and employment) attributes. Index: ``zone_id`` (see ``activitysim.abm.tables.landuse.py``)
  * persons - person attributes for each person being simulated.  Index: ``person_id`` (see ``activitysim.abm.tables.persons.py``)
  * time windows - manages person time windows throughout the simulation.  See :ref:`time_windows`.  Index:  ``person_id`` (see the person_windows table create decorator in ``activitysim.abm.tables.time_windows.py``)
  * tours - tour attributes for each tour (mandatory, non-mandatory, joint, and atwork-subtour) being simulated.  Index:  ``tour_id`` (see ``activitysim.abm.models.util.tour_frequency.py``)
  * trips - trip attributes for each trip being simulated.  Index: ``trip_id`` (see ``activitysim.abm.models.stop_frequency.py``)

A few additional tables are also used, which are not really tables, but classes:

  * input store - reads input data tables from the input data store
  * constants - various constants used throughout the model system, such as person type codes
  * shadow pricing - shadow price calculator and associated utility methods, see :ref:`shadow_pricing`
  * size terms - created by reading the ``destination_choice_size_terms.csv`` input file.  Index - ``segment`` (see ``activitysim.abm.tables.size_terms.py``)
  * skims - see :ref:`skims`
  * table dictionary - stores which tables should be registered as random number generator channels for restartability of the pipeline

Data Schema
~~~~~~~~~~~

The following table lists the pipeline data tables, each final field, the data type, the step that created it, and the
number of columns and rows in the table at the time of creation.  The ``other_resources\scripts\make_pipeline_output.py`` script
uses the information stored in the pipeline file to create the table below for a small sample of households.

+----------------------------+-------------------------------+---------+------------------------------+------+------+
| Table                      | Field                         | DType   | Creator                      |NCol  |NRow  |
+============================+===============================+=========+==============================+======+======+
| accessibility              | auPkRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auPkTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auOpRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | auOpTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trPkRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trPkTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trOpRetail                    | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | trOpTotal                     | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | nmRetail                      | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| accessibility              | nmTotal                       | float32 | compute_accessibility        | 10   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | TAZ                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | SERIALNO                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | PUMA5                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hhsize                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | HHT                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | UNITTYPE                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | NOC                           | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | BLDGSZ                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | TENURE                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | VEHICL                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hinccat1                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hinccat2                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hhagecat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hsizecat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hfamily                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hunittype                     | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hNOCcat                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwrkrcat                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h0004                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h0511                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1215                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1617                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h1824                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h2534                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h3549                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h5064                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h6579                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | h80up                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_workers                   | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwork_f                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hwork_p                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | huniv                         | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hnwork                        | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hretire                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hpresch                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hschpred                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hschdriv                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | htypdwel                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hownrent                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadnwst                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadwpst                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hadkids                       | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | bucketBin                     | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | originalPUMA                  | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hmultiunit                    | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | chunk_id                      | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income_in_thousands           | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | income_segment                | int32   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | median_value_of_time          | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hh_value_of_time              | float64 | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_non_workers               | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_drivers                   | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_adults                    | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children                  | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_young_children            | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children_5_to_15          | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_children_16_to_17         | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_college_age               | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_young_adults              | int8    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | non_family                    | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | family                        | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | home_is_urban                 | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | home_is_rural                 | bool    | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | auto_ownership                | int64   | initialize_households        | 65   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | hh_work_auto_savings_ratio    | float32 | workplace_location           | 66   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_under16_not_at_school     | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active             | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_adults      | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_preschoolers| int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_travel_active_children    | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 |num_travel_active_non_presch   | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | participates_in_jtf_model     | int8    | cdap_simulate                | 73   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | joint_tour_frequency          | object  | joint_tour_frequency         | 75   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| households                 | num_hh_joint_tours            | int8    | joint_tour_frequency         | 75   | 100  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | tour_id                       | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | household_id                  | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | person_id                     | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| joint_tour_participants    | participant_num               | int64   | joint_tour_participation     | 4    | 13   |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | DISTRICT                      | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SD                            | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | county_id                     | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTHH                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHPOP                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTPOP                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | EMPRES                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SFDU                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | MFDU                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ1                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ2                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ3                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HHINCQ4                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTACRE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | RESACRE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | CIACRE                        | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | SHPOP62P                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOTEMP                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE0004                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE0519                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE2044                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE4564                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGE65P                        | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | RETEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | FPSEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HEREMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | OTHEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | AGREMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | MWTEMPN                       | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | PRKCST                        | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | OPRKCST                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | area_type                     | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | HSENROLL                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | COLLFTE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | COLLPTE                       | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TOPOLOGY                      | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | TERMINAL                      | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | ZERO                          | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | hhlds                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | sftaz                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | gqpop                         | int64   | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | household_density             | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | employment_density            | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| land_use                   | density_index                 | float64 | initialize_landuse           | 44   | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 4                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 5                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 6                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 7                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 8                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 9                             | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 10                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 11                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 12                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 13                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 14                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 15                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 16                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 17                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 18                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 19                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 20                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 21                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 22                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 23                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| person_windows             | 24                            | int8    | initialize_households        | 21   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | household_id                  | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | RELATE                        | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | ESR                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | GRADE                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | PNUM                          | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | PAUG                          | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | DDP                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | sex                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | WEEKS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | HOURS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | MSP                           | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | POVERTY                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | EARNS                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pagecat                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pemploy                       | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | pstudent                      | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | ptype                         | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | padkid                        | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age_16_to_19                  | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | age_16_p                      | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | adult                         | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | male                          | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | female                        | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_non_worker                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_retiree                   | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_preschool_kid             | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_driving_kid               | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_school_kid                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_full_time                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_part_time                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_university                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | student_is_employed           | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | nonstudent_to_school          | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_student                    | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_gradeschool                | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_highschool                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_university                 | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | school_segment                | int8    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | is_worker                     | bool    | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | home_taz                      | int64   | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | value_of_time                 | float64 | initialize_households        | 42   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | school_taz                    | int32   | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | distance_to_school            | float32 | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | roundtrip_auto_time_to_school | float32 | school_location              | 45   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | workplace_taz                 | int32   | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | distance_to_work              | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | workplace_in_cbd              | bool    | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_zone_area_type           | float64 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | roundtrip_auto_time_to_work   | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_auto_savings             | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_auto_savings_ratio       | float32 | workplace_location           | 52   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | free_parking_at_work          | bool    | free_parking                 | 53   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | cdap_activity                 | object  | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | cdap_rank                     | int64   | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | travel_active                 | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | under16_not_at_school         | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_preschool_kid_at_home     | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | has_school_kid_at_home        | bool    | cdap_simulate                | 59   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | mandatory_tour_frequency      | object  | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_and_school_and_worker    | bool    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | work_and_school_and_student   | bool    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_mand                      | int8    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_work_tours                | int8    | mandatory_tour_frequency     | 64   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_joint_tours               | int8    | joint_tour_participation     | 65   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | non_mandatory_tour_frequency  | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_non_mand                  | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_escort_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_eatout_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_shop_tours                | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_maint_tours               | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_discr_tours               | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_social_tours              | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| persons                    | num_non_escort_tours          | int8    | non_mandatory_tour_frequency | 74   | 271  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | gradeschool                   | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | highschool                    | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_destination_size    | university                    | float64 | initialize_households        | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | gradeschool                   | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | highschool                    | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| school_modeled_size        | university                    | int32   | school_location              | 3    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | person_id                     | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type                     | object  | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type_count               | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_type_num                 | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_num                      | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_count                    | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_category                 | object  | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | number_of_participants        | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | destination                   | int32   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | origin                        | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | household_id                  | int64   | mandatory_tour_frequency     | 11   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | start                         | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | end                           | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | duration                      | int8    | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tdd                           | int64   | mandatory_tour_scheduling    | 15   | 153  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | composition                   | object  | joint_tour_composition       | 16   | 159  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | tour_mode                     | object  | tour_mode_choice_simulate    | 17   | 319  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | atwork_subtour_frequency      | object  | atwork_subtour_frequency     | 19   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | parent_tour_id                | float64 | atwork_subtour_frequency     | 19   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | stop_frequency                | object  | stop_frequency               | 21   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| tours                      | primary_purpose               | object  | stop_frequency               | 21   | 344  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | person_id                     | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | household_id                  | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | tour_id                       | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | primary_purpose               | object  | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_num                      | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | outbound                      | bool    | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_count                    | int64   | stop_frequency               | 7    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | purpose                       | object  | trip_purpose                 | 8    | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | destination                   | int32   | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | origin                        | int32   | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | failed                        | bool    | trip_destination             | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | depart                        | float64 | trip_scheduling              | 11   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| trips                      | trip_mode                     | object  | trip_mode_choice             | 12   | 859  |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_high                     | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_low                      | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_med                      | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_destination_size | work_veryhigh                 | float64 | initialize_households        | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_high                     | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_low                      | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_med                      | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+
| workplace_modeled_size     | work_veryhigh                 | int32   | workplace_location           | 4    | 1454 |
+----------------------------+-------------------------------+---------+------------------------------+------+------+

.. index:: skims
.. index:: omx_file
.. index:: skim matrices

.. _skims:

Skims
~~~~~

The skims class defines Inject injectables to access the skim matrices.  The skims class reads the
skims from the omx_file on disk.  The injectables and omx_file for the example are listed below.
The skims are float64 matrix.

Skims are named <PATH TYPE>_<MEASURE>__<TIME PERIOD>:

* Highway paths:

  * SOV - SOV free
  * HOV2 - HOV2 free
  * HOV3 - HOV3 free
  * SOVTOLL - SOV toll
  * HOV2TOLL - HOV2 toll
  * HOV3TOLL - HOV3 toll
  
* Transit paths:

  * Walk access and walk egress - WLK_COM_WLK, WLK_EXP_WLK, WLK_HVY_WLK, WLK_LOC_WLK, WLK_LRF_WLK
  * Walk access and drive egress - WLK_COM_DRV, WLK_EXP_DRV, WLK_HVY_DRV, WLK_LOC_DRV, WLK_LRF_DRV
  * Drive access and walk egress - DRV_COM_WLK, DRV_EXP_WLK, DRV_HVY_WLK, DRV_LOC_WLK, DRV_LRF_WLK
  * COM = commuter rail, EXP = express bus, HVY = heavy rail, LOC = local bus, LRF = light rail ferry

* Non-motorized paths:

  * WALK
  * BIKE

* Measures:

  * TIME - Time (minutes)
  * DIST - Distance (miles)
  * BTOLL - Bridge toll (cents)
  * VTOLL - Value toll (cents)

  * IVT - In-vehicle time, time (minutes x 100)
  * IWAIT - Initial wait time, time (minutes x 100)
  * XWAIT - Transfer wait time, time (minutes x 100)
  * WACC - Walk access time, time (minutes x 100)
  * WAUX - Auxiliary walk time, time (minutes x 100)
  * WEGR - Walk egress time, time (minutes x 100)
  * DTIM - Drive access and/or egress time, time (minutes x 100)
  * DDIST - Drive access and/or egress distance, distance (miles x 100)
  * FAR - Fare, cents
  * BOARDS - Boardings, number
  * TOTIVT - Total in-vehicle time, time (minutes x 100)
  * KEYIVT - Transit submode in-vehicle time, time (minutes x 100)
  * FERRYIVT - Ferry in-vehicle time, time (minutes x 100)

* Time periods:

  * EA
  * AM
  * MD
  * PM
  * EV

+------------------------------+-----------------+
|                        Field |            Type |
+==============================+=================+
|                 SOV_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__AM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__AM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__EA |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__EA |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__EV |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__EV |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__MD |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__MD |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                 SOV_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|                SOV_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV2_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|               HOV2_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|                HOV3_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|               HOV3_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|             SOVTOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            SOVTOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV2TOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV2TOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_TIME__PM |  float64 matrix |
+------------------------------+-----------------+
|            HOV3TOLL_DIST__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_BTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|           HOV3TOLL_VTOLL__PM |  float64 matrix |
+------------------------------+-----------------+
|                    \DIST\    |  float64 matrix |
+------------------------------+-----------------+
|                \DISTWALK\    |  float64 matrix |
+------------------------------+-----------------+
|                \DISTBIKE\    |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__AM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__EA |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__EA |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__EA |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__EA |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__EV |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__EV |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__EV |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__EV |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__MD |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_COM_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_COM_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_COM_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_COM_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_EXP_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_EXP_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_EXP_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_EXP_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_HVY_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_HVY_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_HVY_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_HVY_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LOC_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LOC_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LOC_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LOC_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     DRV_LRF_WLK_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          DRV_LRF_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         DRV_LRF_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        DRV_LRF_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       DRV_LRF_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_COM_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_COM_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_COM_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_COM_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_EXP_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_EXP_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_EXP_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_EXP_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_HVY_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_HVY_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_HVY_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_HVY_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LOC_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LOC_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LOC_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LOC_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_DRV_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_DRV_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_DTIM__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_DDIST__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_DRV_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_DRV_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_DRV_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_TOTIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_KEYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|     WLK_LRF_WLK_FERRYIVT__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_LRF_WLK_FAR__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_LRF_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_LRF_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|       WLK_LRF_WLK_BOARDS__PM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__AM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__AM |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__MD |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__MD |  float64 matrix |
+------------------------------+-----------------+
|          WLK_TRN_WLK_IVT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_IWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|        WLK_TRN_WLK_XWAIT__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WACC__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WAUX__PM |  float64 matrix |
+------------------------------+-----------------+
|         WLK_TRN_WLK_WEGR__PM |  float64 matrix |
+------------------------------+-----------------+
