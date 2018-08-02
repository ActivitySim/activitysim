
How the System Works
====================

This page describes describes how the ActivitySim software works and the example data schema.

.. _how_the_system_works:

Execution Flow
--------------

The example model run starts by running ``simulation.py``.

Initialization
~~~~~~~~~~~~~~

The first steps of ``simulation.py`` are:

::

  import orca
  from activitysim import abm 
  
which starts orca, which will now take over running the system and defines the orca/pandas tables and their data 
sources but does not load the data.  The second statement loads :mod:`activitysim.abm.__init__`, which calls:

::

   import misc 
   import tables
   import models

which then loads the misc, tables, and models class definitions.  Loading :mod:`activitysim.abm.misc` defines orca injectables 
(functions) for the ``settings`` object based on the setting yaml file, the ``store`` based on the HDF5 input 
file, and the trace settings.  The Python decorator ``@inject.injectable`` overrides the function definition ``store`` 
to execute this function whenever ``store`` is called by orca.  The ``misc`` class depends on 
:mod:`activitysim.core.inject` and :mod:`activitysim.core.pipeline`, which wrap orca and manage the data pipeline.  

:: 

  @inject.injectable(cache=True)
  def store(data_dir, settings):
    #...
    file = pd.HDFStore(fname, mode='r')
    pipeline.close_on_exit(file, fname)
    return file

Next, the tables module executes the following import statements in :mod:`activitysim.abm.tables.__init__` to 
define the dynamic orca tables (households, 
persons, skims, etc.), but does not load them. It also defines the core dynamic orca injectables (functions) 
defined in the classes. The Python decorator ``@inject.table`` override the function definitions so the function name
becomes the table name.  Additional implementation specific table fields are defined in annotation preprocessors for
each step, as discussed later.  

::

  import households
  import persons
  #etc...
  
  #then in households.py
  @inject.table()
  def households(store, households_sample_size, trace_hh_id):
  
The models module then loads all the sub-models, which are registered as orca model steps with 
the ``@inject.step()`` decorator.  These steps will eventually be run by the pipeline manager.

::

  import initialize
  import accessibility
  import auto_ownership
  #etc...
  
  #then in accessibility.py
  @inject.step()
  def compute_accessibility(settings, accessibility_spec,
                          accessibility_settings,
                          skim_dict, omx_file, land_use, trace_od):

Back in the main ``simulation.py`` script, the next steps are to load the pipeline manager.

::

  from activitysim.core import pipeline


The next step in the example is to read and run the pipeline.  The ``resume_after`` argument is set to None
in order to start the pipeline from the beginning.

::
  
  MODELS = setting('models')
  
  pipeline.run(models=MODELS, resume_after=None)

The :func:`activitysim.core.pipeline.run` method loops through the list of models, calls ``inject.run(model_step)``, 
and manages the data pipeline.  The first microsimulation model run is school location.  The school location 
model is broken into three steps:

  * school_location_sample - selects a sample of alternative school locations for the next model step. This selects X locations from the full set of model zones using a simple utility.
  * school_location_logsums - starts with the table created above and calculates and adds the mode choice logsum expression for each alternative school location.
  * school_location_simulate - starts with the table created above and chooses a final school location, this time with the mode choice logsum included.

School Location Sample
~~~~~~~~~~~~~~~~~~~~~~

The school location sample model is run via:

::
  
  #run model step
  inject.run(["school_location_sample"])
          
  #define model step
  @inject.step()
  def school_location_sample(persons_merged,
                             school_location_sample_spec,
                             school_location_settings,
                             skim_dict,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):
                             
The ``school_location_sample`` step requires the objects defined in the function definition 
above.  Since they are not yet loaded, orca goes looking for them.  This is called lazy 
loading (or on-demand loading).  The steps to get the persons data loaded is illustrated below.
The various calls also setup logging, tracing, and stable random number management. 

::

  #persons_merged is in the step function signature
  
  #persons_merged is defined in persons.py and needs persons
  @inject.table()
  def persons_merged(persons, households, land_use, accessibility):
    return inject.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
        
  #persons in persons.py requires store, households_sample_size, households, trace_hh_id
  @inject.table()
  def persons(store, households_sample_size, households, trace_hh_id):

    df = store["persons"]

    if households_sample_size > 0:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    logger.info("loaded persons %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table('persons', df)

    pipeline.get_rn_generator().add_channel(df, 'persons')

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "persons", warn_if_empty=True)

    return df
  
  #households requires store, households_sample_size, trace_hh_id
  @inject.table()
  def households(store, households_sample_size, trace_hh_id):

    df_full = store["households"]

    # if we are tracing hh exclusively
    if trace_hh_id and households_sample_size == 1:

        # df contains only trace_hh (or empty if not in full store)
        df = tracing.slice_ids(df_full, trace_hh_id)

    # if we need sample a subset of full store
    elif households_sample_size > 0 and len(df_full.index) > households_sample_size:

        # take the requested random sample
        df = asim.random_rows(df_full, households_sample_size)

        # if tracing and we missed trace_hh in sample, but it is in full store
        if trace_hh_id and trace_hh_id not in df.index and trace_hh_id in df_full.index:
                # replace first hh in sample with trace_hh
                logger.debug("replacing household %s with %s in household sample" %
                             (df.index[0], trace_hh_id))
                df_hh = tracing.slice_ids(df_full, trace_hh_id)
                df = pd.concat([df_hh, df[1:]])

    else:
        df = df_full

    logger.info("loaded households %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table('households', df)

    pipeline.get_rn_generator().add_channel(df, 'households')

    if trace_hh_id:
        tracing.register_traceable_table('households', df)
        tracing.trace_df(df, "households", warn_if_empty=True)

    return df
  
  #etc.... until all the required dependencies are resolved 

``school_location_sample`` also sets the persons merged table as choosers, reads the expressions 
specification file, settings yaml file, and destination_size_terms file, and also sets the chunk 
size and trace id if specified.  The skims dictionary is also passed in, as explained next.

::

  def school_location_sample(persons_merged,
                             school_location_sample_spec,
                             school_location_settings,
                             skim_dict,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):
    
Inside the method, the skim matrix lookups required for this model are configured. The following code 
set the keys for looking up the skim values for this model. In this case there is a ``TAZ`` column 
in the choosers, which was in the ``households`` table that was joined with ``persons`` to make 
``persons_merged`` and a ``TAZ`` in the alternatives generation code which get merged during 
interaction as renamed ``TAZ_r``.  The skims are lazy loaded under the name "skims" and are 
available in the expressions using ``@skims``.

::

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")
    locals_d = {
        'skims': skims
    }

The next step is to call the :func:`activitysim.core.interaction_sample.interaction_sample` function which 
selects a sample of alternatives by running a MNL choice model simulation in which alternatives must be 
merged with choosers because there are interaction terms.  The choosers table, the alternatives table, the 
sample size, the model specification expressions file, the skims, the skims lookups, the chunk size, and the 
trace labels are passed in.  

:: 

  choices = interaction_sample(
                choosers_segment,
                alternatives_segment,
                sample_size=sample_size,
                alt_col_name=alt_col_name,
                spec=school_location_sample_spec[[school_type]],
                skims=skims,
                locals_d=locals_d,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(trace_label, school_type))
    
This function solves the utilities, calculates probabilities, draws random numbers, selects choices with 
replacement, and returns the choices. This is done in a for loop of chunks of chooser records in order to avoid 
running out of RAM when building the often large data tables. This method does a lot, and eventually 
calls :func:`activitysim.core.interaction_simulate.eval_interaction_utilities`, which loops through each 
expression in  the expression file and solves it at once for all records in the chunked chooser 
table using either pandas' eval() or Python's eval().

The :func:`activitysim.core.interaction_sample.interaction_sample` method is currently only a multinomial 
logit choice model.  The :func:`activitysim.core.simulate.simple_simulate` method supports both MNL and NL as specified by 
the ``LOGIT_TYPE`` setting in the model settings YAML file.   The ``auto_ownership.yaml`` file for example specifies 
the ``LOGIT_TYPE`` as ``MNL.``

If the expression is a skim matrix, then the entire column of chooser OD pairs is retrieved from the matrix (i.e. numpy array) 
in one vectorized step.  The ``orig`` and ``dest`` objects in ``self.data[orig, dest]`` in :mod:`activitysim.core.skim` are vectors
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
    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)

    choices_df = make_sample_choices(
        choosers, probs, interaction_utilities,
        sample_size, alternative_count, alt_col_name, trace_label)

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

The model creates the ``school_location_sample`` table using the choices above.  This table is 
then used for the next model step - solving the logsums for the sample.

:: 

    inject.add_table('school_location_sample', choices)
    

School Location Logsums
~~~~~~~~~~~~~~~~~~~~~~~

The school location logsums model is called via:

::

  #run model step
  inject.run(["school_location_logsums"])
          
  #define model step
  @inject.step()
  def school_location_logsums(
        persons_merged,
        land_use,
        skim_dict, skim_stack,
        school_location_sample,
        configs_dir,
        chunk_size,
        trace_hh_id):
                             
The ``school_location_logsums`` step requires the objects defined in the function definition 
above.  Some of these are not yet loaded, so orca goes looking for them.  The next steps are
similar to what the sampling model does, except this time the sampled locations table is the choosers
and the model is calculating and adding the mode choice logsums using the logsums expression files:

::

    for school_type, school_type_id in SCHOOL_TYPE_ID.iteritems():

        segment = 'university' if school_type == 'university' else 'school'
        logsum_spec = get_segment_and_unstack(omnibus_logsum_spec, segment)
        
        choosers = location_sample[location_sample['school_type'] == school_type_id]

        choosers = pd.merge(
            choosers,
            persons_merged,
            left_index=True,
            right_index=True,
            how="left")

        logsums = logsum.compute_logsums(
            choosers, logsum_spec,
            logsum_settings, school_location_settings,
            skim_dict, skim_stack,
            chunk_size, trace_hh_id,
            tracing.extend_trace_label(trace_label, school_type))

    inject.add_column("school_location_sample", "mode_choice_logsum", logsums)

The :func:`activitysim.abm.models.util.logsums.compute_logsums` method goes through a similar series
of steps as the interaction_sample function but ends up calling 
:func:`activitysim.core.simulate.simple_simulate_logsums` since it supports nested logit models, which 
are required for the mode choice logsum calculation.  The 
:func:`activitysim.core.simulate.simple_simulate_logsums` returns a vector of logsums (instead of a vector 
choices). The resulting logsums are added to the ``school_location_sample`` table as the 
``mode_choice_logsum`` column.

School Location Final Choice 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final school location choice model operates on the ``school_location_sample`` table created 
above and is called as follows:

:: 

  #run model step
  inject.run(["school_location_simulate"])
  
  #define model step
  @inject.step()
  def school_location_simulate(persons_merged, persons,
                             school_location_sample,
                             school_location_spec,
                             school_location_settings,
                             skim_dict,
                             land use, size_terms,
                             chunk_size,
                             trace_hh_id):

The ``school_location_simulate`` step requires the objects defined in the function definition 
above.  The operations executed by this model are very similar to the earlier models, except 
this time the sampled locations table is the choosers and the model selects one alternative for
each chooser using the school location simulate expression files and the 
:func:`activitysim.core.interaction_sample_simulate.interaction_sample_simulate` function.  

The model adds the choices as a column to the ``persons`` table and adds 
additional output columns using a postprocessor table annotation.  Refer to :ref:`table_annotation` 
for more information and the :func:`activitysim.abm.models.util.expressions.assign_columns` function.

:: 

   # We only chose school locations for the subset of persons who go to school
   # so we backfill the empty choices with -1 to code as no school location
   persons['school_taz'] = choices.reindex(persons.index).fillna(-1).astype(int)
   
   expressions.assign_columns(
        df=persons,
        model_settings=school_location_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

Finishing Up 
~~~~~~~~~~~~

The last models to be run by the data pipeline are:

* ``write_data_dictionary``, which writes the table_name, number of rows, number of columns, and number of bytes for each checkpointed table
* ``write_tables``, which writes pipeline tables as csv files as specified by the output_tables setting

Back in the main ``simulation.py`` script, the final steps are to:

* close the data pipeline (and attached HDF5 file)
* print the elapsed model runtime

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
the ``tours`` table managed in the data pipeline.  This is the same basic pattern used for creating all new tables - 
tours, trips, etc.

::

  @inject.step()
  def mandatory_tour_frequency(persons, persons_merged,
                             mandatory_tour_frequency_spec,
                             mandatory_tour_frequency_settings,
                             mandatory_tour_frequency_alternatives,
                             chunk_size,
                             trace_hh_id):
  
  mandatory_tours = process_mandatory_tours(
      persons=persons[~persons.mandatory_tour_frequency.isnull()],
      mandatory_tour_frequency_alts=mandatory_tour_frequency_alternatives
  )

  tours = pipeline.extend_table("tours", mandatory_tours)

    
Vectorized 3D Skim Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mode choice model uses the :class:`activitysim.core.skim.SkimStackWrapper` class in addition to the skims (2D) 
class.  The SkimStackWrapper class represents a collection of skims with a third dimension, which in this case 
is time period.  Setting up the 3D index for SkimStackWrapper is done as follows:

::

  # setup three skim keys based on columns in the chooser table
  # origin, destination, time period; destination, origin, time period; origin, destination
  odt_skim_stack_wrapper = skim_stack.wrap(left_key='TAZ', right_key='destination', skim_key="out_period")
  dot_skim_stack_wrapper = skim_stack.wrap(left_key='destination', right_key='TAZ', skim_key="in_period")
  od_skims               = skim_dict.wrap('TAZ', 'destination')
  
  #pass these into simple_simulate so they can be used in expressions
  locals_d = {
    "odt_skims": odt_skim_stack_wrapper,
    "dot_skims": dot_skim_stack_wrapper,
    "od_skims": od_skim_stack_wrapper
  }

When model expressions such as ``@odt_skims['WLK_LOC_WLK_TOTIVT']`` are solved,
the ``WLK_LOC_WLK_TOTIVT`` skim matrix values for all chooser table origins, destinations, and 
out_periods can be retrieved in one vectorized request.

All the skims are preloaded (cached) by the pipeline manager at the beginning of the model 
run in order to avoid repeatedly reading the skims from the OMX files on disk.  This saves
significant model runtime.

See :ref:`skims_in_detail` for more information on skim handling.

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


.. index:: data tables
.. index:: tables
.. index:: data schema

Data Schema
-----------

The ActivitySim data schema depends on the sub-models implemented.  The data schema listed below is for
the example model.  These tables and skims are defined in the :mod:`activitysim.abm.tables` package.

.. index:: constants
.. index:: households
.. index:: land use
.. index:: persons
.. index:: random channels
.. index:: size terms
.. index:: time windows table
.. index:: tours 
.. index:: trips

Data Tables
~~~~~~~~~~~

The following tables are currently implemented:

  * households - household attributes for each household being simulated.  Index: ``HHID`` (see ``scripts\data_mover.ipynb``)
  * landuse - zonal land use (such as population and employment) attributes. Index: ``TAZ`` (see ``scripts\data_mover.ipynb``)
  * persons - person attributes for each person being simulated.  Index: ``PERID`` (see ``scripts\data_mover.ipynb``)
  * time windows - manages person time windows throughout the simulation.  See :ref:`time_windows`.  Index:  ``PERID`` (see the person_windows table create decorator in ``activitysim.abm.tables.time_windows.py``)
  * tours - tour attributes for each tour (mandatory, non-mandatory, joint, and atwork-subtour) being simulated.  Index:  ``TOURID`` (see ``activitysim.abm.models.util.tour_frequency.py``)
  * trips - trip attributes for each trip being simulated.  Index: ``TRIPID`` (see ``activitysim.abm.models.stop_frequency.py``)

A few additional tables are also used, which are not really tables, but classes:

  * constants - various codes used throughout the model system, such as person type codes
  * random channels - random channel management settings 
  * size terms - created by reading the ``destination_choice_size_terms.csv`` input file.  Index - ``segment`` (see ``activitysim.abm.tables.size_terms.py``)
  * skims - see :ref:`skims` 
  
Data Schema
~~~~~~~~~~~

The following table lists the pipeline data tables, each final field, the data type, the step that created it, and the  
number of columns and rows in the table at the time of creation.  The ``scripts\make_pipeline_output.py`` script 
uses the information stored in the pipeline file to create the table below for a small sample of households.  

+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| Table                             | Field                         | DType   | Creator                            |NCol|NRow |
+===================================+===============================+=========+====================================+====+=====+
| households                        | TAZ                           | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | SERIALNO                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | PUMA5                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | income                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hhsize                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | HHT                           | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | UNITTYPE                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | NOC                           | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | BLDGSZ                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | TENURE                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | VEHICL                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hinccat1                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hinccat2                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hhagecat                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hsizecat                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hfamily                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hunittype                     | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hNOCcat                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hwrkrcat                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h0004                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h0511                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h1215                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h1617                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h1824                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h2534                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h3549                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h5064                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h6579                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | h80up                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_workers                   | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hwork_f                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hwork_p                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | huniv                         | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hnwork                        | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hretire                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hpresch                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hschpred                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hschdriv                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | htypdwel                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hownrent                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hadnwst                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hadwpst                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hadkids                       | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | bucketBin                     | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | originalPUMA                  | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | hmultiunit                    | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | chunk_id                      | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | income_in_thousands           | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | income_segment                | int32   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_non_workers               | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_drivers                   | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_adults                    | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_children                  | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_young_children            | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_children_5_to_15          | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_children_16_to_17         | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_college_age               | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_young_adults              | float64 | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | non_family                    | bool    | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | family                        | bool    | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | home_is_urban                 | bool    | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | home_is_rural                 | bool    | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | work_tour_auto_time_savings   | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | auto_ownership                | int64   | initialize                         | 64 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_under16_not_at_school     | int32   | cdap_simulate                      | 68 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_travel_active             | int32   | cdap_simulate                      | 68 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_travel_active_adults      | int32   | cdap_simulate                      | 68 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_travel_active_children    | int32   | cdap_simulate                      | 68 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | joint_tour_frequency          | object  | joint_tour_frequency               | 70 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| households                        | num_hh_joint_tours            | int8    | joint_tour_frequency               | 70 | 100 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | DISTRICT                      | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | SD                            | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | county_id                     | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TOTHH                         | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HHPOP                         | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TOTPOP                        | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | EMPRES                        | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | SFDU                          | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | MFDU                          | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HHINCQ1                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HHINCQ2                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HHINCQ3                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HHINCQ4                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TOTACRE                       | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | RESACRE                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | CIACRE                        | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | SHPOP62P                      | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TOTEMP                        | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGE0004                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGE0519                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGE2044                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGE4564                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGE65P                        | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | RETEMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | FPSEMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HEREMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | OTHEMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | AGREMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | MWTEMPN                       | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | PRKCST                        | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | OPRKCST                       | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | area_type                     | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | HSENROLL                      | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | COLLFTE                       | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | COLLPTE                       | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TOPOLOGY                      | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | TERMINAL                      | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | ZERO                          | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | hhlds                         | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | sftaz                         | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | gqpop                         | int64   | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | household_density             | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | employment_density            | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | density_index                 | float64 | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| land_use                          | county_name                   | object  | initialize                         | 45 | 25  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 4                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 5                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 6                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 7                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 8                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 9                             | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 10                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 11                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 12                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 13                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 14                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 15                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 16                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 17                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 18                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 19                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 20                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 21                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 22                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 23                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| person_windows                    | 24                            | int8    | initialize                         | 21 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | household_id                  | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | age                           | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | RELATE                        | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | ESR                           | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | GRADE                         | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | PNUM                          | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | PAUG                          | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | DDP                           | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | sex                           | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | WEEKS                         | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | HOURS                         | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | MSP                           | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | POVERTY                       | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | EARNS                         | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | pagecat                       | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | pemploy                       | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | pstudent                      | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | ptype                         | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | padkid                        | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | age_16_to_19                  | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | age_16_p                      | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | adult                         | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | male                          | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | female                        | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_non_worker                | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_retiree                   | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_preschool_kid             | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_driving_kid               | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_school_kid                | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_full_time                 | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_part_time                 | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_university                | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | student_is_employed           | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | nonstudent_to_school          | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | is_worker                     | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | is_student                    | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | is_gradeschool                | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | is_highschool                 | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | is_university                 | bool    | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | home_taz                      | int64   | initialize                         | 40 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | school_taz                    | int32   | school_location_simulate           | 43 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | distance_to_school            | float64 | school_location_simulate           | 43 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | roundtrip_auto_time_to_school | float64 | school_location_simulate           | 43 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | workplace_taz                 | int32   | workplace_location_simulate        | 48 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | distance_to_work              | float64 | workplace_location_simulate        | 48 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | roundtrip_auto_time_to_work   | float64 | workplace_location_simulate        | 48 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | workplace_in_cbd              | bool    | workplace_location_simulate        | 48 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | work_taz_area_type            | float64 | workplace_location_simulate        | 48 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | cdap_activity                 | object  | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | cdap_rank                     | int64   | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | travel_active                 | bool    | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | under16_not_at_school         | bool    | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_preschool_kid_at_home     | bool    | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | has_school_kid_at_home        | bool    | cdap_simulate                      | 54 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | mandatory_tour_frequency      | object  | mandatory_tour_frequency           | 59 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | work_and_school_and_worker    | bool    | mandatory_tour_frequency           | 59 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | work_and_school_and_student   | bool    | mandatory_tour_frequency           | 59 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_mand                      | int8    | mandatory_tour_frequency           | 59 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_work_tours                | int8    | mandatory_tour_frequency           | 59 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | non_mandatory_tour_frequency  | float64 | non_mandatory_tour_frequency       | 64 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_non_mand                  | float64 | non_mandatory_tour_frequency       | 64 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_escort_tours              | float64 | non_mandatory_tour_frequency       | 64 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_non_escort_tours          | float64 | non_mandatory_tour_frequency       | 64 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| persons                           | num_eatout_tours              | float64 | non_mandatory_tour_frequency       | 64 | 157 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | person_id                     | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_type                     | object  | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_type_count               | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_type_num                 | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_num                      | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_count                    | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_category                 | object  | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | number_of_participants        | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | destination                   | int32   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | origin                        | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | household_id                  | int64   | mandatory_tour_frequency           | 11 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | start                         | int64   | mandatory_tour_scheduling          | 15 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | end                           | int64   | mandatory_tour_scheduling          | 15 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | duration                      | int64   | mandatory_tour_scheduling          | 15 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tdd                           | int64   | mandatory_tour_scheduling          | 15 | 71  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | composition                   | object  | joint_tour_composition             | 16 | 73  |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | tour_mode                     | object  | joint_tour_mode_choice             | 17 | 183 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | atwork_subtour_frequency      | object  | atwork_subtour_frequency           | 19 | 186 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | parent_tour_id                | float64 | atwork_subtour_frequency           | 19 | 186 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | stop_frequency                | object  | stop_frequency                     | 21 | 186 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| tours                             | primary_purpose               | object  | stop_frequency                     | 21 | 186 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | person_id                     | int64   | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | household_id                  | int64   | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | tour_id                       | int64   | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | primary_purpose               | object  | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | trip_num                      | int64   | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | outbound                      | bool    | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | trip_count                    | int64   | stop_frequency                     | 7  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | purpose                       | object  | trip_purpose                       | 8  | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | destination                   | int32   | trip_destination                   | 11 | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | origin                        | int32   | trip_destination                   | 11 | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | failed                        | bool    | trip_destination                   | 11 | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | depart                        | int64   | trip_scheduling                    | 12 | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+
| trips                             | trip_mode                     | int64   | trip_mode_choice                   | 13 | 428 |
+-----------------------------------+-------------------------------+---------+------------------------------------+----+-----+

.. index:: skims
.. index:: omx_file
.. index:: skim matrices

.. _skims:

Skims
~~~~~

The skims class defines orca injectables to access the skim matrices.  The skims class reads the
skims from the omx_file on disk.  The injectables and omx_file for the example are listed below.
The skims are float64 matrix.

+-------------+-----------------+------------------------------------------------------------------------+
|       Table |            Type |                                            Creation                    |
+=============+=================+========================================================================+
|   skim_dict |        SkimDict | skims.py defines skim_dict which reads omx_file                        |
+-------------+-----------------+------------------------------------------------------------------------+
|  skim_stack |       SkimStack | skims.py defines skim_stack which calls skim_dict which reads omx_file |
+-------------+-----------------+------------------------------------------------------------------------+

Skims are named <PATHTYPE>_<MEASURE>__<TIME PERIOD>:

* Highway paths are SOV, HOV2, HOV3, SOVTOLL, HOV2TOLL, HOV3TOLL
* Transit paths are:

  * Walk access and walk egress - WLK_COM_WLK, WLK_EXP_WLK, WLK_HVY_WLK, WLK_LOC_WLK, WLK_LRF_WLK
  * Walk access and drive egress - WLK_COM_DRV, WLK_EXP_DRV, WLK_HVY_DRV, WLK_LOC_DRV, WLK_LRF_DRV
  * Drive access and walk egress - DRV_COM_WLK, DRV_EXP_WLK, DRV_HVY_WLK, DRV_LOC_WLK, DRV_LRF_WLK
  * COM = commuter rail, EXP = express bus, HVY = heavy rail, LOC = local bus, LRF = light rail ferry
  
* Non-motorized paths are WALK, BIKE
* Time periods are EA, AM, MD, PM, EV

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
|                    \DIST__\  |  float64 matrix |
+------------------------------+-----------------+
|                \DISTWALK__\  |  float64 matrix |
+------------------------------+-----------------+
|                \DISTBIKE__\  |  float64 matrix |
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
