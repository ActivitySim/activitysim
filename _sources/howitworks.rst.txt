
How the System Works
====================

This page describes describes how the ActivitySim software works.

.. _how_the_system_works:

Execution Flow
--------------

The example model run starts by running ``simulation.py``, which calls:

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
(functions) for the ``settings`` object based on the setting.yaml file and the ``store`` based on the HDF5 input 
file.  The Python decorator ``@orca.injectable`` overrides the function definition ``store`` to execute this 
function whenever ``store`` is called by orca.

:: 

  @orca.injectable(cache=True)
  def store(data_dir, settings):
    return pd.HDFStore(os.path.join(data_dir, settings["store"]),mode='r')

Next, the tables module executes the following import statements to define the dynamic orca tables households, 
persons, skims, etc., but does not load them. It also defines the core dynamic orca table columns (calculated fields) 
and injectables (functions) defined in the classes.  The Python decorator ``@orca.table`` and 
``@orca.column("households")`` override the function definitions so the function name
becomes the table name in the first case, whereas the function name becomes the column name in the second case.  The 
argument to ``households`` in ``@orca.column("households")`` is the table (either real or virtual) that the 
column is added to.  The columns defined in these classes are thought to be generic across AB model implementations.
Additional implementation specific columns can be defined in an extensions folder, as discussed later.  

::

  import households
  import persons
  #etc...
  
  @orca.table(cache=True)
    def households(store, households_sample_size, trace_hh_id):
    
  @orca.column("households")
  def income_in_thousands(households):
    return households.income / 1000
  
The models module then loads all the sub-models, which are registered as orca model steps with 
the ``@orca.step()`` decorator.  These steps will eventually be run by the pipeline manager.

::

  import accessibility
  import auto_ownership
  #etc...
  
  @orca.step()
  def compute_accessibility(settings, accessibility_spec,
                          accessibility_settings,
                          skim_dict, omx_file, land_use, trace_od):

Back in the main ``simulation.py`` script, the next steps are to load the pipeline manager and import the example
extensions.  The example extensions are additional orca computed columns that are specific to the example.  This
includes columns such as person age bins, which are not included in the core person table since they often vary
by implementation.

::

  from activitysim.core import pipeline
  import extensions

The next step in the example is to define and run the pipeline.  The ``resume_after`` argument is set to None
in order to start the pipeline from the beginning.

::
  
  _MODELS = [
    'compute_accessibility',
    'school_location_simulate',
  #etc...
  
  pipeline.run(models=_MODELS, resume_after=None)

The :func:`activitysim.core.pipeline.run` method loops through the list of models, calls ``orca.run(model_step)``, 
and manages the data pipeline.  The first microsimulation model run is school location, which is called via:

::

  orca.run(["school_location_simulate"])

  @orca.step()
  def school_location_simulate(persons_merged,
                             school_location_spec,
                             school_location_settings,
                             skim_dict,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):
                             
The ``school_location_simulate`` step requires the objects defined in the function definition 
above.  Since they are not yet loaded, orca goes looking for them.  This is called lazy 
loading (or on-demand loading).  The steps to get the persons data loaded is illustrated below.

::

  #persons_merged is in the step function signature

  @orca.table()
  def persons_merged(persons, households, land_use, accessibility):
    return orca.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
        
  #it required persons, households, land_use, accessibility
  @orca.table(cache=True)
  def persons(persons_internal):
      return persons_internal.to_frame()
      
  #persons requires store, households_sample_size, households, trace_hh_id
  @orca.table()
  def persons(store, households_sample_size, households, trace_hh_id):

    df = store["persons"]

    if households_sample_size > 0:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    logger.info("loaded persons %s" % (df.shape,))

    # replace table function with dataframe
    orca.add_table('persons', df)

    pipeline.get_rn_generator().add_channel(df, 'persons')

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "persons", warn_if_empty=True)

    return df
  
  #households requires store, households_sample_size, trace_hh_id
  @orca.table()
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
    orca.add_table('households', df)

    pipeline.get_rn_generator().add_channel(df, 'households')

    if trace_hh_id:
        tracing.register_traceable_table('households', df)
        tracing.trace_df(df, "households", warn_if_empty=True)

    return df
  
  #etc.... until all the required dependencies are resolved
  
The various calls are also setting up the logging, the tracing, and the random number generators.  

``school_location_simulate`` also sets the persons merged table as choosers, reads the expressions 
specification file, settings yaml file, and destination_size_terms file, and also sets the chunk 
size and trace id if specified.  The skims dictionary is also passed in, as explained next.

::

  def school_location_simulate(persons_merged,
                             school_location_spec,
                             school_location_settings,
                             skim_dict,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):
    
Inside the method, the skim lookups required for this model are configured. The following code 
set the keys for looking up the skim values for this model. In this case there is a ``TAZ`` column 
in the choosers, which was in the ``households`` table that was joined with ``persons`` to make 
``persons_merged`` and a ``TAZ`` in the alternatives generation code which get merged during 
interaction as renamed ``TAZ_r``.  The skims are lazy loaded under the name "skims" and are 
available in the expressions using ``@skims``.

::

    skims.set_keys("TAZ", "TAZ_r")
    locals_d = {"skims": skims}

The next step is to call the :func:`activitysim.core.simulate.interaction_simulate` function which run a MNL choice model simulation 
in which alternatives must be merged with choosers because there are interaction terms or because 
alternatives are sampled.  The choosers table, the alternatives table, the model specification 
expressions file, the skims, the skims lookups, the sample size, the chunk size, and trace labels are passed in.  

:: 

  choices = asim.interaction_simulate(
      choosers_segment,
      alternatives_segment,
      spec=school_location_spec[[school_type]],
      skims=skims,
      locals_d=locals_d,
      sample_size=sample_size,
      chunk_size=chunk_size,
      trace_label='school_location.%s' % school_type,
      trace_choice_name='school_location')
    
This function solves the utilities, calculates probabilities, draws random numbers, selects choices, and returns a column of choices. 
This is done in a for loop of chunks of chooser records in order to avoid running out of RAM when building the often large data tables.
This method does a lot, and eventually calls :func:`activitysim.core.simulate.eval_interaction_utilities`, which loops through each expression 
in the expression file and solves it at once for all records in the chunked chooser table using either pandas' eval() or Python's eval().

The :func:`activitysim.core.simulate.interaction_simulate` method is currently only a multinomial logit choice model.  The 
:func:`activitysim.core.simulate.simple_simulate` method supports both MNL and NL as specified by 
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
        interaction_utilities.as_matrix().reshape(len(choosers), sample_size),
        index=choosers.index)

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(utilities, trace_label=trace_label, trace_choosers=choosers)

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = logit.make_choices(probs, trace_label=trace_label, trace_choosers=choosers)

    # offsets is the offset into model_design df of first row of chooser alternatives
    offsets = np.arange(len(positions)) * sample_size
    # resulting pandas Int64Index has one element per chooser row and is in same order as choosers
    choices = interaction_utilities.index.take(positions + offsets)

    # create a series with index from choosers and the index of the chosen alternative
    choices = pd.Series(choices, index=choosers.index)


Finally, the model adds the choices as a column to the applicable table - ``persons`` - and adds 
additional dependent columns.  The dependent columns are those orca columns with the virtual table 
name ``persons_school``.

:: 

   orca.add_column("persons", "school_taz", choices)
   
   pipeline.add_dependent_columns("persons", "persons_school")

   # columns to update after the school location choice model
   @orca.table()
   def persons_school(persons):
    return pd.DataFrame(index=persons.index)
    
   @orca.column("persons_school")
   def distance_to_school(persons, skim_dict):
       distance_skim = skim_dict.get('DIST')
       return pd.Series(distance_skim.get(persons.home_taz,
                                          persons.school_taz),
                        index=persons.index)
   
   @orca.column("persons_school")
   def roundtrip_auto_time_to_school(persons, skim_dict):
       sovmd_skim = skim_dict.get(('SOV_TIME', 'MD'))
       return pd.Series(sovmd_skim.get(persons.home_taz,
                                       persons.school_taz) +
                        sovmd_skim.get(persons.school_taz,
                                       persons.home_taz),
                        index=persons.index)

Any orca columns that are required are calculated-on-the-fly, such as ``roundtrip_auto_time_to_school``
which in turn uses skims from the skim_dict orca injectable.

The rest of the microsimulation models operate in a similar fashion with a few notable additions:

* creating new tables
* vectorized 3D skims indexing
* the accessibilities model step

Creating New Tables
-------------------

The mandatory tour frequency model sets the ``persons.mandatory_tour_frequency`` column.  Once the number of tours
is known, then the next step is to create tours records for subsequent models.  This is done with the following code,
which requires the ``persons`` table and returns a new pandas DataFrame which is registered as an 
orca table named ``mandatory_tours``.

::

  @orca.table(cache=True)
  def mandatory_tours(persons):
    persons = persons.to_frame(columns=["mandatory_tour_frequency","is_worker"])
    persons = persons[~persons.mandatory_tour_frequency.isnull()]
    return process_mandatory_tours(persons)
  
  #processes the mandatory_tour_frequency column that comes out of the model 
  #and turns into a DataFrame that represents the mandatory tours that were generated
  def process_mandatory_tours(persons):
    #...
    return pd.DataFrame(tours, columns=["person_id", "tour_type", "tour_num"])

Vectorized 3D Skim Indexing
---------------------------

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

Accessibilities Model Step
--------------------------

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
