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

