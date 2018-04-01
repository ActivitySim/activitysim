# ActivitySim
# See full license in LICENSE.txt.

import copy
import string
import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import simulate

from activitysim.core.util import assign_in_place

import expressions


"""
At this time, these utilities are mostly for transforming the mode choice
spec, which is more complicated than the other specs, into something that
looks like the other specs.
"""


def evaluate_expression_list(expressions, constants):
    """
    Evaluate a list of expressions - each one can depend on the one before
    it.  These are usually used for the coefficients which have relationships
    to each other.  So ivt=.7 and then ivt_lr=ivt*.9.

    Parameters
    ----------
    expressions : Series
        Same as below except the values are accumulated from expression to
        expression and there is no assumed "$" at the beginning.  This is a
        Series because the index are the names of the expressions which are
        used in subsequent evals - thus naming the expressions is required.
        For better or worse, the expressions are assumed to evaluate to
        floats and this is guaranteed by casting to float after eval-ing.
    constants : dict
        will be passed as the scope of eval - usually a separate set of
        constants are passed in here

    Returns
    -------
    expressions : Series

    """
    d = {}
    # this could be a simple expression except that the dictionary
    # is accumulating expressions - i.e. they're not all independent
    # and must be evaluated in order
    for k, v in expressions.iteritems():
        # make sure it can be converted to a float
        result = float(eval(str(v), copy.copy(d), constants))
        d[k] = result

    return pd.Series(d)


def substitute_coefficients(expressions, constants):
    """
    Substitute the named coeffcients in expressions with their numeric values

    Parameters
    ----------
    expressions : Series
        Same as below except there is no assumed "$" at the beginning.
        For better or worse, the expressions are assumed to evaluate to
        floats and this is guaranteed by casting to float after eval-ing.
    constants : dict
        will be passed as the scope of eval - usually a separate set of
        constants are passed in here

    Returns
    -------
    expressions : Series

    """
    return pd.Series([float(eval(e, constants)) for e in expressions], index=expressions.index)


def expand_alternatives(df):
    """
    Alternatives are kept as a comma separated list.  At this stage we need
    need to split them up so that there is only one alternative per row, and
    where an expression is shared among alternatives, that row is copied
    with each alternative alternative value (pun intended) substituted for
    the alternative value for each row.  The DataFrame needs an Alternative
    column which is a comma separated list of alternatives.  See the test for
    an example.
    """

    # first split up the alts using string.split
    alts = [string.split(s, ",") for s in df.reset_index()['Alternative']]

    # this is the number of alternatives in each row
    len_alts = [len(x) for x in alts]

    # this repeats the locs for the number of alternatives in each row
    ilocs = np.repeat(np.arange(len(df)), len_alts)

    # grab the rows the right number of times (after setting a rowid)
    df['Rowid'] = np.arange(len(df))
    df = df.iloc[ilocs]

    # now concat all the lists
    new_alts = sum(alts, [])

    df.reset_index(["Alternative"], inplace=True)
    df["Alternative"] = new_alts
    # rowid needs to be set here - we're going to unstack this and we need
    # a unique identifier to keep track of the rows during the unstack
    df = df.set_index(['Rowid', 'Alternative'], append=True)

    return df


def _mode_choice_spec(mode_choice_spec_df, mode_choice_coeffs, mode_choice_settings,
                      trace_spec=False, trace_label=None):
    """
    Ok we have read in the spec - we need to do several things to reformat it
    to the same style spec that all the other models have.

    mode_choice_spec_df : DataFrame
        This is the actual spec DataFrame, the same as all of the other spec
        dataframes, except that 1) expressions can be prepended with a "$"
        - see pre_process_expressions above 2) There is an Alternative column -
        see expand_alternatives above and 3) there are assumed to be
        expressions in the coefficient column which get evaluated by
        evaluate_expression_list above
    mode_choice_coeffs : DataFrame
        This has the same columns as the spec (columns are assumed to be
        independent segments of the model), and the coefficients (values) in
        the spec DataFrame refer to values in the mode_choice_coeffs
        DataFrame.  The mode_choice_coeffs DataFrame can also contain
        expressions which refer to previous rows in the same column.  Is is
        assumed that all values in mode_choice_coeffs can be cast to float
        after running evaluate_expression_list, and that these floats are
        substituted in multiple place in the mode_choice_spec_df.
    mode_choice_settings : Dict, usually read from YAML
        Has key CONSTANTS which is used as the scope for the evals which
        take place here.

    Returns
    -------
    new_spec_df : DataFrame
        A new spec DataFrame which is exactly like all of the other models.
    """

    trace_label = tracing.extend_trace_label(trace_label, '_mode_choice_spec')

    constants = mode_choice_settings['CONSTANTS']
    df = mode_choice_spec_df

    if trace_spec:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'raw'),
                         slicer='NONE', transpose=False)

    # set index to ['Expression', 'Alternative']
    df = df.set_index('Alternative', append=True)

    if trace_spec:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'pre_process_expressions'),
                         slicer='NONE', transpose=False)

    # for each segment - e.g. eatout vs social vs work vs ...
    for col in df.columns:

        # first the coeffs come as expressions that refer to previous cells
        # as well as constants that come from the settings file
        mode_choice_coeffs[col] = evaluate_expression_list(
            mode_choice_coeffs[col],
            constants=constants)

        # then use the coeffs we just evaluated within the spec (they occur
        # multiple times in the spec which is why they get stored uniquely
        # in a different file
        df[col] = substitute_coefficients(
            df[col],
            mode_choice_coeffs[col].to_dict())

    if trace_spec:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'evaluate_expression_list'),
                         slicer='NONE', transpose=False)

    df = expand_alternatives(df)

    if trace_spec:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'expand_alternatives'),
                         slicer='NONE', transpose=False)

    return df


def get_segment_and_unstack(omnibus_spec, segment):
    """
    This does what it says.  Take the spec, get the column from the spec for
    the given segment, and unstack.  It is assumed that the last column of
    the multiindex is alternatives so when you do this unstacking,
    each alternative is in a column (which is the format this as used for the
    simple_simulate call.  The weird nuance here is the "Rowid" column -
    since many expressions are repeated (e.g. many are just "1") a Rowid
    column is necessary to identify which alternatives are actually part of
    which original row - otherwise the unstack is incorrect (i.e. the index
    is not unique)
    """
    spec = omnibus_spec[segment].unstack().reset_index(level="Rowid", drop=True).fillna(0)

    spec = spec.groupby(spec.index).sum()

    return spec


def mode_choice_simulate(
        records,
        skims,
        spec,
        constants,
        nest_spec,
        chunk_size,
        trace_label=None, trace_choice_name=None):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    locals_dict = skims.copy()
    if constants is not None:
        locals_dict.update(constants)

    choices = simulate.simple_simulate(
        records,
        spec,
        nest_spec,
        skims=list(skims.values()),
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name)

    alts = spec.columns
    choices = choices.map(dict(zip(range(len(alts)), alts)))

    return choices


def annotate_preprocessors(
        tours_df, locals_dict, skims,
        model_settings, trace_label):

    locals_d = {}
    locals_d.update(locals_dict)
    locals_d.update(skims)

    annotations = []

    preprocessor_settings = model_settings.get('preprocessor_settings', [])
    if not isinstance(preprocessor_settings, list):
        assert isinstance(preprocessor_settings, dict)
        preprocessor_settings = [preprocessor_settings]

    simulate.add_skims(tours_df, list(skims.values()))

    annotations = None
    for model_settings in preprocessor_settings:

        results = expressions.compute_columns(
            df=tours_df,
            model_settings=model_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

        assign_in_place(tours_df, results)

        if annotations is None:
            annotations = results
        else:
            annotations = pd.concat([annotations, results], axis=1)

    return annotations
