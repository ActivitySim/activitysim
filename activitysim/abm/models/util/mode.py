# ActivitySim
# See full license in LICENSE.txt.

import copy
import string
import pandas as pd
import numpy as np

from activitysim.core import tracing


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


def pre_process_expressions(expressions, variable_templates):
    """
    This one is pretty simple - pass in a list of expressions which contain
    references to templates and pass a dictionary of the templates themselves.
    Strings will only be evaluated which are prepended with $.

    Parameters
    ----------
    expressions : list of strs
        These are the expressions that will be evaluated - generally these
        contain templates that get passed below.  So will be something like
        ['$SKIM_TEMPLATE.format(sk="AMPEAK")']
    variable_templates : dict of templates
        Will be passed as the scope of eval.  Keys are usually template names
        and values are strings.  The dict could be something like
        {'SKIM_TEMPLATE': 'skims[{sk}]'}

    Returns
    -------
    expressions : list of strs
        Each expression is evaluated with variable_templates in the scope and
        the result is returned.
    """
    return [eval(e[1:], variable_templates) if e.startswith('$') else e for
            e in expressions]


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


def _mode_choice_spec(mode_choice_spec_df, mode_choice_coeffs,
                      mode_choice_settings, trace_label=None):
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
        Has two values which are used.  One key in CONSTANTS which is used as
        the scope for the evals which take place here and one that is
        VARIABLE_TEMPLATES which is used as the scope for expressions in
        mode_choice_spec_df which are prepended with "$"

    Returns
    -------
    new_spec_df : DataFrame
        A new spec DataFrame which is exactly like all of the other models.
    """

    trace_label = tracing.extend_trace_label(trace_label, '_mode_choice_spec')

    constants = mode_choice_settings['CONSTANTS']
    templates = mode_choice_settings['VARIABLE_TEMPLATES']
    df = mode_choice_spec_df
    index_name = df.index.name

    if trace_label:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'raw'),
                         slicer='NONE', transpose=False)

    # FIXME - this is no longer used and should probably be removed
    # the expressions themselves can be prepended with a "$" in order to use
    # model templates that are shared by several different expressions
    df.index = pre_process_expressions(df.index, templates)
    df.index.name = index_name

    # set index to ['Expression', 'Alternative']
    df = df.set_index('Alternative', append=True)

    if trace_label:
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

    if trace_label:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'evaluate_expression_list'),
                         slicer='NONE', transpose=False)

    df = expand_alternatives(df)

    if trace_label:
        tracing.trace_df(df,
                         tracing.extend_trace_label(trace_label, 'expand_alternatives'),
                         slicer='NONE', transpose=False)

    return df
